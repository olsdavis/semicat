"""
Changed architecture from Duo to be able to edit.
"""

import math
from typing import Literal

# import einops
import flash_attn
import flash_attn.layers.rotary
import torch
import torch.nn as nn
import torch.nn.functional as F

from semicat.jvp_utils.functional import safe_sdpa_jvp


def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class Rotary(torch.nn.Module):
    def __init__(self, seq_len, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # dims are: batch, seq_len, qkv, head, dim
        cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        # This makes the transformation on v an identity.
        cos_cached[:, :, 2, :, :].fill_(1.0)
        sin_cached[:, :, 2, :, :].fill_(0.0)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self):
        return self.cos_cached, self.sin_cached


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    cos, sin = rotary_cos_sin
    #cos = cos.to(qkv.dtype)
    #sin = sin.to(qkv.dtype)
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
    return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half)
            / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None] * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN, seq_len, cond_dim=None, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN
        self.dim = dim
        self.dim_per_head = dim // n_heads
        self.seq_len = seq_len

        self.norm1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = nn.Dropout(p=dropout)

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c=None, jvp_attention=False):
        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            )
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x).reshape(x.shape[0], self.seq_len, 3, self.n_heads, self.dim_per_head)
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

        if jvp_attention:
            x = safe_sdpa_jvp(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            attention_output = F.scaled_dot_product_attention(
                query=q.transpose(1, 2),
                key=k.transpose(1, 2),
                value=v.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            # [batch_size, seq_len, num_heads, head_dim]
            x = attention_output.transpose(1, 2)

        # B, S, (H D)
        x = x.reshape(x.shape[0], self.seq_len, self.dim)

        if self.adaLN:
            x = self.dropout(self.attn_out(x) * gate_msa) + x_skip
            # x = self.dropout(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp) + x
            x_skip = x
            x = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
            x = self.dropout(self.mlp(x) * gate_mlp) + x_skip
        else:
            x = self.dropout(self.attn_out(x)) + x_skip
            x = self.dropout(self.mlp(self.norm2(x))) + x
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        # self.layer_norm = nn.LayerNorm(vocab_dim)
        self.seq = nn.Linear(vocab_dim, dim)
        self._coeff = vocab_dim ** 0.5

    def forward(self, x, t, c):
        # t and c are ignored but are not in the other version
        # x = self.layer_norm(x)
        x = x / self._coeff
        return self.seq(x)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class RMSEmbeddingLayer(nn.Module):
    """
    Replacement input embedding for noisy continuous/discrete mixture inputs xt.
    - RMS-based time scaling
    - shallow MLP with small residual
    - FiLM conditioning on `cond` (expected shape (B, d_model))
    - LayerNorm on output hidden dim
    NOTE: x expected shape (B, L, vocab_dim).
    """

    def __init__(
        self,
        vocab_dim: int,
        d_model: int,
        cond_dim: int,
        hidden_dim: int | None = None,
        sigma0: float = 1.0,
        eps: float = 1e-6,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (d_model * 2)
        self.vocab_dim = vocab_dim
        self.d_model = d_model
        self.sigma0 = float(sigma0)
        self.eps = float(eps)
        self.residual_scale = float(residual_scale)

        # projection from vocab-space -> model dim
        self.proj = nn.Linear(vocab_dim, d_model, bias=True)

        # small MLP in hidden space
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, d_model),
        )

        # FiLM (time/scale conditioning) if cond_dim provided
        self.cond_dim = cond_dim
        self.film_gamma = nn.Linear(cond_dim, d_model, bias=True)
        self.film_beta  = nn.Linear(cond_dim, d_model, bias=True)

        self.final_norm = nn.RMSNorm(d_model, eps=1e-8)

        # init recommended for stability
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        """
        x: (B, L, V)
        t: (B,)
        returns: (B, L, d_model)
        """
        B, L, V = x.shape

        # expected squared-norm of x_t per sample:
        # E||x_t||^2 = t^2 * ||one-hot||^2 + (1-t)^2 * D * sigma0^2
        # for one-hot ||one-hot||^2 == 1
        exp_norm2 = t**2 + (1.0 - t)**2 * (self.vocab_dim * (self.sigma0**2))
        scale = 1.0 / torch.sqrt(exp_norm2 + self.eps)          # (B,)

        # broadcast scale to (B, L, V)
        scale = scale.view(B, 1, 1)
        x_scaled = x * scale

        # linear projection
        h = self.proj(x_scaled)          # (B, L, d_model)

        # small residual MLP to add nonlinearity / capacity
        h = h + self.residual_scale * self.mlp(h)

        # FiLM conditional modulation
        # cond expected (B, cond_dim) -> expand to (B,1,d_model)
        gamma = self.film_gamma(cond).unsqueeze(1)  # (B,1,d_model)
        beta  = self.film_beta(cond).unsqueeze(1)
        h = (1.0 + gamma) * h + beta

        # final normalization
        h = self.final_norm(h)
        return h


class DIT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        length: int,
        embed_type: Literal["naive", "rms"] = "naive",
    ):
        super().__init__()
        self.adaLN = True
        self.vocab_size = vocab_size
        if embed_type == "naive":
            self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        elif embed_type == "rms":
            self.vocab_embed = RMSEmbeddingLayer(vocab_size, hidden_size, cond_dim)
        else:
            raise ValueError(f"illegal 'embed_type': {embed_type}")

        self.s_map = TimestepEmbedder(cond_dim)
        self.t_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(length, hidden_size // n_heads)

        blocks = []
        for _ in range(n_blocks):
            block = DDiTBlock(
                dim=hidden_size,
                n_heads=n_heads,
                cond_dim=cond_dim,
                adaLN=self.adaLN,
                dropout=dropout,
                seq_len=length,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=hidden_size,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )

    def forward(self, x, s, t, jvp_attention: bool = False):
        # time reparameterisation
        t = t - s

        s_cond = F.silu(self.s_map(s))
        t_cond = F.silu(self.t_map(t))
        cond = s_cond + t_cond
        x = self.vocab_embed(x, s, cond)

        rotary_cos_sin = self.rotary_emb()

        for b in self.blocks:
            x = b(x, rotary_cos_sin, c=cond, jvp_attention=jvp_attention)
        x = self.output_layer(x, c=cond)

        return x
