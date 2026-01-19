"""
Changed architecture from Duo to be able to edit.
"""

import math

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


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


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
            x = self.dropout(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp) + x
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

    def forward(self, x):
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
    ):
        super().__init__()
        self.adaLN = True
        self.vocab_size = vocab_size
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
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

        x = self.vocab_embed(x)
        s_cond = F.silu(self.s_map(s))
        t_cond = F.silu(self.t_map(t))
        cond = s_cond + t_cond

        rotary_cos_sin = self.rotary_emb()

        for b in self.blocks:
            x = b(x, rotary_cos_sin, c=cond, jvp_attention=jvp_attention)
        x = self.output_layer(x, c=cond)

        return x
