import functools
import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from semicat.jvp_utils.functional import sdpa_jvp


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    """
    NOTE: Modified Rotary caching: when using this model (DIT) with JVP,
    Rotary is called as well, of course. The previously cached tensors as
    ```
    def __init__(self, dim, base=10_000):
        ...
        self.sin_cached = ...
    ```
    later updated in the first forward pass are wrapped in ~GradTrackingTensors
    because of JVP, and because of that cannot be serialised automatically,
    so it causes issues when checkpointing. This seems to work perfectly fine.
    """

    def __init__(self, dim, base=10_000):
        super().__init__()
        # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.base = base
        self.dim = dim

    @functools.lru_cache(maxsize=1)
    def _compute_rotary(self, seq_len, device):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        # dims are: batch, seq_len, qkv, head, dim
        cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        # This makes the transformation on v an identity.
        cos_cached[:, :, 2, :, :].fill_(1.0)
        sin_cached[:, :, 2, :, :].fill_(0.0)
        return cos_cached, sin_cached

    @torch.no_grad()
    def forward(self, x, seq_dim=1):
        return self._compute_rotary(x.shape[seq_dim], x.device)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
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


@torch.jit.script
def regular_attention_multi_headed(q, k, v):
    # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.matmul(attn_weight, v).reshape(q.size(0), q.size(1), -1)

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
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


class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=qkv.device,
        )
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True
        )

        x = einops.rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN, cond_dim=None, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c=None, jvp_attention: bool = False):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            # self.adaLN_modulation(c): (128, 1536)
            # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
            # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            )
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = einops.rearrange(
            self.attn_qkv(x),
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

        if False:
            assert {q.dtype, k.dtype, v.dtype} <= {torch.bfloat16, torch.float16}, "JVP attention only works with bfloat16 or float16"
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = sdpa_jvp(q, k, v)
        else:
            x = regular_attention_multi_headed(q, k, v)

        if self.adaLN:
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, gate_msa, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None,
                gate_mlp,
                x,
                self.dropout,
            )
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout
            )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        # torch.nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            x,  #torch.nn.functional.softmax(x, dim=-1).float(),
            self.embedding.float(),
        ).to(x.dtype)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        # self.linear.weight.data.zero_()
        # self.linear.bias.data.zero_()
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


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        causal_attention: bool = False,
    ):
        super().__init__()
        self.causal = causal_attention
        self.adaLN = not self.causal
        self.vocab_size = vocab_size
        dim = hidden_size
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
            self.t_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // n_heads)

        blocks = []
        for _ in range(n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim, n_heads=n_heads, dropout=dropout
                )
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=dropout,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )
        # do not init weights except for the new ones for t_map:
        # the rest is loaded from checkpoint
        # self._init_weights()

    def reset_time_weights(self):
        nn.init.constant_(self.t_map.mlp[0].weight, 0.0)
        nn.init.constant_(self.t_map.mlp[0].bias, 0)
        nn.init.constant_(self.t_map.mlp[2].weight, 0.0)
        nn.init.constant_(self.t_map.mlp[2].bias, 0)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def _init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)

        self.apply(_basic_init)
        # timestep embedders
        nn.init.normal_(self.t_map.mlp[0].weight, std=0.01)
        nn.init.constant_(self.t_map.mlp[0].bias, 0)
        nn.init.normal_(self.t_map.mlp[2].weight, std=0.01)
        nn.init.constant_(self.t_map.mlp[2].bias, 0)
        nn.init.normal_(self.sigma_map.mlp[0].weight, std=0.01)
        nn.init.constant_(self.sigma_map.mlp[0].bias, 0)
        nn.init.normal_(self.sigma_map.mlp[2].weight, std=0.01)
        nn.init.constant_(self.sigma_map.mlp[2].bias, 0)
        # final layer
        nn.init.normal_(self.output_layer.linear.weight, std=1e-3)

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, jvp_attention: bool = False) -> torch.Tensor:
        s = s.view(-1)
        t = t.view(-1)
        # reparam:
        t = t - s
        x = self.vocab_embed(x)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(s)) + F.silu(self.t_map(t))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond, jvp_attention=jvp_attention)
            x = self.output_layer(x, c=t_cond)

        return x
