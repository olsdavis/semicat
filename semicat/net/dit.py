# From https://github.com/lumalabs/tvm/blob/main/training/dit.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Mlp

from einops import rearrange

from semicat.jvp_utils.functional import safe_sdpa_jvp


class Attention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, use_sdpa_jvp=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = (
            qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # q,k,v: (B, H, L, D)
        if use_sdpa_jvp:
            x = safe_sdpa_jvp(q, k, v)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def rmsnorm_noweight(input, eps=1e-6):
    return input * torch.rsqrt(torch.mean(input**2, dim=-1, keepdim=True) + eps)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        #torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        torch.nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            # TODO: ablate?
            # torch.nn.functional.softmax(x, dim=-1).float(),
            x,
            self.embedding,
        )


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_timestep_embedding(t, dim, max_period=10000):
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
            * torch.arange(start=0, end=half, dtype=torch.float64)
            / half
        ).to(device=t.device)
        args = t[:, None].to(torch.float64) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.positional_timestep_embedding(t, self.frequency_embedding_size)

        t_emb = self.mlp(t_freq.to(dtype=t.dtype))
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        temb_size,
        mlp_ratio=4.0,
        dropout=0,
        do_mod_norm=True,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(temb_size, 6 * hidden_size)
        )
        self.do_mod_norm = do_mod_norm

    def forward(self, x, c, use_sdpa_jvp=False):
        adaln_out = self.adaLN_modulation(c)

        if self.do_mod_norm:
            adaln_out = rmsnorm_noweight(rearrange(adaln_out, "b (n d) -> b n d", n=6))
            adaln_out = rearrange(adaln_out, "b n d -> b (n d)")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaln_out.chunk(6, dim=1)
        )

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), use_sdpa_jvp=use_sdpa_jvp
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, in_size, do_mod_norm=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Linear(hidden_size, in_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.do_mod_norm = do_mod_norm

    def forward(self, x, c):
        adaln_out = self.adaLN_modulation(c)
        if self.do_mod_norm:
            adaln_out = rmsnorm_noweight(rearrange(adaln_out, "b (n d) -> b n d", n=2))
            adaln_out = rearrange(adaln_out, "b n d -> b (n d)")
        shift, scale = adaln_out.chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_dim,
        sequence_length,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        embedding_kwargs={},
        temb_mult=1,
        dropout=0,
        qk_norm=True,
        learn_guidance=False,
        do_mod_norm=True,
        init_type="spectral",
        time_specinit=False,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.learn_guidance = learn_guidance

        self.x_proj = EmbeddingLayer(hidden_size, in_dim)

        temb_size = hidden_size * temb_mult

        self.s_embedder = TimestepEmbedder(temb_size, **embedding_kwargs)
        self.guidance_embedder = (
            TimestepEmbedder(temb_size, **embedding_kwargs) if learn_guidance else None
        )

        self.t_embedder = TimestepEmbedder(temb_size, **embedding_kwargs)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, sequence_length, hidden_size), requires_grad=False
        )

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.sequence_length**0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    temb_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    do_mod_norm=do_mod_norm,
                )
                for _ in range(depth)
            ]
        )
        self.init_type = init_type
        self.time_specinit = time_specinit
        self.final_layer = FinalLayer(
            hidden_size, in_dim, do_mod_norm=do_mod_norm
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if self.init_type == "xunif":
                    nn.init.xavier_uniform_(module.weight)
                elif self.init_type == "spectral":
                    nn.init.xavier_normal_(module.weight)
                    u, s, v = torch.svd(module.weight)
                    module.weight.data = 1.0 * module.weight.data / s[0]

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.RMSNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # if self.init_type == "spectral":
        #     nn.init.xavier_normal_(w.view([w.shape[0], -1]))
        #     u, s, v = torch.svd(w.view([w.shape[0], -1]))
        #     w.data = 1.0 * w.data / s[0]
        # else:
        #     nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        if self.t_embedder:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[-1].weight, std=0.02)
            if self.init_type == "spectral" and self.time_specinit:
                nn.init.xavier_normal_(self.t_embedder.mlp[0].weight)
                nn.init.xavier_normal_(self.t_embedder.mlp[-1].weight)
                u, s, v = torch.svd(self.t_embedder.mlp[0].weight)
                self.t_embedder.mlp[0].weight.data = (
                    1.0 * self.t_embedder.mlp[0].weight.data / s[0]
                )
                u, s, v = torch.svd(self.t_embedder.mlp[-1].weight)
                self.t_embedder.mlp[-1].weight.data = (
                    1.0 * self.t_embedder.mlp[-1].weight.data / s[0]
                )

        if self.learn_guidance:
            nn.init.normal_(self.guidance_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.guidance_embedder.mlp[-1].weight, std=0.02)
            if self.init_type == "spectral" and self.time_specinit:
                nn.init.xavier_normal_(self.guidance_embedder.mlp[0].weight)
                nn.init.xavier_normal_(self.guidance_embedder.mlp[-1].weight)
                u, s, v = torch.svd(self.guidance_embedder.mlp[0].weight)
                self.guidance_embedder.mlp[0].weight.data = (
                    1.0 * self.guidance_embedder.mlp[0].weight.data / s[0]
                )
                u, s, v = torch.svd(self.guidance_embedder.mlp[-1].weight)
                self.guidance_embedder.mlp[-1].weight.data = (
                    1.0 * self.guidance_embedder.mlp[-1].weight.data / s[0]
                )

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.s_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.s_embedder.mlp[-1].weight, std=0.02)
        if self.init_type == "spectral" and self.time_specinit:
            nn.init.xavier_normal_(self.s_embedder.mlp[0].weight)
            nn.init.xavier_normal_(self.s_embedder.mlp[-1].weight)
            u, s, v = torch.svd(self.s_embedder.mlp[0].weight)
            self.s_embedder.mlp[0].weight.data = (
                1.0 * self.s_embedder.mlp[0].weight.data / s[0]
            )
            u, s, v = torch.svd(self.s_embedder.mlp[-1].weight)
            self.s_embedder.mlp[-1].weight.data = (
                1.0 * self.s_embedder.mlp[-1].weight.data / s[0]
            )

        # Zero-out adaLN modulation layers in DiT blocks:
        #for block in self.blocks:
        #    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x,
        s,
        t,
        jvp_attention=False,
    ):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        use_sdpa_jvp = jvp_attention
        noise_labels_t = t
        noise_labels_ts = s

        x = self.x_proj(x) + self.pos_embed  # (N, T, D)

        t = self.t_embedder(noise_labels_t)  # (N, D)
        ts = self.s_embedder(noise_labels_ts)

        t = t + ts

        # y = self.y_embedder(y, self.training)  # (N, D)
        c = t  # + y  # (N, D)

        for block in self.blocks:
            x = block(x, c, use_sdpa_jvp=use_sdpa_jvp)  # (N, T, D)

        x = self.final_layer(x, c)  # (N, T, in_dim)

        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
