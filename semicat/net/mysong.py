# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import silu

from semicat.jvp_utils.functional import safe_sdpa_jvp


class WrapGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__(
            min(num_groups, num_channels // min_channels_per_group),
            num_channels,
            eps=eps,
        )


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_proj=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        # self.skip_scale = skip_scale
        self.register_buffer("skip_scale", torch.tensor(skip_scale, dtype=torch.float32), persistent=False)

        self.norm0 = WrapGroupNorm(num_channels=in_channels, eps=eps)
        if up:
            self.conv0 = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        elif down:
            self.conv0 = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            self.conv0 = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        self.affine = nn.Linear(
            in_features=emb_channels,
            out_features=out_channels,
        )
        self.norm1 = WrapGroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        self.skip = None
        if (out_channels != in_channels or up or down) and (resample_proj or out_channels != in_channels):
            if up:
                self.skip = nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    output_padding=1,
                )
            elif down:
                self.skip = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                )
            else:
                self.skip = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                )

        if self.num_heads:
            self.norm2 = WrapGroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel_size=1,
            )
            self.proj = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, emb, jvp_attention: bool = False):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(-1)
        x = silu(self.norm1(x.add(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0], self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(3)
            )
            # w = AttentionOp.apply(q, k)
            # a = torch.einsum("nqk,nck->ncq", w, v)
            if jvp_attention:
                q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
                a = safe_sdpa_jvp(q, k, v)
            else:
                a = F.scaled_dot_product_attention(q, k, v)

            x = self.proj(a.reshape(*x.shape)).add(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer("freqs", freqs[None, ...], persistent=False)

    def forward(self, x):
        x = x[..., None] * self.freqs
        s = x.sin()
        c = x.cos()
        return torch.stack((s, c), dim=-1).view(x.shape[0], -1)


class MySongUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        # out_channels,  # Number of color channels at output.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.1,  # Dropout probability of intermediate activations.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
    ):
        super().__init__()
        out_channels = in_channels

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_proj=True,
        )

        # Mapping.
        self.s_map = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        )
        self.t_map = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        )
        self.embed_s = torch.nn.Sequential(
            nn.Linear(noise_channels, emb_channels),
            torch.nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            torch.nn.SiLU(),
        )
        self.embed_t = torch.nn.Sequential(
            nn.Linear(noise_channels, emb_channels),
            torch.nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            torch.nn.SiLU(),
        )

        # Encoder.
        self.enc = nn.ModuleList()
        cout = in_channels
        self.pre_proj = nn.Conv1d(in_channels=cout, out_channels=model_channels, kernel_size=3, padding=1)

        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
            else:
                self.enc.append(
                    UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                )

            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc.append(
                    UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
                )
        skips = [self.pre_proj.out_channels] + [block.out_channels for block in self.enc]

        # Decoder.
        self.dec = torch.nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec.append(
                    UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                )
                self.dec.append(
                    UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                )
            else:
                self.dec.append(
                    UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec.append(
                    UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
                )
            if level == 0:
                continue

        self.fin_norm = WrapGroupNorm(num_channels=cout, eps=1e-6)
        self.fin_conv = nn.Conv1d(in_channels=cout, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, s, t, jvp_attention: bool = False):
        # reparam
        t = t - s
        # swap classes and sequence positions
        x = x.transpose(1, 2)

        # Embedding t
        emb_t = self.t_map(t)
        emb_t = self.embed_t(emb_t)
        # Embedding s
        emb_s = self.s_map(s)
        emb_s = self.embed_s(emb_s)
        emb = emb_t + emb_s

        # Encoder.
        x = self.pre_proj(x)
        skips = [x]

        for block in self.enc:
            x = block(x, emb, jvp_attention=jvp_attention)
            skips.append(x)

        # Decoder.
        for block in self.dec:
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb, jvp_attention=jvp_attention)

        tmp = self.fin_norm(x)
        tmp = self.fin_conv(silu(tmp))

        return tmp.transpose(1, 2)
