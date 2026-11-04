"""Reusable neural-network layers shared by the encoder and the output heads."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Transformer-style sinusoidal features of a scalar per row.

    ``t`` has shape ``[N]`` (any real scale; diffusion passes integer steps,
    flow matching passes ``1000 * t`` for ``t in [0, 1]``). Returns ``[N, dim]``.
    """
    if dim % 2:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal features followed by a two-layer MLP (as in DDPM / DiT)."""

    def __init__(self, dim: int, frequency_dim: int = 128):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.frequency_dim))


class AdaLNResBlock(nn.Module):
    """Residual MLP block with adaptive layer norm (shift, scale, gate) from a condition.

    The modulation is zero-initialised so each block starts as the identity,
    which keeps deep denoisers stable at initialisation (Peebles & Xie, DiT).
    """

    def __init__(self, width: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * width))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class AdaLNMLP(nn.Module):
    """Conditional MLP ``f(x, cond) -> out`` used as denoiser / vector-field network.

    Mirrors the small AdaLN MLP used as the per-token diffusion head in
    Li et al. (2024), "Autoregressive Image Generation without Vector
    Quantization"; reimplemented here, not copied.
    """

    def __init__(self, in_dim: int, cond_dim: int, width: int, depth: int, out_dim: int):
        super().__init__()
        self.input = nn.Linear(in_dim, width)
        self.cond = nn.Linear(cond_dim, width)
        self.blocks = nn.ModuleList(AdaLNResBlock(width, width) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, out_dim)
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        # Zero output so the initial prediction is 0 (noise/velocity of mean zero).
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond(cond)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        return self.output(self.final_norm(h) * (1 + scale) + shift)


class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, width), nn.GELU(), nn.Dropout(dropout)]
            dim = width
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
