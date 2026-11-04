"""DDPM noise schedules, forward corruption and the Gaussian posterior.

Notation (Ho et al., 2020): ``alpha_t = 1 - beta_t`` and
``abar_t = prod_{s<=t} alpha_s`` for steps ``t = 0..T-1``. The forward
process is::

    q(x_t | x_0) = N( sqrt(abar_t) x_0, (1 - abar_t) I )

and the reverse posterior used for ancestral sampling is::

    q(x_{t-1} | x_t, x_0) = N( c0_t x_0 + ct_t x_t, beta~_t I )
    c0_t    = beta_t sqrt(abar_{t-1}) / (1 - abar_t)
    ct_t    = (1 - abar_{t-1}) sqrt(alpha_t) / (1 - abar_t)
    beta~_t = beta_t (1 - abar_{t-1}) / (1 - abar_t)

with ``abar_{-1} = 1``. Coefficients are computed in float64 and stored as
float32 buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from eventflow.errors import ConfigError

MAX_BETA = 0.999  # Nichol & Dhariwal (2021): avoids singular final steps of the cosine schedule.


def linear_betas(steps: int, start: float = 1e-4, end: float = 0.02) -> torch.Tensor:
    """Ho et al. linear schedule, rescaled so any ``steps`` matches the 1000-step endpoints."""
    scale = 1000.0 / steps
    return torch.linspace(start * scale, min(end * scale, MAX_BETA), steps, dtype=torch.float64)


def cosine_betas(steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule: ``abar(t) = cos^2(((t/T)+s)/(1+s) * pi/2)``."""
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = 1 - abar[1:] / abar[:-1]
    return betas.clamp(max=MAX_BETA)


@dataclass(frozen=True)
class NoiseSchedule:
    """Precomputed schedule tensors (all ``[T]``)."""

    betas: torch.Tensor
    timesteps: torch.Tensor  # original training step index of each (possibly respaced) step

    @classmethod
    def create(cls, kind: str, steps: int) -> NoiseSchedule:
        if kind == "linear":
            betas = linear_betas(steps)
        elif kind == "cosine":
            betas = cosine_betas(steps)
        else:
            raise ConfigError(f"unknown noise schedule {kind!r}")
        return cls(betas, torch.arange(steps))

    def __post_init__(self) -> None:
        if torch.any(self.betas <= 0) or torch.any(self.betas >= 1):
            raise ConfigError("betas must lie in (0, 1)")

    @property
    def num_steps(self) -> int:
        return int(self.betas.shape[0])

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return torch.cumprod(1 - self.betas.double(), dim=0)

    def respace(self, num_steps: int) -> NoiseSchedule:
        """Schedule over an evenly spaced subset of steps (Nichol & Dhariwal, sec. 4).

        Keeps ``abar`` at the retained steps exactly, so a model trained on
        the full chain can be sampled with fewer steps.
        """
        if not 1 <= num_steps <= self.num_steps:
            raise ConfigError(f"cannot respace {self.num_steps} steps to {num_steps}")
        keep = torch.linspace(0, self.num_steps - 1, num_steps).round().long().unique()
        abar = self.alphas_cumprod[keep]
        previous = torch.cat([torch.ones(1, dtype=abar.dtype), abar[:-1]])
        betas = (1 - abar / previous).clamp(min=1e-12, max=MAX_BETA)
        return NoiseSchedule(betas, self.timesteps[keep])

    def _gather(self, values: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        out = values.to(like.device)[t].to(like.dtype)
        return out.view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        abar = self.alphas_cumprod
        return (
            self._gather(abar.sqrt(), t, x0) * x0 + self._gather((1 - abar).sqrt(), t, x0) * noise
        )

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Invert ``q_sample`` given a noise estimate: ``(x_t - sqrt(1-abar) eps) / sqrt(abar)``.

        At high noise ``sqrt(abar)`` is tiny, so noise-estimate errors are
        amplified by up to ``1/sqrt(abar_T)``; samplers clip this estimate.
        """
        abar = self.alphas_cumprod
        return (x_t - self._gather((1 - abar).sqrt(), t, x_t) * eps) / self._gather(
            abar.sqrt(), t, x_t
        )

    def posterior(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of ``q(x_{t-1} | x_t, x_0)``."""
        abar = self.alphas_cumprod
        prev = torch.cat([torch.ones(1, dtype=abar.dtype), abar[:-1]])
        betas = self.betas.double()
        c0 = betas * prev.sqrt() / (1 - abar)
        ct = (1 - prev) * (1 - betas).sqrt() / (1 - abar)
        var = betas * (1 - prev) / (1 - abar)
        mean = self._gather(c0, t, x0) * x0 + self._gather(ct, t, x_t) * x_t
        return mean, self._gather(var, t, x_t)
