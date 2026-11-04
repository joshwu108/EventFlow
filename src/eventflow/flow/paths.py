"""Conditional probability paths for flow matching.

We use the optimal-transport conditional path of Lipman et al. (2023),
"Flow Matching for Generative Modeling", eq. 20-23. With source sample
``x_0 ~ N(0, I)`` and data sample ``x_1``::

    x_t = (1 - (1 - sigma_min) t) x_0 + t x_1,          t in [0, 1]
    u_t = d x_t / dt = x_1 - (1 - sigma_min) x_0

``u_t`` is the conditional target vector field. Regressing a network
``v_theta(x_t, t, c)`` onto ``u_t`` (the CFM loss) has the same gradients as
regressing onto the intractable marginal field, whose flow transports
``N(0, I)`` at ``t = 0`` to (a ``sigma_min``-smoothed) data distribution at
``t = 1``. With ``sigma_min = 0`` this is the rectified-flow / linear
interpolant path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OTPath:
    sigma_min: float = 1e-4

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - (1 - self.sigma_min) * t) * x0 + t * x1

    def target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - (1 - self.sigma_min) * x0


def sample_times(
    n: int,
    kind: str,
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Training times ``t in (0, 1)``.

    ``uniform`` is the original CFM choice. ``logit_normal`` (Esser et al.,
    2024) concentrates training on intermediate times, where the regression
    target is hardest.
    """
    if kind == "uniform":
        return torch.rand(n, device=device, generator=generator)
    if kind == "logit_normal":
        return torch.sigmoid(torch.randn(n, device=device, generator=generator))
    raise ValueError(f"unknown time sampling {kind!r}")
