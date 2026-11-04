"""Conditional flow-matching head for the continuous fields of one event.

Training (masked CFM loss)::

    L(x_1 | c) = E_{t, x_0} [ (1/|m|) sum_d m_d (v_theta(x_t, t, c)_d - u_t,d)^2 ]

with ``x_t`` and ``u_t`` from :class:`~eventflow.flow.paths.OTPath` and the
continuous mask ``m`` as in the DDPM head. Sampling draws ``x_0 ~ N(0, I)``
(scaled by ``temperature``) and integrates ``dx/dt = v_theta(x, t, c)`` from
``t = 0`` to ``t = 1`` with a fixed-step solver.

Masked dimensions of ``x_1`` are set to zero, so for those dimensions the
network learns the (well-defined) flow towards a point mass at zero; the
generated values are ignored downstream.
"""

from __future__ import annotations

import torch
from torch import nn

from eventflow.flow.paths import OTPath, sample_times
from eventflow.flow.solvers import integrate
from eventflow.models.config import FlowConfig
from eventflow.models.layers import AdaLNMLP, TimestepEmbedding

TIME_SCALE = 1000.0  # sinusoidal features expect step-like magnitudes


class FlowMatchingHead(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        dim: int,
        config: FlowConfig,
        width: int,
        depth: int,
        loss_repeats: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.config = config
        self.loss_repeats = loss_repeats
        self.path = OTPath(config.sigma_min)
        self.time_embedding = TimestepEmbedding(width)
        self.cond_projection = nn.Linear(cond_dim, width)
        self.network = AdaLNMLP(dim, width, width, depth, dim)

    def velocity(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.network(x, self.cond_projection(cond) + self.time_embedding(t * TIME_SCALE))

    def loss(
        self,
        cond: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r = self.loss_repeats
        cond_r, x_r, mask_r = (v.repeat_interleave(r, dim=0) for v in (cond, x, mask))
        maskf = mask_r.float()
        x1 = x_r * maskf
        x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
        t = sample_times(
            x1.shape[0], self.config.time_sampling, device=x1.device, generator=generator
        )
        x_t = self.path.interpolate(x0, x1, t)
        err = (self.velocity(x_t, t, cond_r) - self.path.target(x0, x1)).pow(2) * maskf
        per_row = err.sum(-1) / maskf.sum(-1).clamp_min(1.0)
        return per_row.view(-1, r).mean(-1)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        steps: int | None = None,
    ) -> torch.Tensor:
        x0 = temperature * torch.randn(
            (cond.shape[0], self.dim), device=cond.device, generator=generator
        )
        return integrate(
            lambda x, t: self.velocity(x, t, cond),
            x0,
            steps=steps or self.config.sample_steps,
            method=self.config.solver,
        )
