"""Conditional DDPM head for the continuous fields of one event.

Training objective (epsilon prediction, Ho et al. 2020, "simple" loss)::

    L(x_0 | c) = E_{t, eps} [ (1/|m|) sum_d m_d (eps_theta(x_t, t, c)_d - eps_d)^2 ]

where ``m`` is the continuous mask: undefined dimensions (e.g. the log gap
of a zero-gap event) are set to zero in ``x_0`` and excluded from the loss,
so the network is never asked to denoise a placeholder. Each conditioning
vector is reused ``loss_repeats`` times with independent ``(t, eps)``, which
reduces gradient variance cheaply because the encoder runs once (the
"diffusion batch multiplier" of Li et al. 2024).

Only continuous fields are diffused. Categorical fields are handled by the
softmax chain (factorised mode) or, in relaxed mode, embedded as ``+-1``
one-hot vectors -- see :mod:`eventflow.models.config`.
"""

from __future__ import annotations

import torch
from torch import nn

from eventflow.diffusion.schedule import NoiseSchedule
from eventflow.models.config import DiffusionConfig
from eventflow.models.layers import AdaLNMLP, TimestepEmbedding


class DDPMHead(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        dim: int,
        config: DiffusionConfig,
        width: int,
        depth: int,
        loss_repeats: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.config = config
        self.loss_repeats = loss_repeats
        self.schedule = NoiseSchedule.create(config.schedule, config.train_steps)
        self.sampling_schedule = self.schedule.respace(config.sample_steps)
        self.time_embedding = TimestepEmbedding(width)
        self.cond_projection = nn.Linear(cond_dim, width)
        self.denoiser = AdaLNMLP(dim, width, width, depth, dim)

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``eps_theta``; ``t`` holds *original* training step indices."""
        return self.denoiser(x_t, self.cond_projection(cond) + self.time_embedding(t))

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
        x0 = x_r * maskf
        t = torch.randint(
            0, self.schedule.num_steps, (x0.shape[0],), device=x0.device, generator=generator
        )
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        x_t = self.schedule.q_sample(x0, t, noise)
        err = (self.predict_noise(x_t, t, cond_r) - noise).pow(2) * maskf
        per_row = err.sum(-1) / maskf.sum(-1).clamp_min(1.0)
        return per_row.view(-1, r).mean(-1)

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, generator: torch.Generator | None = None, temperature: float = 1.0
    ) -> torch.Tensor:
        """Ancestral sampling over the respaced chain.

        Numerical safeguards: the clean estimate ``x0_hat`` is clipped to
        ``+-x0_clip`` (standardised units) before forming the posterior mean,
        which bounds the error amplification of :meth:`NoiseSchedule.predict_x0`
        at high noise; posterior variances are computed in float64.
        """
        sched = self.sampling_schedule
        x = temperature * torch.randn(
            (cond.shape[0], self.dim), device=cond.device, generator=generator
        )
        for i in reversed(range(sched.num_steps)):
            local = torch.full((x.shape[0],), i, dtype=torch.long, device=x.device)
            original = sched.timesteps.to(x.device)[local]
            eps = self.predict_noise(x, original, cond)
            x0 = sched.predict_x0(x, local, eps).clamp(-self.config.x0_clip, self.config.x0_clip)
            mean, var = sched.posterior(x0, x, local)
            if i > 0:
                noise = torch.randn(x.shape, device=x.device, generator=generator)
                x = mean + temperature * var.sqrt() * noise
            else:
                x = mean
        return x
