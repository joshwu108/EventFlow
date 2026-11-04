"""Neural next-event generative model shared by all four neural families.

``EventGenerativeModel`` = causal context encoder + (optional) factorised
categorical chain + continuous head (MDN, DDPM or flow matching). See
:mod:`eventflow.models.config` for how the presets map onto these parts.

Generation of one event (factorised mode)::

    h      = Encoder(events_{0..t})[t]                      1. encode history
    c      ~ CategoricalChain(h)          (constrained)     2. sample discrete fields
    x      ~ ContinuousHead(h + E(c))                       3. sample continuous fields
    event  = Tokenizer.decode(c, x)       (constrained)     4-6. inverse transforms
    events_{0..t+1} = events_{0..t} + event                 7. update context

Relaxed mode replaces 2-3 by one draw ``[r, x] ~ ContinuousHead(h)`` where
``r`` holds ``+-1`` one-hot relaxations of the categorical fields, decoded by
constrained argmax.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from eventflow.diffusion.head import DDPMHead
from eventflow.flow.head import FlowMatchingHead
from eventflow.models.config import ModelConfig
from eventflow.models.encoder import ContextEncoder
from eventflow.models.heads import CategoricalChain, CategoricalConstraints, MixtureDensityHead
from eventflow.tokenization.tokenizer import LOG_GAP, ZERO_GAP, EventTokenizer, FieldLayout
from eventflow.tokenization.windows import EventBatch


@dataclass(frozen=True)
class LossOutput:
    """``total`` is the optimised scalar; ``components`` are detached per-part means."""

    total: torch.Tensor
    components: dict[str, float]
    num_targets: int


@dataclass(frozen=True)
class NextEvent:
    """Encoded next event per batch row."""

    categorical: torch.Tensor  # [B, K] long
    continuous: torch.Tensor  # [B, D] float
    continuous_mask: torch.Tensor  # [B, D] bool


class EventGenerativeModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layout: FieldLayout,
        num_instruments: int,
        time_scale: float,
        constraints: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.config = config
        self.layout = layout
        self.num_instruments = num_instruments
        self.time_scale = time_scale
        d = config.d_model
        self.encoder = ContextEncoder(layout, config, num_instruments, time_scale)
        constraint_tensors = {
            k: torch.as_tensor(v, dtype=torch.bool) for k, v in (constraints or {}).items()
        }
        self.relaxed = config.categorical_mode == "relaxed"
        self.categorical: CategoricalChain | None = None
        if self.relaxed:
            # Relaxed models need only the constraint tables for argmax decoding.
            self.constraints = CategoricalConstraints(
                layout.categorical_names, layout.categorical_sizes, constraint_tensors
            )
        else:
            self.categorical = CategoricalChain(
                layout.categorical_names,
                layout.categorical_sizes,
                d,
                config.head_width,
                constraint_tensors,
                config.dropout,
            )
            self.constraints = self.categorical.constraints
        n_cont = len(layout.continuous)
        n_relaxed = sum(layout.categorical_sizes)
        self.continuous_dim = n_cont + (n_relaxed if self.relaxed else 0)
        self.continuous_head = self._build_head(d, self.continuous_dim)
        self.gap_index = layout.cont_index(LOG_GAP)
        self.zero_gap_index = layout.cat_index(ZERO_GAP)

    def _build_head(self, cond_dim: int, dim: int) -> nn.Module:
        c = self.config
        if c.continuous_head == "mdn":
            return MixtureDensityHead(cond_dim, dim, c.mdn_components, c.mdn_width or c.head_width)
        if c.continuous_head == "ddpm":
            return DDPMHead(cond_dim, dim, c.diffusion, c.head_width, c.head_depth, c.loss_repeats)
        return FlowMatchingHead(cond_dim, dim, c.flow, c.head_width, c.head_depth, c.loss_repeats)

    @classmethod
    def from_tokenizer(cls, config: ModelConfig, tokenizer: EventTokenizer) -> EventGenerativeModel:
        return cls(
            config,
            tokenizer.layout,
            len(tokenizer.instruments),
            tokenizer.time_scale,
            tokenizer.categorical_constraints(),
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # --------------------------------------------------------------- relaxation
    def relax(self, categorical: torch.Tensor) -> torch.Tensor:
        """``+-1`` one-hot encoding of every categorical field, concatenated."""
        parts = [
            2.0 * nn.functional.one_hot(categorical[:, k], size).float() - 1.0
            for k, size in enumerate(self.layout.categorical_sizes)
        ]
        return torch.cat(parts, dim=-1)

    def unrelax(self, relaxed: torch.Tensor) -> torch.Tensor:
        """Constrained argmax decoding of :meth:`relax` (event type decoded first)."""
        out = torch.zeros(
            relaxed.shape[0], len(self.layout.categorical), dtype=torch.long, device=relaxed.device
        )
        offset = 0
        for k, size in enumerate(self.layout.categorical_sizes):
            scores = relaxed[:, offset : offset + size]
            scores = self.constraints.mask(k, out[:, 0], scores)
            out[:, k] = scores.argmax(-1)
            offset += size
        return out

    # ----------------------------------------------------------------- training
    def encode_targets(self, batch: EventBatch) -> tuple[torch.Tensor, ...]:
        """Context summaries and next-event targets at all supervised positions.

        Returns ``(h, cat, cont, cont_mask)`` with a leading ``N`` dimension.
        """
        h = self.encoder(batch.inputs())
        cat, cont, cmask, target_valid = batch.targets()
        return h[target_valid], cat[target_valid], cont[target_valid], cmask[target_valid]

    def loss(self, batch: EventBatch) -> LossOutput:
        h, cat, cont, cmask = self.encode_targets(batch)
        n = int(h.shape[0])
        if n == 0:
            zero = h.sum() * 0.0
            return LossOutput(zero, {}, 0)
        components: dict[str, float] = {}
        if self.relaxed:
            x = torch.cat([self.relax(cat), cont], dim=-1)
            mask = torch.cat(
                [torch.ones_like(cat[:, :1]).expand(-1, x.shape[1] - cont.shape[1]).bool(), cmask],
                dim=-1,
            )
            cont_loss = self.continuous_head.loss(h, x, mask)
            total = self.config.continuous_loss_weight * cont_loss.mean()
            components["continuous"] = float(cont_loss.mean().detach())
        else:
            assert self.categorical is not None
            ce = self.categorical.loss(h, cat)
            ce_total = sum(ce.values())
            cond = self.categorical.condition(h, cat)
            cont_loss = self.continuous_head.loss(cond, cont, cmask)
            total = ce_total.mean() + self.config.continuous_loss_weight * cont_loss.mean()
            components.update({f"ce/{k}": float(v.mean().detach()) for k, v in ce.items()})
            components["ce/total"] = float(ce_total.mean().detach())
            components["continuous"] = float(cont_loss.mean().detach())
        components["total"] = float(total.detach())
        return LossOutput(total, components, n)

    # ----------------------------------------------------------------- sampling
    def last_context(self, context: EventBatch) -> torch.Tensor:
        """Encoder output at each row's last real event, ``[B, d_model]``."""
        h = self.encoder(context)
        last = context.valid.long().sum(1) - 1
        if torch.any(last < 0):
            raise ValueError("every context row needs at least one event")
        return h[torch.arange(h.shape[0], device=h.device), last]

    def _continuous_mask(self, categorical: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(
            categorical.shape[0],
            len(self.layout.continuous),
            dtype=torch.bool,
            device=categorical.device,
        )
        mask[:, self.gap_index] = categorical[:, self.zero_gap_index] == 0
        return mask

    @torch.no_grad()
    def sample_next(
        self,
        context: EventBatch,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        categorical: torch.Tensor | None = None,
    ) -> NextEvent:
        """Sample the next event of every row.

        ``categorical`` (factorised mode only) fixes the discrete fields and
        samples only the continuous ones: this isolates the continuous
        component for ablations (conditional on true discrete fields).
        """
        h = self.last_context(context)
        if self.relaxed:
            if categorical is not None:
                raise ValueError("relaxed models cannot condition on given categorical fields")
            x = self.continuous_head.sample(h, generator=generator, temperature=temperature)
            n_rel = self.continuous_dim - len(self.layout.continuous)
            cat = self.unrelax(x[:, :n_rel])
            cont = x[:, n_rel:]
        else:
            assert self.categorical is not None
            cat = (
                categorical
                if categorical is not None
                else self.categorical.sample(h, generator=generator, temperature=temperature)
            )
            cont = self.continuous_head.sample(
                self.categorical.condition(h, cat), generator=generator, temperature=temperature
            )
        mask = self._continuous_mask(cat)
        return NextEvent(cat, cont * mask.float(), mask)

    @torch.no_grad()
    def sample_categorical(
        self,
        context: EventBatch,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Discrete fields only (ablation of the categorical component)."""
        h = self.last_context(context)
        if self.categorical is None:
            return self.sample_next(context, generator, temperature).categorical
        return self.categorical.sample(h, generator=generator, temperature=temperature)
