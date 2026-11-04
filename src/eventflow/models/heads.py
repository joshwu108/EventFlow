"""Output heads: factorised categorical chain and a mixture-density continuous head.

Factorisation of one event given the context summary ``h``::

    p(event | h) = p(c_1 | h) * prod_{k>1} p(c_k | h, c_{<k}) * p(x | h, c)

where ``c_1`` is the event type, ``c_k`` the remaining categorical fields in
:data:`~eventflow.tokenization.tokenizer.CATEGORICAL_ORDER` and ``x`` the
continuous fields. Training uses teacher forcing (true ``c_{<k}``); sampling
draws the fields in order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from eventflow.models.layers import MLP

_NEG_INF = -1e9  # finite stand-in for -inf: keeps softmax defined if a row is fully masked


class ContinuousHead(Protocol):
    """Interface implemented by the MDN, DDPM and flow-matching heads.

    ``cond`` is ``[N, C]``, ``x`` and ``mask`` are ``[N, D]``; ``mask`` marks
    the dimensions that are defined for each row. ``loss`` returns ``[N]``.
    """

    def loss(self, cond: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...

    def sample(
        self,
        cond: torch.Tensor,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor: ...


class CategoricalConstraints(nn.Module):
    """Boolean tables ``allowed_k[event_type, value]`` for fields ``k >= 1``.

    Built from :meth:`EventTokenizer.categorical_constraints`; fields without
    a table allow every value.
    """

    def __init__(
        self,
        names: Sequence[str],
        sizes: Sequence[int],
        constraints: Mapping[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        for k, name in enumerate(names[1:], start=1):
            table = None if constraints is None else constraints.get(name)
            if table is None:
                table = torch.ones(sizes[0], sizes[k], dtype=torch.bool)
            table = torch.as_tensor(table, dtype=torch.bool)
            if tuple(table.shape) != (sizes[0], sizes[k]):
                raise ValueError(f"constraint table for {name} has shape {tuple(table.shape)}")
            self.register_buffer(f"allowed_{k}", table)

    def allowed(self, k: int) -> torch.Tensor:
        return getattr(self, f"allowed_{k}")

    def mask(self, k: int, event_type: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Replace scores of values not allowed with ``event_type`` by a large negative."""
        if k == 0:
            return scores
        return scores.masked_fill(~self.allowed(k)[event_type], _NEG_INF)


class CategoricalChain(nn.Module):
    """Autoregressive softmax heads over the categorical fields of one event.

    ``constraints[name][event_type, value]`` (from the tokenizer) restricts
    sampling to categorical combinations observed in training. Constraints are
    applied at sampling time only; the training loss is the unconstrained
    cross-entropy so that the likelihood of evaluation data stays finite.
    """

    def __init__(
        self,
        names: Sequence[str],
        sizes: Sequence[int],
        d_model: int,
        width: int,
        constraints: Mapping[str, torch.Tensor] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.names = tuple(names)
        self.sizes = tuple(sizes)
        self.heads = nn.ModuleList(
            MLP(d_model, width, size, depth=2, dropout=dropout) for size in sizes
        )
        # Embeddings of earlier fields feed later heads; a separate set embeds all
        # fields of the target event to condition the continuous head.
        self.prefix_embeddings = nn.ModuleList(nn.Embedding(size, d_model) for size in sizes[:-1])
        self.condition_embeddings = nn.ModuleList(nn.Embedding(size, d_model) for size in sizes)
        self.constraints = CategoricalConstraints(names, sizes, constraints)

    def logits(self, h: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
        """Teacher-forced logits for every field: list of ``[N, size_k]``."""
        out = []
        prefix = h
        for k, head in enumerate(self.heads):
            out.append(head(prefix))
            if k < len(self.prefix_embeddings):
                prefix = prefix + self.prefix_embeddings[k](targets[:, k])
        return out

    def loss(self, h: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-field cross-entropy, each ``[N]``."""
        return {
            name: F.cross_entropy(logits, targets[:, k], reduction="none")
            for k, (name, logits) in enumerate(
                zip(self.names, self.logits(h, targets), strict=True)
            )
        }

    def condition(self, h: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Conditioning vector ``h + sum_k E_k(c_k)`` for the continuous head."""
        return h + sum(emb(values[:, k]) for k, emb in enumerate(self.condition_embeddings))

    @torch.no_grad()
    def sample(
        self, h: torch.Tensor, generator: torch.Generator | None = None, temperature: float = 1.0
    ) -> torch.Tensor:
        """Draw all categorical fields in chain order; returns ``[N, K]`` long."""
        values = torch.zeros(h.shape[0], len(self.heads), dtype=torch.long, device=h.device)
        prefix = h
        for k, head in enumerate(self.heads):
            logits = head(prefix) / max(temperature, 1e-6)
            logits = self.constraints.mask(k, values[:, 0], logits)
            probs = torch.softmax(logits.float(), dim=-1)
            values[:, k] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            if k < len(self.prefix_embeddings):
                prefix = prefix + self.prefix_embeddings[k](values[:, k])
        return values


class MixtureDensityHead(nn.Module):
    """Diagonal Gaussian mixture ``p(x | cond)`` with masked dimensions.

    For a diagonal component, marginalising an undefined dimension simply
    drops its factor, so the masked log-likelihood is exact::

        log p(x_obs) = logsumexp_j [ log pi_j + sum_{d in obs} log N(x_d; mu_jd, s_jd^2) ]
    """

    LOG_STD_MIN, LOG_STD_MAX = -7.0, 5.0

    def __init__(self, cond_dim: int, dim: int, components: int, width: int):
        super().__init__()
        self.dim, self.components = dim, components
        self.net = MLP(cond_dim, width, components * (1 + 2 * dim), depth=3)

    def _params(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(cond)
        J, D = self.components, self.dim
        logit_pi = raw[:, :J]
        mu = raw[:, J : J + J * D].view(-1, J, D)
        log_std = raw[:, J + J * D :].view(-1, J, D).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return logit_pi, mu, log_std

    def loss(self, cond: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logit_pi, mu, log_std = self._params(cond)
        z = (x[:, None, :] - mu) * torch.exp(-log_std)
        log_n = -0.5 * z.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        log_n = (log_n * mask[:, None, :].float()).sum(-1)
        log_p = torch.logsumexp(torch.log_softmax(logit_pi, -1) + log_n, dim=-1)
        return -log_p

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, generator: torch.Generator | None = None, temperature: float = 1.0
    ) -> torch.Tensor:
        logit_pi, mu, log_std = self._params(cond)
        j = torch.multinomial(torch.softmax(logit_pi.float(), -1), 1, generator=generator).squeeze(
            -1
        )
        rows = torch.arange(cond.shape[0], device=cond.device)
        noise = torch.randn(mu[rows, j].shape, generator=generator, device=cond.device)
        return mu[rows, j] + temperature * torch.exp(log_std[rows, j]) * noise
