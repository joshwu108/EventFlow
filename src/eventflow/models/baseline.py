"""Statistical autoregressive baseline: Markov chain + conditional bootstrap.

The model has no learned neural parameters:

* **Categorical fields.** The joint categorical tuple ``s_t`` (event type,
  side, price bin, zero-gap flag) follows a first-order Markov chain per
  instrument, estimated by counting transitions with additive (Laplace)
  smoothing ``alpha`` over the states observed in training::

      p(s_{t+1} = j | s_t = i) = (n_ij + alpha) / (n_i + alpha * |S|)

  States are joint tuples, so the baseline respects every categorical
  combination observed in training by construction.

* **Continuous fields.** Given the sampled next state ``j``, the continuous
  vector is drawn uniformly from the training events of instrument ``g`` in
  state ``j`` (falling back to all instruments, then to all events, when that
  pool is empty). This reproduces conditional marginals exactly in the
  large-sample limit but has no temporal dependence beyond the chain.

It answers "how much does a learned context encoder add over first-order
categorical dynamics and empirical conditional marginals?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from eventflow.errors import ConfigError
from eventflow.models.generative import NextEvent
from eventflow.tokenization.tokenizer import LOG_GAP, ZERO_GAP, EncodedEvents, FieldLayout
from eventflow.tokenization.windows import EventBatch


@dataclass
class MarkovBaseline:
    """Fit with :meth:`fit`; implements the ``sample_next`` protocol of neural models."""

    layout: FieldLayout
    num_instruments: int
    smoothing: float = 0.5
    states: dict[tuple[int, ...], int] = field(default_factory=dict)
    counts: np.ndarray | None = None  # [G, S, S]
    pools: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    state_pools: dict[int, np.ndarray] = field(default_factory=dict)
    continuous: np.ndarray | None = None
    continuous_mask: np.ndarray | None = None
    state_list: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.smoothing <= 0:
            raise ConfigError("baseline smoothing must be positive")

    @classmethod
    def fit(
        cls,
        encoded: EncodedEvents,
        layout: FieldLayout,
        num_instruments: int,
        smoothing: float = 0.5,
    ) -> MarkovBaseline:
        model = cls(layout, num_instruments, smoothing)
        tuples = [tuple(int(v) for v in row) for row in encoded.categorical]
        for t in tuples:
            model.states.setdefault(t, len(model.states))
        state = np.array([model.states[t] for t in tuples], dtype=np.int64)
        S = len(model.states)
        counts = np.zeros((num_instruments, S, S), dtype=np.float64)
        same = encoded.session[1:] == encoded.session[:-1]
        np.add.at(counts, (encoded.instrument[1:][same], state[:-1][same], state[1:][same]), 1.0)
        model.counts = counts
        model.continuous = encoded.continuous.astype(np.float32)
        model.continuous_mask = encoded.continuous_mask
        model.state_list = np.array(list(model.states), dtype=np.int64)
        # Session-first events have an undefined gap (masked placeholder); they
        # never serve as bootstrap donors for the continuous fields.
        gap_defined = encoded.continuous_mask[:, layout.cont_index(LOG_GAP)] | (
            encoded.categorical[:, layout.cat_index(ZERO_GAP)] == 1
        )
        for g in range(num_instruments):
            for s in range(S):
                rows = np.flatnonzero((encoded.instrument == g) & (state == s) & gap_defined)
                if rows.size:
                    model.pools[(g, s)] = rows
        for s in range(S):
            rows = np.flatnonzero((state == s) & gap_defined)
            model.state_pools[s] = rows if rows.size else np.flatnonzero(gap_defined)
        return model

    def transition_probs(self, instrument: int, state: int) -> np.ndarray:
        assert self.counts is not None
        row = self.counts[instrument, state] + self.smoothing
        return row / row.sum()

    def _state_of(self, categorical: np.ndarray) -> int:
        # Unseen context tuples (possible in rollouts of other models) map to the
        # most frequent training state rather than failing.
        key = tuple(int(v) for v in categorical)
        if key in self.states:
            return self.states[key]
        assert self.counts is not None
        return int(self.counts.sum(axis=(0, 1)).argmax())

    def categorical_nll(self, encoded: EncodedEvents) -> float:
        """Mean next-event categorical negative log-likelihood (nats) within sessions.

        Comparable with the ``ce/total`` component of the neural models.
        Target states never seen in training receive the smoothing mass of an
        extra state, keeping the NLL finite.
        """
        assert self.counts is not None
        total, n = 0.0, 0
        for i in range(1, len(encoded)):
            if encoded.session[i] != encoded.session[i - 1]:
                continue
            g = int(encoded.instrument[i])
            prev = self._state_of(encoded.categorical[i - 1])
            row = self.counts[g, prev]
            key = tuple(int(v) for v in encoded.categorical[i])
            count = row[self.states[key]] if key in self.states else 0.0
            denom = row.sum() + self.smoothing * (row.size + 1)
            total -= float(np.log((count + self.smoothing) / denom))
            n += 1
        return total / max(n, 1)

    def sample_next(
        self,
        context: EventBatch,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> NextEvent:
        """Sample one next event per row (``temperature`` is accepted and ignored)."""
        assert self.continuous is not None and self.state_list is not None
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        rng = np.random.default_rng(seed)
        last = (context.valid.long().sum(1) - 1).cpu().numpy()
        cat = context.categorical.cpu().numpy()
        inst = context.instrument.cpu().numpy()
        B = cat.shape[0]
        out_cat = np.zeros((B, len(self.layout.categorical)), dtype=np.int64)
        out_cont = np.zeros((B, len(self.layout.continuous)), dtype=np.float32)
        for b in range(B):
            g = int(inst[b])
            state = self._state_of(cat[b, last[b]])
            nxt = int(rng.choice(len(self.state_list), p=self.transition_probs(g, state)))
            out_cat[b] = self.state_list[nxt]
            pool = self.pools.get((g, nxt))
            if pool is None:
                pool = self.state_pools.get(nxt, np.arange(len(self.continuous)))
            out_cont[b] = self.continuous[int(rng.choice(pool))]
        gap = self.layout.cont_index(LOG_GAP)
        zero = self.layout.cat_index(ZERO_GAP)
        mask = np.ones_like(out_cont, dtype=bool)
        mask[:, gap] = out_cat[:, zero] == 0
        device = context.categorical.device
        return NextEvent(
            torch.from_numpy(out_cat).to(device),
            torch.from_numpy(out_cont * mask).to(device),
            torch.from_numpy(mask).to(device),
        )

    def eval(self) -> MarkovBaseline:  # parity with nn.Module in generation code
        return self

    def num_parameters(self) -> int:
        """Number of estimated quantities (transition counts), for budget reporting."""
        return 0 if self.counts is None else int(self.counts.size)

    def summary(self) -> dict[str, Any]:
        return {
            "states": len(self.states),
            "smoothing": self.smoothing,
            "transition_entries": self.num_parameters(),
        }
