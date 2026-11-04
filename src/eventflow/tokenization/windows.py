"""Windowed sequence datasets, batching and attention masks.

A *window* is a contiguous run of at most ``context_length + 1`` events from a
single session. Position ``t`` of the model input is event ``t`` of the window
and the training target at position ``t`` is event ``t + 1``. Windows never
cross session boundaries, so no sequence mixes two trading days or two
instruments.

Masks
-----
``valid[b, t]``        True where position ``t`` holds a real event (False = padding).
``target_valid[b, t]`` True where both event ``t`` and event ``t + 1`` are real;
                       the loss is averaged over these positions only.
:func:`causal_mask`    Boolean ``[T, T]`` matrix, True where attention is *blocked*
                       (key position after query position).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eventflow.errors import ConfigError
from eventflow.tokenization.tokenizer import EncodedEvents


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """``[length, length]`` bool mask, True above the diagonal (future keys blocked)."""
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


@dataclass(frozen=True)
class EventBatch:
    """A padded batch of event windows (all tensors on the same device)."""

    categorical: torch.Tensor  # [B, T, K] long
    continuous: torch.Tensor  # [B, T, D] float
    continuous_mask: torch.Tensor  # [B, T, D] bool
    instrument: torch.Tensor  # [B] long
    session_time: torch.Tensor  # [B, T] float, seconds since session start
    valid: torch.Tensor  # [B, T] bool

    @property
    def batch_size(self) -> int:
        return int(self.categorical.shape[0])

    @property
    def length(self) -> int:
        return int(self.categorical.shape[1])

    def to(self, device: torch.device | str) -> EventBatch:
        return EventBatch(*(t.to(device) for t in self._tensors()))

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.categorical,
            self.continuous,
            self.continuous_mask,
            self.instrument,
            self.session_time,
            self.valid,
        )

    def inputs(self) -> EventBatch:
        """Positions ``0..T-2``: the events a model conditions on."""
        return EventBatch(
            self.categorical[:, :-1],
            self.continuous[:, :-1],
            self.continuous_mask[:, :-1],
            self.instrument,
            self.session_time[:, :-1],
            self.valid[:, :-1],
        )

    def targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(categorical, continuous, continuous_mask, target_valid)`` for positions ``1..T-1``."""
        target_valid = self.valid[:, 1:] & self.valid[:, :-1]
        return (
            self.categorical[:, 1:],
            self.continuous[:, 1:],
            self.continuous_mask[:, 1:],
            target_valid,
        )

    def append(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        continuous_mask: torch.Tensor,
        session_time: torch.Tensor,
    ) -> EventBatch:
        """Return a new batch with one event per row appended (for rollouts).

        Requires every row to be unpadded at the end (rollouts left-align
        their prompts; see :func:`eventflow.generation.rollout.rollout`).
        """
        return EventBatch(
            torch.cat([self.categorical, categorical[:, None]], dim=1),
            torch.cat([self.continuous, continuous[:, None]], dim=1),
            torch.cat([self.continuous_mask, continuous_mask[:, None]], dim=1),
            self.instrument,
            torch.cat([self.session_time, session_time[:, None]], dim=1),
            torch.cat([self.valid, torch.ones_like(self.valid[:, :1])], dim=1),
        )

    def last(self, length: int) -> EventBatch:
        """Keep the final ``length`` positions (sliding context window)."""
        if self.length <= length:
            return self
        return EventBatch(
            self.categorical[:, -length:],
            self.continuous[:, -length:],
            self.continuous_mask[:, -length:],
            self.instrument,
            self.session_time[:, -length:],
            self.valid[:, -length:],
        )


def window_starts(
    session: np.ndarray, window: int, stride: int, *, min_length: int = 2
) -> list[tuple[int, int]]:
    """``[start, stop)`` windows of at most ``window`` rows that stay inside sessions.

    Every session of at least ``min_length`` rows contributes at least one
    window; the final window of a session is aligned to the session end so no
    tail events are dropped.
    """
    if window < 2 or stride < 1:
        raise ConfigError("window must be >= 2 and stride >= 1")
    out: list[tuple[int, int]] = []
    if session.size == 0:
        return out
    change = np.flatnonzero(session[1:] != session[:-1]) + 1
    bounds = zip(
        np.concatenate([[0], change]), np.concatenate([change, [session.size]]), strict=True
    )
    for a, b in bounds:
        a, b = int(a), int(b)
        if b - a < min_length:
            continue
        if b - a <= window:
            out.append((a, b))
            continue
        starts = list(range(a, b - window + 1, stride))
        if starts[-1] != b - window:
            starts.append(b - window)
        out.extend((s, s + window) for s in starts)
    return out


class EventWindowDataset(Dataset[dict[str, np.ndarray]]):
    """Map-style dataset of session-local windows over :class:`EncodedEvents`."""

    def __init__(self, encoded: EncodedEvents, context_length: int, stride: int | None = None):
        if context_length < 1:
            raise ConfigError("context_length must be >= 1")
        self.encoded = encoded
        self.context_length = context_length
        self.windows = window_starts(encoded.session, context_length + 1, stride or context_length)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        a, b = self.windows[index]
        e = self.encoded
        return {
            "categorical": e.categorical[a:b],
            "continuous": e.continuous[a:b],
            "continuous_mask": e.continuous_mask[a:b],
            "instrument": e.instrument[a],
            "session_time": e.session_time[a:b],
        }

    def num_targets(self) -> int:
        """Number of supervised next-event targets per pass (windows overlap by one event)."""
        return sum(b - a - 1 for a, b in self.windows)


def collate_events(items: Sequence[dict[str, np.ndarray]]) -> EventBatch:
    """Right-pad windows to a common length and build the validity mask."""
    lengths = [len(item["categorical"]) for item in items]
    T = max(lengths)
    B = len(items)
    K = items[0]["categorical"].shape[1]
    D = items[0]["continuous"].shape[1]
    cat = np.zeros((B, T, K), dtype=np.int64)
    cont = np.zeros((B, T, D), dtype=np.float32)
    cmask = np.zeros((B, T, D), dtype=bool)
    stime = np.zeros((B, T), dtype=np.float32)
    valid = np.zeros((B, T), dtype=bool)
    for i, (item, n) in enumerate(zip(items, lengths, strict=True)):
        cat[i, :n] = item["categorical"]
        cont[i, :n] = item["continuous"]
        cmask[i, :n] = item["continuous_mask"]
        stime[i, :n] = item["session_time"]
        valid[i, :n] = True
    instrument = np.array([int(item["instrument"]) for item in items], dtype=np.int64)
    return EventBatch(
        torch.from_numpy(cat),
        torch.from_numpy(cont),
        torch.from_numpy(cmask),
        torch.from_numpy(instrument),
        torch.from_numpy(stime),
        torch.from_numpy(valid),
    )
