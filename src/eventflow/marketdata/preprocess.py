"""Configurable preprocessing, chronological partitioning and dataset fingerprints.

Every function here returns a *new* frame; inputs are never modified.

Chronological splitting is the only supported partitioning. Shuffled splits
leak future information through overlapping windows and shared regimes, so
they are deliberately not offered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from eventflow.errors import ConfigError, SchemaError
from eventflow.marketdata.schema import (
    BOOK_FIELDS,
    CORE_FIELDS,
    Capability,
    EventFrame,
    SessionInfo,
)

MissingPolicy = Literal["error", "drop"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Cleaning steps, applied in the order of the fields below.

    ``missing``
        What to do with rows whose ``price_change`` or ``quantity`` is NaN
        when the dataset claims that capability: ``"error"`` refuses the
        frame, ``"drop"`` removes the rows. Imputation is deliberately not
        offered: an imputed event is a fabricated event.
    ``drop_event_types``
        Event-type labels to remove (e.g. ``"halt"``).
    ``max_gap``
        Split a session wherever consecutive events are further apart than
        this many seconds (``None`` disables).
    ``min_session_events``
        Drop sessions shorter than this after the steps above.
    """

    missing: MissingPolicy = "error"
    drop_event_types: tuple[str, ...] = ()
    max_gap: float | None = None
    min_session_events: int = 2

    def __post_init__(self) -> None:
        if self.missing not in ("error", "drop"):
            raise ConfigError(f"preprocess.missing must be 'error' or 'drop', got {self.missing!r}")
        if self.max_gap is not None and self.max_gap <= 0:
            raise ConfigError("preprocess.max_gap must be positive")
        if self.min_session_events < 2:
            raise ConfigError("preprocess.min_session_events must be at least 2")


@dataclass(frozen=True)
class PreprocessResult:
    frame: EventFrame
    log: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _renumber_sessions(
    frame: EventFrame, new_session: np.ndarray, names: dict[int, str]
) -> EventFrame:
    """Re-index sessions to 0..k-1 in order of appearance, keeping metadata."""
    _, first = np.unique(new_session, return_index=True)
    order = new_session[np.sort(first)]
    lookup = {int(s): i for i, s in enumerate(order)}
    session = np.array([lookup[int(s)] for s in new_session], dtype=np.int64)
    infos = []
    for old, new in lookup.items():
        row = int(np.flatnonzero(new_session == old)[0])
        infos.append(
            SessionInfo(
                new,
                int(frame.instrument[row]),
                names.get(old, f"session{old}"),
                float(frame.timestamp[row]),
            )
        )
    return frame.replace(session=session, sessions=tuple(infos))


def preprocess(frame: EventFrame, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    """Apply :class:`PreprocessConfig` and return the cleaned frame plus a step log."""
    cfg = cfg or PreprocessConfig()
    log: list[dict[str, Any]] = []
    current = frame

    missing = np.zeros(len(current), dtype=bool)
    if current.has(Capability.PRICE):
        missing |= np.isnan(current.price_change)
    if current.has(Capability.QUANTITY):
        missing |= np.isnan(current.quantity)
    if missing.any():
        if cfg.missing == "error":
            raise SchemaError(
                f"{int(missing.sum())} events have missing price_change/quantity; "
                "set preprocess.missing='drop' to remove them"
            )
        current = current.take(~missing)
        log.append({"step": "drop_missing", "removed": int(missing.sum())})

    if cfg.drop_event_types:
        unknown = set(cfg.drop_event_types) - set(current.event_types)
        if unknown:
            raise ConfigError(f"drop_event_types {sorted(unknown)} not in {current.event_types}")
        codes = [current.event_types.index(name) for name in cfg.drop_event_types]
        drop = np.isin(current.event_type, codes)
        current = current.take(~drop)
        log.append(
            {
                "step": "drop_event_types",
                "types": list(cfg.drop_event_types),
                "removed": int(drop.sum()),
            }
        )

    names = {s.session: s.name for s in current.sessions}
    if cfg.max_gap is not None and len(current) > 1:
        gap = np.nan_to_num(current.interarrival, nan=0.0) > cfg.max_gap
        if gap.any():
            # Each gap starts a new session; encode as (old session, segment index).
            segment = np.cumsum(gap)
            new_session = current.session.astype(np.int64) * (len(current) + 1) + segment
            seg_names = {
                int(
                    s
                ): f"{names.get(int(s) // (len(current) + 1), 's')}#{int(s) % (len(current) + 1)}"
                for s in np.unique(new_session)
            }
            current = _renumber_sessions(current, new_session, seg_names)
            names = {s.session: s.name for s in current.sessions}
            log.append({"step": "split_on_gap", "max_gap": cfg.max_gap, "splits": int(gap.sum())})

    counts = np.bincount(current.session, minlength=int(current.session.max(initial=-1)) + 1)
    short = counts[current.session] < cfg.min_session_events
    if short.any():
        kept = current.take(~short)
        current = _renumber_sessions(
            kept, kept.session.astype(np.int64), {s.session: s.name for s in kept.sessions}
        )
        log.append({"step": "drop_short_sessions", "removed_events": int(short.sum())})
    if len(current) == 0:
        raise SchemaError("preprocessing removed every event")
    return PreprocessResult(current, tuple(log))


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split fractions and boundary handling.

    ``unit="session"`` assigns whole sessions (ordered by open time) to the
    partitions so that cumulative event counts approximate the fractions.
    ``unit="time"`` cuts at global timestamps and may split a session in two;
    ``embargo`` seconds after each cut are discarded so that no training
    window ends right before the first evaluation event.
    """

    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15
    unit: Literal["session", "time"] = "session"
    embargo: float = 0.0

    def __post_init__(self) -> None:
        parts = (self.train, self.validation, self.test)
        if any(p < 0 for p in parts) or not np.isclose(sum(parts), 1.0):
            raise ConfigError("split fractions must be non-negative and sum to 1")
        if self.train <= 0:
            raise ConfigError("split.train must be positive")
        if self.unit not in ("session", "time"):
            raise ConfigError("split.unit must be 'session' or 'time'")
        if self.embargo < 0:
            raise ConfigError("split.embargo must be non-negative")


@dataclass(frozen=True)
class Partitions:
    train: EventFrame
    validation: EventFrame | None
    test: EventFrame | None

    def items(self) -> list[tuple[str, EventFrame]]:
        out = [("train", self.train)]
        if self.validation is not None:
            out.append(("validation", self.validation))
        if self.test is not None:
            out.append(("test", self.test))
        return out


def _subset(frame: EventFrame, mask: np.ndarray) -> EventFrame | None:
    if not mask.any():
        return None
    sub = frame.take(mask)
    return _renumber_sessions(
        sub, sub.session.astype(np.int64), {s.session: s.name for s in sub.sessions}
    )


def chronological_split(frame: EventFrame, cfg: SplitConfig | None = None) -> Partitions:
    """Partition ``frame`` into train / validation / test in time order."""
    cfg = cfg or SplitConfig()
    if cfg.unit == "session":
        bounds = frame.session_bounds()
        opens = np.array([frame.timestamp[a] for a, _ in bounds])
        sizes = np.array([b - a for a, b in bounds], dtype=np.float64)
        order = np.argsort(opens, kind="stable")
        cumulative = np.cumsum(sizes[order]) / sizes.sum()
        labels = np.empty(len(bounds), dtype=np.int64)
        # A session goes to the partition containing its event-count midpoint.
        midpoints = cumulative - 0.5 * sizes[order] / sizes.sum()
        labels[order] = np.searchsorted(
            np.cumsum([cfg.train, cfg.validation]), midpoints, side="right"
        )
        row_label = np.repeat(labels, sizes.astype(np.int64))
        keep = np.ones(len(frame), dtype=bool)
    else:
        cut1, cut2 = np.quantile(frame.timestamp, [cfg.train, cfg.train + cfg.validation])
        row_label = np.searchsorted([cut1, cut2], frame.timestamp, side="right")
        keep = ~(
            ((frame.timestamp > cut1) & (frame.timestamp <= cut1 + cfg.embargo))
            | ((frame.timestamp > cut2) & (frame.timestamp <= cut2 + cfg.embargo))
        )
        # Rows on either side of a cut in the same original session become separate sessions.
        new_session = frame.session.astype(np.int64) * 3 + row_label
        names = {
            int(s): f"{_session_name(frame, int(s) // 3)}@p{int(s) % 3}"
            for s in np.unique(new_session)
        }
        frame = _renumber_sessions(frame, new_session, names)
    partitions = [_subset(frame, (row_label == k) & keep) for k in range(3)]
    if partitions[0] is None:
        raise SchemaError("chronological split produced an empty training partition")
    return Partitions(partitions[0], partitions[1], partitions[2])


def _session_name(frame: EventFrame, session: int) -> str:
    for info in frame.sessions:
        if info.session == session:
            return info.name
    return f"session{session}"


def fingerprint(frame: EventFrame) -> str:
    """SHA-256 over every column, vocabulary and capability of the frame.

    Two frames share a fingerprint iff their contents are identical; the
    provenance text is excluded so the same data loaded from two paths
    fingerprints identically.
    """
    digest = hashlib.sha256()
    header = {
        "event_types": frame.event_types,
        "instruments": frame.instruments,
        "capabilities": sorted(c.value for c in frame.capabilities),
        "book": frame.book is not None,
        "labels": sorted(frame.labels),
    }
    digest.update(json.dumps(header, sort_keys=True).encode())
    for name in CORE_FIELDS:
        digest.update(np.ascontiguousarray(getattr(frame, name)).tobytes())
    if frame.book is not None:
        for name in BOOK_FIELDS:
            digest.update(np.ascontiguousarray(frame.book[name]).tobytes())
    for name in sorted(frame.labels):
        digest.update(np.ascontiguousarray(frame.labels[name]).tobytes())
    return digest.hexdigest()
