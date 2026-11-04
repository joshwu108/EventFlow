"""Columnar market-event representation, capability flags and validation.

An :class:`EventFrame` stores one row per market event in parallel NumPy
arrays. Arrays are made read-only on construction; every transformation in
EventFlow returns a new frame (see :meth:`EventFrame.replace` and
:meth:`EventFrame.take`).

Field semantics
---------------

``timestamp``
    Seconds, float64, relative to an arbitrary per-dataset origin. Within a
    session timestamps are non-decreasing; equal timestamps are legal
    (simultaneous events are common in message data).
``instrument``
    int32 index into ``instruments``.
``event_type``
    int16 index into ``event_types``.
``side``
    int8 in ``{0: buy, 1: sell, 2: unknown}``. For LOBSTER-style message data
    this is the side of the *resting* limit order (the source convention).
``price_change``
    Relative price movement of the event in ticks, float64. NaN means missing.
``quantity``
    Event size in shares/contracts, float64, strictly positive. NaN = missing.
``session``
    int32 session index. Sessions are contiguous blocks; no sequence used for
    modelling ever crosses a session boundary.
``interarrival``
    Derived, never supplied: seconds since the previous event of the same
    session, NaN for the first event of each session.
``bid`` / ``ask`` / ``bid_size`` / ``ask_size``
    Optional top-of-book state *after* the event, in ticks and size units.
``labels``
    Optional auxiliary integer columns (e.g. ground-truth regimes of a
    synthetic generator) used only for stratified evaluation, never as model
    inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from eventflow.errors import CapabilityError, SchemaError

SIDE_BUY = 0
SIDE_SELL = 1
SIDE_UNKNOWN = 2
SIDE_NAMES: tuple[str, ...] = ("buy", "sell", "unknown")

BOOK_FIELDS: tuple[str, ...] = ("bid", "ask", "bid_size", "ask_size")
CORE_FIELDS: tuple[str, ...] = (
    "timestamp",
    "instrument",
    "event_type",
    "side",
    "price_change",
    "quantity",
    "session",
)
_DTYPES: dict[str, np.dtype[Any]] = {
    "timestamp": np.dtype(np.float64),
    "instrument": np.dtype(np.int32),
    "event_type": np.dtype(np.int16),
    "side": np.dtype(np.int8),
    "price_change": np.dtype(np.float64),
    "quantity": np.dtype(np.float64),
    "session": np.dtype(np.int32),
    "bid": np.dtype(np.float64),
    "ask": np.dtype(np.float64),
    "bid_size": np.dtype(np.float64),
    "ask_size": np.dtype(np.float64),
}


class Capability(StrEnum):
    """Properties a data source may or may not genuinely provide.

    Metrics and model features declare the capabilities they need; requesting
    one the dataset lacks raises :class:`~eventflow.errors.CapabilityError`
    instead of silently computing a statistic from placeholder values.
    """

    TIMESTAMPS = "timestamps"
    SIDE = "side"
    PRICE = "price"
    QUANTITY = "quantity"
    MULTI_EVENT_TYPE = "multi_event_type"
    MESSAGE_LEVEL = "message_level"
    BOOK_TOP = "book_top"
    LABELS = "labels"


@dataclass(frozen=True)
class Provenance:
    """Where a frame came from and how it was produced."""

    source: str
    adapter: str
    content_sha256: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "adapter": self.adapter,
            "content_sha256": self.content_sha256,
            "parameters": dict(self.parameters),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SessionInfo:
    """Metadata for one session (e.g. one instrument-day)."""

    session: int
    instrument: int
    name: str
    open_time: float | None = None


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _coerce(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    target = _DTYPES[name]
    if array.dtype.kind == "f" and target.kind in "iu":
        if not np.all(np.isfinite(array)):
            raise SchemaError(f"{name}: integer field contains non-finite values")
        if not np.all(array == np.round(array)):
            raise SchemaError(f"{name}: integer field contains non-integral values")
    try:
        return _readonly(array.astype(target, copy=False))
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name}: cannot convert to {target}") from exc


def compute_interarrival(timestamp: np.ndarray, session: np.ndarray) -> np.ndarray:
    """Seconds since the previous event of the same session; NaN at session starts."""
    out = np.full(timestamp.shape, np.nan, dtype=np.float64)
    if timestamp.size > 1:
        same = session[1:] == session[:-1]
        diffs = np.diff(timestamp)
        out[1:] = np.where(same, diffs, np.nan)
    return out


@dataclass(frozen=True, eq=False)
class EventFrame:
    """Immutable columnar collection of market events.

    Construct through :meth:`from_columns`, which coerces dtypes, derives the
    interarrival column and validates the frame.
    """

    timestamp: np.ndarray
    instrument: np.ndarray
    event_type: np.ndarray
    side: np.ndarray
    price_change: np.ndarray
    quantity: np.ndarray
    session: np.ndarray
    interarrival: np.ndarray
    event_types: tuple[str, ...]
    instruments: tuple[str, ...]
    capabilities: frozenset[Capability]
    provenance: Provenance
    sessions: tuple[SessionInfo, ...] = ()
    book: Mapping[str, np.ndarray] | None = None
    labels: Mapping[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_columns(
        cls,
        *,
        timestamp: Any,
        instrument: Any,
        event_type: Any,
        side: Any,
        price_change: Any,
        quantity: Any,
        session: Any,
        event_types: Sequence[str],
        instruments: Sequence[str],
        provenance: Provenance,
        capabilities: Sequence[Capability] | None = None,
        sessions: Sequence[SessionInfo] = (),
        book: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
        validate: bool = True,
    ) -> EventFrame:
        """Build a frame from array-likes.

        ``capabilities`` lists what the *source* provides; it is intersected
        with what the arrays can support (e.g. ``BOOK_TOP`` is dropped when no
        book columns are present) and never widened.
        """
        columns = {
            "timestamp": _coerce("timestamp", timestamp),
            "instrument": _coerce("instrument", instrument),
            "event_type": _coerce("event_type", event_type),
            "side": _coerce("side", side),
            "price_change": _coerce("price_change", price_change),
            "quantity": _coerce("quantity", quantity),
            "session": _coerce("session", session),
        }
        n = columns["timestamp"].shape[0]
        for name, array in columns.items():
            if array.ndim != 1 or array.shape[0] != n:
                raise SchemaError(f"{name}: expected shape ({n},), got {array.shape}")
        book_arrays: dict[str, np.ndarray] | None = None
        if book is not None:
            missing = [k for k in BOOK_FIELDS if k not in book]
            if missing:
                raise SchemaError(f"book: missing columns {missing}")
            book_arrays = {k: _coerce(k, book[k]) for k in BOOK_FIELDS}
            for k, array in book_arrays.items():
                if array.shape != (n,):
                    raise SchemaError(f"book.{k}: expected shape ({n},), got {array.shape}")
        label_arrays = {
            k: _readonly(np.asarray(v, dtype=np.int64)) for k, v in (labels or {}).items()
        }
        for k, array in label_arrays.items():
            if array.shape != (n,):
                raise SchemaError(f"labels.{k}: expected shape ({n},), got {array.shape}")

        supported = {Capability.TIMESTAMPS, Capability.PRICE, Capability.QUANTITY, Capability.SIDE}
        if len(event_types) > 1:
            supported.add(Capability.MULTI_EVENT_TYPE)
        supported.add(Capability.MESSAGE_LEVEL)
        if book_arrays is not None:
            supported.add(Capability.BOOK_TOP)
        if label_arrays:
            supported.add(Capability.LABELS)
        declared = (
            set(capabilities)
            if capabilities is not None
            else supported - {Capability.MESSAGE_LEVEL}
        )
        frame = cls(
            **columns,
            interarrival=_readonly(compute_interarrival(columns["timestamp"], columns["session"])),
            event_types=tuple(event_types),
            instruments=tuple(instruments),
            capabilities=frozenset(declared & supported),
            provenance=provenance,
            sessions=tuple(sessions),
            book=book_arrays,
            labels=label_arrays,
        )
        if validate:
            validate_frame(frame).raise_if_failed()
        return frame

    def __len__(self) -> int:
        return int(self.timestamp.shape[0])

    @property
    def num_sessions(self) -> int:
        return int(np.unique(self.session).size)

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def require(self, *capabilities: Capability, purpose: str = "this operation") -> None:
        """Raise :class:`CapabilityError` unless every capability is available."""
        missing = [c.value for c in capabilities if c not in self.capabilities]
        if missing:
            raise CapabilityError(
                f"{purpose} requires {missing}; dataset '{self.provenance.source}' provides "
                f"{sorted(c.value for c in self.capabilities)}"
            )

    def columns(self) -> dict[str, np.ndarray]:
        """Core columns (plus book columns when present) as a name -> array mapping."""
        out = {name: getattr(self, name) for name in CORE_FIELDS}
        if self.book is not None:
            out.update(self.book)
        return out

    def take(self, index: np.ndarray, *, validate: bool = True) -> EventFrame:
        """Return a new frame containing rows ``index`` (boolean mask or integer indices).

        Interarrival is recomputed from the retained rows, so dropping rows
        never leaves stale gaps.
        """
        index = np.asarray(index)
        book = None if self.book is None else {k: v[index] for k, v in self.book.items()}
        kept_sessions = set(np.unique(self.session[index]).tolist())
        return EventFrame.from_columns(
            **{name: getattr(self, name)[index] for name in CORE_FIELDS},
            event_types=self.event_types,
            instruments=self.instruments,
            provenance=self.provenance,
            capabilities=tuple(self.capabilities),
            sessions=tuple(s for s in self.sessions if s.session in kept_sessions),
            book=book,
            labels={k: v[index] for k, v in self.labels.items()},
            validate=validate,
        )

    def replace(self, **changes: Any) -> EventFrame:
        """Return a copy with selected constructor arguments replaced, re-validated."""
        kwargs: dict[str, Any] = {name: getattr(self, name) for name in CORE_FIELDS}
        kwargs.update(
            event_types=self.event_types,
            instruments=self.instruments,
            provenance=self.provenance,
            capabilities=tuple(self.capabilities),
            sessions=self.sessions,
            book=self.book,
            labels=self.labels,
        )
        unknown = set(changes) - set(kwargs) - {"validate"}
        if unknown:
            raise TypeError(f"unknown fields: {sorted(unknown)}")
        kwargs.update(changes)
        return EventFrame.from_columns(**kwargs)

    def session_bounds(self) -> list[tuple[int, int]]:
        """``[start, stop)`` row ranges of each contiguous session block."""
        if len(self) == 0:
            return []
        change = np.flatnonzero(self.session[1:] != self.session[:-1]) + 1
        starts = np.concatenate([[0], change])
        stops = np.concatenate([change, [len(self)]])
        return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]

    def summary(self) -> dict[str, Any]:
        """Small JSON-serialisable description used in manifests and the CLI."""
        return {
            "events": len(self),
            "sessions": self.num_sessions,
            "instruments": list(self.instruments),
            "event_types": list(self.event_types),
            "capabilities": sorted(c.value for c in self.capabilities),
            "provenance": self.provenance.to_dict(),
            "time_span": [float(self.timestamp.min()), float(self.timestamp.max())]
            if len(self)
            else None,
        }


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" or "warning"
    check: str
    message: str
    count: int = 0


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    missing_fraction: Mapping[str, float]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def raise_if_failed(self) -> None:
        if not self.ok:
            details = "; ".join(f"{i.check}: {i.message}" for i in self.errors())
            raise SchemaError(f"event frame failed validation: {details}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [dataclasses.asdict(i) for i in self.issues],
            "missing_fraction": dict(self.missing_fraction),
        }


def validate_frame(frame: EventFrame) -> ValidationReport:
    """Check structural invariants of a frame.

    Errors make the frame unusable for modelling; warnings flag properties
    (such as missing values) that preprocessing must resolve explicitly.
    """
    issues: list[ValidationIssue] = []

    def error(check: str, message: str, count: int = 0) -> None:
        issues.append(ValidationIssue("error", check, message, count))

    def warning(check: str, message: str, count: int = 0) -> None:
        issues.append(ValidationIssue("warning", check, message, count))

    n = len(frame)
    if not np.all(np.isfinite(frame.timestamp)):
        error(
            "timestamp_finite",
            "timestamps must be finite",
            int((~np.isfinite(frame.timestamp)).sum()),
        )
    if n > 1:
        session_steps = np.diff(frame.session.astype(np.int64))
        if np.any(session_steps < 0):
            error(
                "session_contiguous",
                "session ids must be non-decreasing (contiguous blocks)",
                int((session_steps < 0).sum()),
            )
        same = session_steps == 0
        backwards = same & (np.diff(frame.timestamp) < 0)
        if np.any(backwards):
            error("timestamp_order", "timestamps decrease within a session", int(backwards.sum()))
    if n and (frame.event_type.min() < 0 or frame.event_type.max() >= len(frame.event_types)):
        error("event_type_vocab", f"event_type outside vocabulary of size {len(frame.event_types)}")
    if n and (frame.instrument.min() < 0 or frame.instrument.max() >= len(frame.instruments)):
        error("instrument_vocab", f"instrument outside vocabulary of size {len(frame.instruments)}")
    bad_side = ~np.isin(frame.side, (SIDE_BUY, SIDE_SELL, SIDE_UNKNOWN))
    if np.any(bad_side):
        error("side_range", "side must be 0 (buy), 1 (sell) or 2 (unknown)", int(bad_side.sum()))
    qty = frame.quantity
    nonpositive = np.isfinite(qty) & (qty <= 0)
    if np.any(nonpositive):
        error("quantity_positive", "quantities must be strictly positive", int(nonpositive.sum()))
    if np.any(np.isinf(qty)) or np.any(np.isinf(frame.price_change)):
        error("finite_values", "quantity and price_change may be NaN (missing) but not infinite")
    if frame.book is not None:
        bid, ask = frame.book["bid"], frame.book["ask"]
        crossed = np.isfinite(bid) & np.isfinite(ask) & (bid > ask)
        if np.any(crossed):
            error("book_crossed", "bid exceeds ask", int(crossed.sum()))
        for k in ("bid_size", "ask_size"):
            negative = np.isfinite(frame.book[k]) & (frame.book[k] < 0)
            if np.any(negative):
                error("book_size", f"{k} must be non-negative", int(negative.sum()))
    if frame.has(Capability.SIDE) and n and np.all(frame.side == SIDE_UNKNOWN):
        warning("side_capability", "SIDE capability declared but every side is unknown")

    missing = {
        "price_change": float(np.isnan(frame.price_change).mean()) if n else 0.0,
        "quantity": float(np.isnan(frame.quantity).mean()) if n else 0.0,
        "side": float((frame.side == SIDE_UNKNOWN).mean()) if n else 0.0,
    }
    for name in ("price_change", "quantity"):
        if missing[name] > 0:
            warning(
                "missing_values",
                f"{name} has {missing[name]:.2%} missing values",
                int(round(missing[name] * n)),
            )
    return ValidationReport(tuple(issues), missing)
