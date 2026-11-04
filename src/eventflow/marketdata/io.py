"""CSV and Parquet adapters for generic event tables.

The generic table layout has one row per event and these columns (names are
configurable through :class:`ColumnMap`):

=============  ========  ====================================================
column         required  meaning
=============  ========  ====================================================
timestamp      yes       seconds (float) relative to any fixed origin
instrument     yes       instrument symbol (string)
event_type     yes       event-type label (string)
side           no        ``buy`` / ``sell`` / empty
price_change   no        relative price movement in ticks
quantity       no        size, strictly positive
session        no        session label; defaults to instrument
bid, ask,      no        top of book after the event (all four or none)
bid_size,
ask_size
=============  ========  ====================================================

Absent optional columns are filled with missing values *and* the matching
capability is withdrawn, so downstream code cannot mistake a placeholder for
data. A table of public trade prints (one event type, no book) therefore
yields a frame without ``MULTI_EVENT_TYPE``, ``MESSAGE_LEVEL`` or
``BOOK_TOP``: trade aggregates are not order-book event streams.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eventflow.errors import SchemaError
from eventflow.marketdata.schema import (
    BOOK_FIELDS,
    SIDE_BUY,
    SIDE_NAMES,
    SIDE_SELL,
    SIDE_UNKNOWN,
    Capability,
    EventFrame,
    Provenance,
    SessionInfo,
)

_SIDE_LOOKUP = {
    "buy": SIDE_BUY,
    "b": SIDE_BUY,
    "bid": SIDE_BUY,
    "1": SIDE_BUY,
    "sell": SIDE_SELL,
    "s": SIDE_SELL,
    "ask": SIDE_SELL,
    "-1": SIDE_SELL,
    "": SIDE_UNKNOWN,
    "unknown": SIDE_UNKNOWN,
    "nan": SIDE_UNKNOWN,
}


@dataclass(frozen=True)
class ColumnMap:
    """Source column names for each logical field."""

    timestamp: str = "timestamp"
    instrument: str = "instrument"
    event_type: str = "event_type"
    side: str = "side"
    price_change: str = "price_change"
    quantity: str = "quantity"
    session: str = "session"
    bid: str = "bid"
    ask: str = "ask"
    bid_size: str = "bid_size"
    ask_size: str = "ask_size"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _float_column(values: Sequence[Any], name: str) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, str) and value.strip() == ""):
            out[i] = np.nan
            continue
        try:
            out[i] = float(value)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{name}: row {i} value {value!r} is not numeric") from exc
    return out


def _side_column(values: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=np.int8)
    for i, value in enumerate(values):
        key = "" if value is None else str(value).strip().lower()
        if key not in _SIDE_LOOKUP:
            raise SchemaError(f"side: row {i} value {value!r} is not buy/sell/unknown")
        out[i] = _SIDE_LOOKUP[key]
    return out


def _index(
    values: Sequence[Any], vocabulary: Sequence[str] | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    labels = [str(v) for v in values]
    vocab = tuple(vocabulary) if vocabulary is not None else tuple(sorted(set(labels)))
    lookup = {name: i for i, name in enumerate(vocab)}
    unknown = sorted(set(labels) - set(lookup))
    if unknown:
        raise SchemaError(f"labels {unknown[:5]} are not in the declared vocabulary")
    return np.array([lookup[v] for v in labels], dtype=np.int64), vocab


def frame_from_table(
    table: Mapping[str, Sequence[Any]],
    *,
    provenance: Provenance,
    columns: ColumnMap | None = None,
    event_types: Sequence[str] | None = None,
    message_level: bool = False,
) -> EventFrame:
    """Build a frame from a column-name -> values mapping.

    Rows are sorted stably by (session, timestamp). ``message_level`` must be
    set explicitly by the caller when the table genuinely holds order-book
    messages; it is never inferred.
    """
    cols = columns or ColumnMap()
    for required in (cols.timestamp, cols.instrument, cols.event_type):
        if required not in table:
            raise SchemaError(f"missing required column '{required}'")
    n = len(table[cols.timestamp])
    timestamp = _float_column(table[cols.timestamp], "timestamp")
    instrument, instruments = _index(table[cols.instrument], None)
    etype, vocab = _index(table[cols.event_type], event_types)
    capabilities = {Capability.TIMESTAMPS}
    # A declared vocabulary is not evidence of multiple event types; observed codes are.
    if np.unique(etype).size > 1:
        capabilities.add(Capability.MULTI_EVENT_TYPE)
    if message_level:
        capabilities.add(Capability.MESSAGE_LEVEL)

    if cols.side in table:
        side = _side_column(table[cols.side])
        if np.any(side != SIDE_UNKNOWN):
            capabilities.add(Capability.SIDE)
    else:
        side = np.full(n, SIDE_UNKNOWN, dtype=np.int8)

    def optional(name: str, capability: Capability) -> np.ndarray:
        if name in table:
            values = _float_column(table[name], name)
            if np.any(np.isfinite(values)):
                capabilities.add(capability)
            return values
        return np.full(n, np.nan)

    price = optional(cols.price_change, Capability.PRICE)
    quantity = optional(cols.quantity, Capability.QUANTITY)

    if cols.session in table:
        session_labels = [str(v) for v in table[cols.session]]
    else:
        session_labels = [instruments[i] for i in instrument]
    # Sessions are ordered by first appearance in time so session ids follow the clock.
    first_time: dict[str, float] = {}
    for label, t in zip(session_labels, timestamp, strict=True):
        first_time[label] = min(first_time.get(label, np.inf), float(t))
    session_order = sorted(first_time, key=lambda s: (first_time[s], s))
    session_lookup = {s: i for i, s in enumerate(session_order)}
    session = np.array([session_lookup[s] for s in session_labels], dtype=np.int64)
    order = np.lexsort((timestamp, session))

    book_names = [getattr(cols, k) for k in BOOK_FIELDS]
    present = [name in table for name in book_names]
    book = None
    if all(present):
        book = {
            k: _float_column(table[name], k)[order]
            for k, name in zip(BOOK_FIELDS, book_names, strict=True)
        }
        capabilities.add(Capability.BOOK_TOP)
    elif any(present):
        raise SchemaError("book columns must be provided together: " + ", ".join(book_names))

    session_instrument: dict[int, int] = {}
    for s, inst in zip(session[order], instrument[order], strict=True):
        if session_instrument.setdefault(int(s), int(inst)) != int(inst):
            raise SchemaError("a session must contain a single instrument")
    sessions = [
        SessionInfo(i, session_instrument[i], label, open_time=first_time[label])
        for label, i in session_lookup.items()
    ]
    return EventFrame.from_columns(
        timestamp=timestamp[order],
        instrument=instrument[order],
        event_type=etype[order],
        side=side[order],
        price_change=price[order],
        quantity=quantity[order],
        session=session[order],
        event_types=vocab,
        instruments=instruments,
        provenance=provenance,
        capabilities=sorted(capabilities),
        sessions=sorted(sessions, key=lambda s: s.session),
        book=book,
    )


def read_csv(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    event_types: Sequence[str] | None = None,
    message_level: bool = False,
) -> EventFrame:
    """Read a generic event CSV (see the module docstring for the layout)."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError(f"{path}: empty CSV")
        table: dict[str, list[str]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                table[name].append(row[name])
    return frame_from_table(
        table,
        provenance=Provenance(str(path), "eventflow.marketdata.io.read_csv", file_sha256(path)),
        columns=columns,
        event_types=event_types,
        message_level=message_level,
    )


def frame_to_table(frame: EventFrame) -> dict[str, list[Any]]:
    """Inverse of :func:`frame_from_table` for the generic layout."""
    session_names = {s.session: s.name for s in frame.sessions}
    table: dict[str, list[Any]] = {
        "timestamp": frame.timestamp.tolist(),
        "instrument": [frame.instruments[i] for i in frame.instrument],
        "event_type": [frame.event_types[i] for i in frame.event_type],
        "side": ["" if s == SIDE_UNKNOWN else SIDE_NAMES[s] for s in frame.side],
        "price_change": frame.price_change.tolist(),
        "quantity": frame.quantity.tolist(),
        "session": [session_names.get(int(s), f"session{int(s)}") for s in frame.session],
    }
    if frame.book is not None:
        for k in BOOK_FIELDS:
            table[k] = frame.book[k].tolist()
    return table


def write_csv(frame: EventFrame, path: str | Path) -> Path:
    path = Path(path)
    table = frame_to_table(frame)
    names = list(table)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in zip(*(table[k] for k in names), strict=True):
            writer.writerow(["" if isinstance(v, float) and np.isnan(v) else v for v in row])
    return path


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Parquet support needs the optional extra: pip install 'eventflow[parquet]'"
        ) from exc
    return pq


def read_parquet(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    event_types: Sequence[str] | None = None,
    message_level: bool = False,
) -> EventFrame:
    """Read a generic event table from Parquet (requires ``pyarrow``)."""
    pq = _require_pyarrow()
    path = Path(path)
    table = pq.read_table(path).to_pydict()
    return frame_from_table(
        table,
        provenance=Provenance(str(path), "eventflow.marketdata.io.read_parquet", file_sha256(path)),
        columns=columns,
        event_types=event_types,
        message_level=message_level,
    )


def write_parquet(frame: EventFrame, path: str | Path) -> Path:
    pq = _require_pyarrow()
    import pyarrow as pa

    path = Path(path)
    pq.write_table(pa.table(frame_to_table(frame)), path)
    return path
