"""Adapter for LOBSTER-format message and order-book files.

LOBSTER (https://lobsterdata.com) distributes, per instrument and day, a
*message* file and an optional *order book* file with one row per message.
This adapter only parses that published layout; it ships no LOBSTER data and
downloads nothing (the data is licensed by its provider).

Message file columns (no header)::

    time       seconds after midnight (float, nanosecond resolution)
    type       1 submission, 2 partial cancellation, 3 deletion,
               4 visible execution, 5 hidden execution, 6 cross trade,
               7 trading halt indicator
    order_id   order reference
    size       shares
    price      dollar price x 10000
    direction  1 = buy limit order, -1 = sell limit order

For executions ``direction`` is the side of the *resting* limit order that was
executed; a buy-initiated trade therefore has direction -1. EventFlow keeps
the source convention and documents it rather than silently flipping it.

Order book file columns (no header), ``levels`` repetitions of::

    ask_price_k, ask_size_k, bid_price_k, bid_size_k

Price change semantics
----------------------
With an order book file, ``price_change`` is the event price minus the mid
price *before* the event, in ticks; the first event of a day is measured
against the mid after it (no earlier state exists). Without a book file it is
the event price minus the previous message price, in ticks. The two are not
interchangeable, so the provenance records which one was used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eventflow.errors import SchemaError
from eventflow.marketdata.io import file_sha256
from eventflow.marketdata.schema import (
    SIDE_BUY,
    SIDE_SELL,
    Capability,
    EventFrame,
    Provenance,
    SessionInfo,
)

LOBSTER_EVENT_TYPES: tuple[str, ...] = (
    "submission",
    "cancel_partial",
    "deletion",
    "execution_visible",
    "execution_hidden",
    "cross_trade",
    "halt",
)
PRICE_SCALE = 10_000.0  # LOBSTER prices are dollars x 10000.
HALT_TYPE = 7


@dataclass(frozen=True)
class LobsterDay:
    """One instrument-day: a message file and an optional order book file."""

    instrument: str
    message_path: Path
    orderbook_path: Path | None = None
    name: str | None = None
    day_index: int | None = None


SECONDS_PER_DAY = 86_400.0


def _load_matrix(path: Path, min_columns: int) -> np.ndarray:
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise SchemaError(f"{path}: not a numeric LOBSTER file") from exc
    if data.shape[1] < min_columns:
        raise SchemaError(f"{path}: expected at least {min_columns} columns, got {data.shape[1]}")
    return data


def read_lobster(
    days: Sequence[LobsterDay],
    *,
    tick_size: float = 0.01,
    drop_halts: bool = True,
) -> EventFrame:
    """Parse LOBSTER instrument-days into one frame (one session per day).

    ``tick_size`` is in dollars. Halt indicator rows (type 7) carry no order
    information and are dropped by default.

    Timestamps stay seconds after midnight (full precision); each session's
    ``clock_offset`` is ``86400 * day_index`` so that chronological splits
    order days correctly. ``day_index`` defaults to the position in
    ``days``, which is only correct for one instrument listed in date order;
    with several instruments per date, set ``day_index`` explicitly so that
    sessions of the same date share an offset.
    """
    if not days:
        raise SchemaError("read_lobster needs at least one LobsterDay")
    if tick_size <= 0:
        raise SchemaError("tick_size must be positive")
    instruments = tuple(dict.fromkeys(day.instrument for day in days))
    tick = tick_size * PRICE_SCALE
    pieces: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "timestamp",
            "instrument",
            "event_type",
            "side",
            "price_change",
            "quantity",
            "session",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
        )
    }
    sessions: list[SessionInfo] = []
    hashes: list[str] = []
    with_book = all(day.orderbook_path is not None for day in days)
    if not with_book and any(day.orderbook_path is not None for day in days):
        raise SchemaError("either every day or no day must provide an order book file")

    for index, day in enumerate(days):
        messages = _load_matrix(day.message_path, 6)
        hashes.append(file_sha256(day.message_path))
        types = messages[:, 1].astype(np.int64)
        if np.any((types < 1) | (types > HALT_TYPE)):
            raise SchemaError(f"{day.message_path}: event type outside 1..7")
        book = None
        if with_book:
            assert day.orderbook_path is not None
            book = _load_matrix(day.orderbook_path, 4)
            hashes.append(file_sha256(day.orderbook_path))
            if book.shape[0] != messages.shape[0]:
                raise SchemaError(f"{day.orderbook_path}: row count differs from message file")
        keep = types != HALT_TYPE if drop_halts else np.ones(types.shape, dtype=bool)
        messages, types = messages[keep], types[keep]
        price = messages[:, 4]
        if book is not None:
            book = book[keep]
            ask, ask_size, bid, bid_size = (book[:, i] for i in range(4))
            mid_after = 0.5 * (ask + bid)
            mid_before = np.concatenate([mid_after[:1], mid_after[:-1]])
            change = (price - mid_before) / tick
            for name, values in (
                ("bid", bid / tick),
                ("ask", ask / tick),
                ("bid_size", bid_size),
                ("ask_size", ask_size),
            ):
                pieces[name].append(values)
        else:
            previous = np.concatenate([price[:1], price[:-1]])
            change = (price - previous) / tick
        direction = messages[:, 5]
        if np.any(~np.isin(direction, (1.0, -1.0))):
            raise SchemaError(f"{day.message_path}: direction must be 1 or -1")
        n = messages.shape[0]
        # LOBSTER times are seconds after midnight with no date. The day offset
        # goes into SessionInfo.clock_offset (used to order sessions) instead of
        # the timestamps, which would lose nanosecond resolution over long spans.
        day_number = index if day.day_index is None else day.day_index
        offset = day_number * SECONDS_PER_DAY
        pieces["timestamp"].append(messages[:, 0])
        pieces["instrument"].append(np.full(n, instruments.index(day.instrument)))
        pieces["event_type"].append(types - 1)
        pieces["side"].append(np.where(direction > 0, SIDE_BUY, SIDE_SELL))
        pieces["price_change"].append(change)
        pieces["quantity"].append(messages[:, 3])
        pieces["session"].append(np.full(n, index))
        name = day.name or f"{day.instrument}/{day.message_path.stem}"
        open_time = float(messages[0, 0]) if n else None
        sessions.append(
            SessionInfo(index, instruments.index(day.instrument), name, open_time, offset)
        )

    def cat(key: str) -> np.ndarray:
        return np.concatenate(pieces[key])

    capabilities = [
        Capability.TIMESTAMPS,
        Capability.SIDE,
        Capability.PRICE,
        Capability.QUANTITY,
        Capability.MULTI_EVENT_TYPE,
        Capability.MESSAGE_LEVEL,
    ]
    if with_book:
        capabilities.append(Capability.BOOK_TOP)
    return EventFrame.from_columns(
        timestamp=cat("timestamp"),
        instrument=cat("instrument"),
        event_type=cat("event_type"),
        side=cat("side"),
        price_change=cat("price_change"),
        quantity=cat("quantity"),
        session=cat("session"),
        event_types=LOBSTER_EVENT_TYPES,
        instruments=instruments,
        provenance=Provenance(
            source=",".join(str(d.message_path) for d in days),
            adapter="eventflow.marketdata.lobster.read_lobster",
            content_sha256=hashes[0] if len(hashes) == 1 else _combine(hashes),
            parameters={
                "tick_size": tick_size,
                "drop_halts": drop_halts,
                "price_change_reference": "mid_before" if with_book else "previous_message_price",
                "side_convention": "resting_limit_order",
            },
        ),
        capabilities=capabilities,
        sessions=sessions,
        book={k: cat(k) for k in ("bid", "ask", "bid_size", "ask_size")} if with_book else None,
    )


def _combine(hashes: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def write_lobster_fixture(
    frame: EventFrame,
    message_path: Path,
    orderbook_path: Path | None = None,
    *,
    tick_size: float = 0.01,
) -> None:
    """Write a synthetic frame in LOBSTER layout (for adapter tests only).

    The frame's event types must already use :data:`LOBSTER_EVENT_TYPES`
    indices; prices are reconstructed from the book mid when available.
    """
    tick = tick_size * PRICE_SCALE
    if frame.book is None:
        raise SchemaError("fixture writer needs a frame with top of book")
    bid = frame.book["bid"] * tick
    ask = frame.book["ask"] * tick
    mid_after = 0.5 * (bid + ask)
    mid_before = np.concatenate([mid_after[:1], mid_after[:-1]])
    price = mid_before + frame.price_change * tick
    direction = np.where(frame.side == SIDE_BUY, 1, -1)
    rows = np.column_stack(
        [
            frame.timestamp,
            frame.event_type + 1,
            np.arange(len(frame)),
            frame.quantity,
            price,
            direction,
        ]
    )
    np.savetxt(message_path, rows, delimiter=",", fmt=["%.9f", "%d", "%d", "%d", "%.1f", "%d"])
    if orderbook_path is not None:
        book = np.column_stack([ask, frame.book["ask_size"], bid, frame.book["bid_size"]])
        np.savetxt(orderbook_path, book, delimiter=",", fmt=["%.1f", "%d", "%.1f", "%d"])
