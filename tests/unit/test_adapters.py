from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eventflow.errors import SchemaError
from eventflow.marketdata import Capability, SyntheticConfig, generate_synthetic
from eventflow.marketdata.io import (
    ColumnMap,
    frame_from_table,
    read_csv,
    read_parquet,
    write_csv,
    write_parquet,
)
from eventflow.marketdata.lobster import (
    LOBSTER_EVENT_TYPES,
    LobsterDay,
    read_lobster,
    write_lobster_fixture,
)
from eventflow.marketdata.preprocess import fingerprint
from eventflow.marketdata.schema import Provenance


@pytest.fixture(scope="module")
def frame():
    return generate_synthetic(
        SyntheticConfig(seed=1, sessions_per_instrument=1, events_per_session=200)
    )


def _assert_same_events(a, b) -> None:
    for name in (
        "timestamp",
        "instrument",
        "event_type",
        "side",
        "price_change",
        "quantity",
        "session",
    ):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), err_msg=name)


def test_csv_round_trip(frame, tmp_path: Path) -> None:
    path = write_csv(frame, tmp_path / "events.csv")
    loaded = read_csv(path, event_types=frame.event_types)
    _assert_same_events(frame, loaded)
    assert loaded.has(Capability.BOOK_TOP)
    assert not loaded.has(Capability.MESSAGE_LEVEL)  # never inferred from a generic table
    assert loaded.provenance.content_sha256 is not None


def test_parquet_round_trip(frame, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = write_parquet(frame, tmp_path / "events.parquet")
    loaded = read_parquet(path, event_types=frame.event_types, message_level=True)
    _assert_same_events(frame, loaded)
    assert loaded.has(Capability.MESSAGE_LEVEL)
    assert fingerprint(loaded) == fingerprint(
        read_parquet(path, event_types=frame.event_types, message_level=True)
    )


def test_trade_prints_do_not_claim_order_book_capabilities() -> None:
    table = {
        "timestamp": [0.0, 1.0, 2.5],
        "instrument": ["X"] * 3,
        "event_type": ["trade"] * 3,
        "price_change": [0, 1, -1],
        "quantity": [100, 5, 7],
    }
    frame = frame_from_table(table, provenance=Provenance("t", "test"))
    assert frame.has(Capability.PRICE) and frame.has(Capability.QUANTITY)
    for cap in (
        Capability.MULTI_EVENT_TYPE,
        Capability.MESSAGE_LEVEL,
        Capability.BOOK_TOP,
        Capability.SIDE,
    ):
        assert not frame.has(cap)


def test_rows_are_sorted_by_session_then_time_and_columns_mapped() -> None:
    table = {
        "t": [3.0, 1.0, 2.0],
        "sym": ["X", "X", "Y"],
        "kind": ["a", "b", "a"],
        "side": ["sell", "buy", ""],
    }
    frame = frame_from_table(
        table,
        provenance=Provenance("t", "test"),
        columns=ColumnMap(timestamp="t", instrument="sym", event_type="kind"),
    )
    np.testing.assert_allclose(frame.timestamp, [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(frame.side, [0, 1, 2])
    assert [s.name for s in frame.sessions] == ["X", "Y"]


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ({"timestamp": [0.0], "instrument": ["X"]}, "event_type"),
        ({"timestamp": ["x"], "instrument": ["X"], "event_type": ["a"]}, "not numeric"),
        ({"timestamp": [0.0], "instrument": ["X"], "event_type": ["a"], "side": ["up"]}, "side"),
        ({"timestamp": [0.0], "instrument": ["X"], "event_type": ["a"], "bid": [1.0]}, "together"),
    ],
)
def test_malformed_tables_are_rejected(table, match) -> None:
    with pytest.raises(SchemaError, match=match):
        frame_from_table(table, provenance=Provenance("t", "test"))


def test_undeclared_event_type_is_rejected() -> None:
    table = {"timestamp": [0.0], "instrument": ["X"], "event_type": ["mystery"]}
    with pytest.raises(SchemaError, match="vocabulary"):
        frame_from_table(table, provenance=Provenance("t", "test"), event_types=("trade",))


def _lobster_frame(frame):
    # Map synthetic types onto LOBSTER codes: add->submission, cancel->deletion, trade->execution.
    mapping = np.array([0, 2, 3])
    return frame.replace(event_type=mapping[frame.event_type], event_types=LOBSTER_EVENT_TYPES)


def test_lobster_round_trip_with_book(frame, tmp_path: Path) -> None:
    source = _lobster_frame(frame.take(frame.session == 0))
    msg, book = tmp_path / "X_message_1.csv", tmp_path / "X_orderbook_1.csv"
    write_lobster_fixture(source, msg, book)
    loaded = read_lobster([LobsterDay("X", msg, book)])
    np.testing.assert_allclose(loaded.price_change, source.price_change, atol=1e-6)
    np.testing.assert_array_equal(loaded.event_type, source.event_type)
    np.testing.assert_array_equal(loaded.side, source.side)
    np.testing.assert_allclose(loaded.quantity, source.quantity)
    assert loaded.has(Capability.MESSAGE_LEVEL) and loaded.has(Capability.BOOK_TOP)
    assert loaded.provenance.parameters["price_change_reference"] == "mid_before"


def test_lobster_without_book_uses_previous_price(frame, tmp_path: Path) -> None:
    source = _lobster_frame(frame.take(frame.session == 0))
    msg = tmp_path / "X_message_1.csv"
    write_lobster_fixture(source, msg)
    loaded = read_lobster([LobsterDay("X", msg)])
    assert not loaded.has(Capability.BOOK_TOP)
    assert loaded.price_change[0] == 0.0
    assert loaded.provenance.parameters["price_change_reference"] == "previous_message_price"


def test_lobster_halts_dropped_and_bad_types_rejected(tmp_path: Path) -> None:
    msg = tmp_path / "m.csv"
    msg.write_text("1.0,1,1,100,1000000,1\n2.0,7,0,0,-1,-1\n3.0,4,1,50,1000000,1\n")
    loaded = read_lobster([LobsterDay("X", msg)])
    assert len(loaded) == 2
    msg.write_text("1.0,9,1,100,1000000,1\n")
    with pytest.raises(SchemaError, match="1..7"):
        read_lobster([LobsterDay("X", msg)])
