from __future__ import annotations

import numpy as np
import pytest

from eventflow.errors import CapabilityError, SchemaError
from eventflow.marketdata import Capability, validate_frame
from eventflow.marketdata.schema import compute_interarrival
from tests.unit.conftest import make_frame


def test_interarrival_is_nan_at_session_starts_and_exact_elsewhere() -> None:
    frame = make_frame()
    expected = np.array([np.nan, 0.5, 0.0, 1.5, np.nan, 0.25])
    np.testing.assert_array_equal(np.isnan(frame.interarrival), np.isnan(expected))
    np.testing.assert_allclose(
        frame.interarrival[~np.isnan(expected)], expected[~np.isnan(expected)]
    )


def test_arrays_are_read_only() -> None:
    frame = make_frame()
    with pytest.raises(ValueError):
        frame.quantity[0] = 1.0


def test_decreasing_timestamp_within_session_is_rejected() -> None:
    with pytest.raises(SchemaError, match="timestamp_order"):
        make_frame(timestamp=np.array([0.0, 1.0, 0.5, 2.0, 10.0, 11.0]))


def test_decreasing_timestamp_across_sessions_is_allowed() -> None:
    frame = make_frame(timestamp=np.array([5.0, 6.0, 7.0, 8.0, 0.0, 1.0]))
    assert len(frame) == 6


def test_non_contiguous_sessions_are_rejected() -> None:
    with pytest.raises(SchemaError, match="session_contiguous"):
        make_frame(session=np.array([0, 1, 0, 1, 1, 1]))


@pytest.mark.parametrize(
    ("field", "values", "check"),
    [
        ("quantity", np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0]), "quantity_positive"),
        ("side", np.array([0, 1, 3, 0, 0, 0]), "side_range"),
        ("event_type", np.array([0, 1, 5, 0, 0, 0]), "event_type_vocab"),
        ("quantity", np.array([1.0, np.inf, 1.0, 1.0, 1.0, 1.0]), "finite_values"),
    ],
)
def test_range_violations_are_rejected(field: str, values: np.ndarray, check: str) -> None:
    with pytest.raises(SchemaError, match=check):
        make_frame(**{field: values})


def test_non_integral_ids_are_rejected() -> None:
    with pytest.raises(SchemaError, match="non-integral"):
        make_frame(event_type=np.array([0.0, 1.5, 2.0, 0.0, 1.0, 1.0]))


def test_missing_values_are_reported_as_warnings() -> None:
    frame = make_frame(quantity=np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0]))
    report = validate_frame(frame)
    assert report.ok
    assert report.missing_fraction["quantity"] == pytest.approx(1 / 6)
    assert any(issue.check == "missing_values" for issue in report.issues)


def test_crossed_book_is_rejected() -> None:
    book = {
        "bid": np.full(6, 100.0),
        "ask": np.array([101.0, 101.0, 99.0, 101.0, 101.0, 101.0]),
        "bid_size": np.ones(6),
        "ask_size": np.ones(6),
    }
    with pytest.raises(SchemaError, match="book_crossed"):
        make_frame(book=book)


def test_capabilities_cannot_exceed_what_arrays_support() -> None:
    frame = make_frame()
    assert not frame.has(Capability.BOOK_TOP)
    assert not frame.has(Capability.MESSAGE_LEVEL)  # never inferred, only declared
    with pytest.raises(CapabilityError, match="book_top"):
        frame.require(Capability.BOOK_TOP, purpose="spread metric")


def test_take_recomputes_interarrival_from_kept_rows() -> None:
    frame = make_frame()
    sub = frame.take(np.array([0, 3, 4, 5]))
    np.testing.assert_allclose(sub.interarrival[1], 2.0)
    assert np.isnan(sub.interarrival[2])


def test_replace_returns_new_validated_frame() -> None:
    frame = make_frame()
    doubled = frame.replace(quantity=frame.quantity * 2)
    np.testing.assert_allclose(doubled.quantity, frame.quantity * 2)
    with pytest.raises(SchemaError):
        frame.replace(quantity=-frame.quantity)
    with pytest.raises(TypeError):
        frame.replace(nonsense=1)


def test_session_bounds() -> None:
    assert make_frame().session_bounds() == [(0, 4), (4, 6)]


def test_compute_interarrival_on_single_event() -> None:
    out = compute_interarrival(np.array([1.0]), np.array([0]))
    assert np.isnan(out[0])
