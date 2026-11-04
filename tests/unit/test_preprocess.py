from __future__ import annotations

import numpy as np
import pytest

from eventflow.errors import ConfigError, SchemaError
from eventflow.marketdata import SyntheticConfig, generate_synthetic
from eventflow.marketdata.preprocess import (
    PreprocessConfig,
    SplitConfig,
    chronological_split,
    fingerprint,
    preprocess,
)
from tests.unit.conftest import make_frame


@pytest.fixture(scope="module")
def frame():
    return generate_synthetic(
        SyntheticConfig(seed=2, sessions_per_instrument=5, events_per_session=100)
    )


def test_missing_values_error_by_default_and_drop_on_request() -> None:
    frame = make_frame(quantity=np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(SchemaError, match="missing"):
        preprocess(frame)
    result = preprocess(frame, PreprocessConfig(missing="drop"))
    assert len(result.frame) == 5
    assert result.log[0]["step"] == "drop_missing"


def test_drop_event_types_and_short_sessions() -> None:
    frame = make_frame()
    result = preprocess(frame, PreprocessConfig(drop_event_types=("trade",), min_session_events=2))
    assert 2 not in result.frame.event_type
    # Session 1 keeps one event after dropping its trade and is removed.
    assert result.frame.num_sessions == 1
    with pytest.raises(ConfigError):
        preprocess(frame, PreprocessConfig(drop_event_types=("nope",)))


def test_split_on_gap_creates_sessions_with_no_cross_gap_interarrival() -> None:
    frame = make_frame(timestamp=np.array([0.0, 1.0, 100.0, 101.0, 0.0, 1.0]))
    result = preprocess(frame, PreprocessConfig(max_gap=10.0))
    assert result.frame.num_sessions == 3
    assert np.nanmax(result.frame.interarrival) <= 10.0


def test_session_split_is_chronological_and_disjoint(frame) -> None:
    parts = chronological_split(frame, SplitConfig(0.6, 0.2, 0.2))
    assert parts.validation is not None and parts.test is not None
    assert parts.train.timestamp.max() < parts.validation.timestamp.min()
    assert parts.validation.timestamp.max() < parts.test.timestamp.min()
    assert len(parts.train) + len(parts.validation) + len(parts.test) == len(frame)
    assert len(parts.train) == 600


def test_time_split_applies_embargo(frame) -> None:
    parts = chronological_split(frame, SplitConfig(0.5, 0.25, 0.25, unit="time", embargo=5.0))
    assert parts.validation is not None
    gap = parts.validation.timestamp.min() - parts.train.timestamp.max()
    assert gap >= 5.0
    assert len(parts.train) + len(parts.validation) + len(parts.test) < len(frame)


def test_split_config_validation() -> None:
    with pytest.raises(ConfigError):
        SplitConfig(0.5, 0.5, 0.5)
    with pytest.raises(ConfigError):
        SplitConfig(0.0, 0.5, 0.5)


def test_fingerprint_is_content_sensitive(frame) -> None:
    same = generate_synthetic(
        SyntheticConfig(seed=2, sessions_per_instrument=5, events_per_session=100)
    )
    assert fingerprint(frame) == fingerprint(same)
    changed = frame.replace(quantity=frame.quantity + 1)
    assert fingerprint(changed) != fingerprint(frame)
