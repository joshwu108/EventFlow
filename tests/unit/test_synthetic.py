from __future__ import annotations

import numpy as np
import pytest

from eventflow.errors import ConfigError
from eventflow.marketdata import Capability, SyntheticConfig, generate_synthetic, validate_frame
from eventflow.marketdata.synthetic import TRADE

CFG = SyntheticConfig(seed=3, sessions_per_instrument=2, events_per_session=1500)


@pytest.fixture(scope="module")
def frame():
    return generate_synthetic(CFG)


def test_generation_is_deterministic(frame) -> None:
    again = generate_synthetic(CFG)
    for name, array in frame.columns().items():
        np.testing.assert_array_equal(array, again.columns()[name], err_msg=name)


def test_different_seed_changes_output(frame) -> None:
    other = generate_synthetic(
        SyntheticConfig(seed=4, sessions_per_instrument=2, events_per_session=1500)
    )
    assert not np.array_equal(frame.quantity, other.quantity)


def test_frame_is_valid_and_declares_capabilities(frame) -> None:
    assert validate_frame(frame).ok
    for cap in (Capability.MESSAGE_LEVEL, Capability.BOOK_TOP, Capability.LABELS):
        assert frame.has(cap)
    assert len(frame) == 2 * 2 * 1500
    assert frame.num_sessions == 4


def test_stylised_properties(frame) -> None:
    assert np.all(frame.quantity >= 1) and np.all(frame.quantity == np.round(frame.quantity))
    gaps = frame.interarrival[~np.isnan(frame.interarrival)]
    assert np.all(gaps >= 0)
    assert 0.1 < np.mean(gaps == 0) < 0.2  # zero_gap_prob = 0.15
    # Limit additions never move the price.
    assert np.all(frame.price_change[frame.event_type == 0] == 0)
    # Volatile regime has larger absolute moves among trades.
    regime = frame.labels["regime"]
    trades = frame.event_type == TRADE
    calm = np.abs(frame.price_change[trades & (regime == 0)]).mean()
    volatile = np.abs(frame.price_change[trades & (regime == 1)]).mean()
    assert volatile > calm


def test_heavy_quantity_tail(frame) -> None:
    q = frame.quantity
    assert np.quantile(q, 0.999) > 10 * np.median(q)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instruments": ()},
        {"events_per_session": 1},
        {"side_persistence": 1.5},
        {"move_prob": (0.0, 0.5)},
        {"transitions": ((1.0, 0.0, 0.0),) * 2},
        {"transitions": ((0.5, 0.5, 0.5),) * 3},
    ],
)
def test_invalid_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        SyntheticConfig(**kwargs)
