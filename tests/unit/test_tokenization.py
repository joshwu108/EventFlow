from __future__ import annotations

import numpy as np
import pytest

from eventflow.errors import CapabilityError, TokenizationError
from eventflow.marketdata import Capability, Provenance
from eventflow.marketdata.io import frame_from_table
from eventflow.tokenization import (
    LOG_GAP,
    LOG_QUANTITY,
    PRICE,
    ZERO_GAP,
    EventTokenizer,
    GroupStandardizer,
    TickCoder,
    TokenizerConfig,
)
from tests.unit.conftest import make_frame


@pytest.fixture(scope="module")
def tokenizer(small_frame):
    return EventTokenizer.fit(small_frame)


def test_layout_reflects_capabilities(tokenizer) -> None:
    assert tokenizer.layout.categorical_names == ("event_type", "side", "price", "zero_gap")
    assert tokenizer.layout.continuous == (LOG_QUANTITY, LOG_GAP)
    assert tokenizer.layout.categorical_sizes[2] == 17


def test_layout_without_side_or_price() -> None:
    table = {
        "timestamp": [0.0, 1.0, 1.0, 3.0],
        "instrument": ["X"] * 4,
        "event_type": ["t"] * 4,
        "quantity": [1, 2, 3, 4],
    }
    frame = frame_from_table(table, provenance=Provenance("t", "test"))
    tok = EventTokenizer.fit(frame)
    assert tok.layout.categorical_names == ("event_type", "zero_gap")
    decoded = tok.decode(*_codes(tok.encode(frame)))
    assert np.all(np.isnan(decoded.price_change))


def _codes(encoded):
    return encoded.categorical, encoded.continuous, encoded.instrument


def test_round_trip_is_exact_for_in_range_events(small_frame, tokenizer) -> None:
    encoded = tokenizer.encode(small_frame)
    decoded = tokenizer.decode(*_codes(encoded))
    inside = ~encoded.price_tail
    np.testing.assert_array_equal(decoded.event_type, small_frame.event_type)
    np.testing.assert_array_equal(decoded.side, small_frame.side)
    np.testing.assert_allclose(decoded.price_change[inside], small_frame.price_change[inside])
    np.testing.assert_allclose(decoded.quantity, small_frame.quantity, rtol=1e-6)
    gaps = small_frame.interarrival
    defined = np.isfinite(gaps)
    np.testing.assert_allclose(decoded.interarrival[defined], gaps[defined], rtol=1e-5)
    assert not decoded.invalid.any()


def test_zero_gaps_are_categorical_and_masked(small_frame, tokenizer) -> None:
    encoded = tokenizer.encode(small_frame)
    zero = encoded.categorical[:, tokenizer.layout.cat_index(ZERO_GAP)] == 1
    gap_mask = encoded.continuous_mask[:, tokenizer.layout.cont_index(LOG_GAP)]
    np.testing.assert_array_equal(zero, small_frame.interarrival == 0)
    assert not np.any(gap_mask & zero)
    # First events of sessions have undefined gaps: masked, not zero.
    starts = [a for a, _ in small_frame.session_bounds()]
    assert not gap_mask[starts].any() and not zero[starts].any()


def test_standardisation_uses_per_instrument_training_stats(small_frame, tokenizer) -> None:
    encoded = tokenizer.encode(small_frame)
    k = tokenizer.layout.cont_index(LOG_QUANTITY)
    for inst in range(len(small_frame.instruments)):
        z = encoded.continuous[encoded.instrument == inst, k]
        assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3


def test_tick_coder_tails_and_grid() -> None:
    coder = TickCoder(max_ticks=3)
    codes, tail = coder.encode(np.array([-10.0, -3.0, -1.0, 0.0, 2.0, 3.0, 99.0]))
    np.testing.assert_array_equal(codes, [0, 0, 2, 3, 5, 6, 6])
    np.testing.assert_array_equal(tail, [True, True, False, False, False, True, True])
    np.testing.assert_allclose(coder.decode(codes), [-3, -3, -1, 0, 2, 3, 3])
    assert coder.off_grid(np.array([0.5, 1.0])).tolist() == [True, False]
    with pytest.raises(TokenizationError):
        coder.encode(np.array([np.nan]))
    with pytest.raises(TokenizationError):
        coder.decode(np.array([7]))


def test_standardizer_unseen_group_falls_back_to_pooled() -> None:
    std = GroupStandardizer.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), num_groups=2)
    assert std.unseen_groups() == [1]
    z = std.forward(np.array([2.0]), np.array([1]))
    np.testing.assert_allclose(z, (2.0 - std.pooled_mean) / std.pooled_std)
    restored = GroupStandardizer.from_dict(std.to_dict())
    np.testing.assert_allclose(restored.inverse(z, np.array([1])), [2.0])


def test_decode_enforces_constraints_and_counts_corrections(tokenizer) -> None:
    layout = tokenizer.layout
    cat = np.zeros((3, len(layout.categorical)), dtype=np.int64)
    cat[:, layout.cat_index(PRICE)] = tokenizer.config.max_price_ticks  # zero change
    cont = np.array([[-50.0, 0.0], [0.0, np.nan], [0.0, -50.0]])
    decoded = tokenizer.decode(cat, cont, np.zeros(3, dtype=np.int64))
    assert np.all(decoded.quantity >= tokenizer.min_quantity)
    assert np.all(decoded.interarrival[[0, 2]] > 0)
    assert decoded.invalid.tolist() == [False, True, False]
    assert decoded.corrections["clipped_continuous"] == 2


def test_unobserved_categorical_combination_is_invalid(tokenizer) -> None:
    layout = tokenizer.layout
    cat = np.zeros((1, len(layout.categorical)), dtype=np.int64)
    cat[0, layout.cat_index(PRICE)] = 0  # limit_add with a -8 tick move never occurs
    decoded = tokenizer.decode(cat, np.zeros((1, 2)), np.zeros(1, dtype=np.int64))
    assert decoded.invalid[0]
    assert not tokenizer.categorical_constraints()[PRICE][0, 0]


def test_serialisation_round_trip(small_frame, tokenizer) -> None:
    restored = EventTokenizer.from_json(tokenizer.to_json())
    a, b = tokenizer.encode(small_frame), restored.encode(small_frame)
    np.testing.assert_array_equal(a.categorical, b.categorical)
    np.testing.assert_allclose(a.continuous, b.continuous)
    with pytest.raises(TokenizationError, match="version"):
        EventTokenizer.from_dict({**tokenizer.to_dict(), "version": 99})


def test_incompatible_frames_are_rejected(tokenizer) -> None:
    with pytest.raises(TokenizationError, match="vocabulary"):
        tokenizer.encode(make_frame())
    with pytest.raises(TokenizationError, match="missing"):
        EventTokenizer.fit(make_frame(quantity=np.array([1.0, np.nan, 1, 1, 1, 1])))


def test_capability_mismatch_is_rejected(small_frame, tokenizer) -> None:
    caps = [c for c in small_frame.capabilities if c is not Capability.SIDE]
    stripped = small_frame.replace(capabilities=caps)
    with pytest.raises(CapabilityError):
        tokenizer.encode(stripped)


def test_config_validation() -> None:
    with pytest.raises(TokenizationError):
        TokenizerConfig(max_price_ticks=0)
    with pytest.raises(TokenizationError):
        TokenizerConfig(zero_gap_threshold=-1)
