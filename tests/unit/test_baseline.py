from __future__ import annotations

import numpy as np
import pytest
import torch

from eventflow.errors import ConfigError
from eventflow.models.baseline import MarkovBaseline
from eventflow.tokenization import PRICE, EventTokenizer
from tests.unit.model_fixtures import tiny_batch


@pytest.fixture(scope="module")
def fitted(small_frame):
    tokenizer = EventTokenizer.fit(small_frame)
    encoded = tokenizer.encode(small_frame)
    model = MarkovBaseline.fit(encoded, tokenizer.layout, len(tokenizer.instruments))
    return tokenizer, encoded, model


def test_transition_rows_are_distributions(fitted) -> None:
    _, _, model = fitted
    probs = model.transition_probs(0, 0)
    assert probs.min() > 0 and probs.sum() == pytest.approx(1.0)


def test_nll_is_finite_and_beats_uniform(fitted) -> None:
    _, encoded, model = fitted
    nll = model.categorical_nll(encoded)
    assert np.isfinite(nll)
    assert nll < np.log(len(model.states))


def test_samples_are_observed_states_with_real_continuous_values(fitted, small_frame) -> None:
    tokenizer, encoded, model = fitted
    batch = tiny_batch(small_frame, tokenizer)
    nxt = model.sample_next(batch, torch.Generator().manual_seed(0))
    for row in nxt.categorical.numpy():
        assert tuple(int(v) for v in row) in model.states
    allowed = tokenizer.categorical_constraints()[PRICE]
    k = tokenizer.layout.cat_index(PRICE)
    cats = nxt.categorical.numpy()
    assert allowed[cats[:, 0], cats[:, k]].all()
    pool = {tuple(np.round(r, 5)) for r in (encoded.continuous * encoded.continuous_mask)}
    for row in nxt.continuous.numpy():
        assert tuple(np.round(row, 5)) in pool


def test_sampling_is_deterministic(fitted, small_frame) -> None:
    tokenizer, _, model = fitted
    batch = tiny_batch(small_frame, tokenizer)
    a = model.sample_next(batch, torch.Generator().manual_seed(5))
    b = model.sample_next(batch, torch.Generator().manual_seed(5))
    assert torch.equal(a.categorical, b.categorical)


def test_invalid_smoothing_is_rejected(fitted) -> None:
    tokenizer, _, _ = fitted
    with pytest.raises(ConfigError):
        MarkovBaseline(tokenizer.layout, 1, smoothing=0.0)
