from __future__ import annotations

import numpy as np
import pytest
import torch

from eventflow.errors import ConfigError
from eventflow.generation import one_step_samples, rollout, sample_prompts
from eventflow.marketdata import Capability, validate_frame
from eventflow.models.baseline import MarkovBaseline
from eventflow.tokenization import EventTokenizer
from tests.unit.model_fixtures import KINDS, tiny_model


@pytest.fixture(scope="module")
def setup(small_frame):
    tokenizer = EventTokenizer.fit(small_frame)
    return tokenizer, tokenizer.encode(small_frame)


def test_prompts_are_session_local_with_true_continuations(setup) -> None:
    _, encoded = setup
    prompts = sample_prompts(encoded, context_length=10, count=6, horizon=3, seed=0)
    assert prompts.context.categorical.shape[:2] == (6, 10)
    assert bool(prompts.context.valid.all())
    assert prompts.continuation is not None and len(prompts.continuation) == 18
    with pytest.raises(ConfigError):
        sample_prompts(encoded, context_length=10_000, count=1)


@pytest.mark.parametrize("kind", KINDS)
def test_rollout_produces_valid_frames_beyond_context(kind, setup) -> None:
    tokenizer, encoded = setup
    model = tiny_model(kind, tokenizer, max_context=8)
    prompts = sample_prompts(encoded, context_length=8, count=3, seed=1)
    result = rollout(
        model,
        tokenizer,
        prompts,
        horizon=12,
        max_context=8,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.categorical.shape[:2] == (3, 12)
    assert len(result.frame) == 36 and result.frame.num_sessions == 3
    assert validate_frame(result.frame).ok
    assert np.all(result.frame.quantity > 0)
    assert np.all(np.nan_to_num(result.frame.interarrival) >= 0)
    assert not result.frame.has(Capability.BOOK_TOP)
    assert result.invalid_rate_by_step().shape == (12,)
    # Timestamps continue from the prompt's last event.
    assert np.all(result.frame.timestamp.reshape(3, 12)[:, 0] >= prompts.start_timestamp)


def test_rollout_is_reproducible(setup) -> None:
    tokenizer, encoded = setup
    model = tiny_model("hybrid", tokenizer)
    prompts = sample_prompts(encoded, context_length=8, count=2, seed=2)
    a = rollout(
        model, tokenizer, prompts, 5, max_context=16, generator=torch.Generator().manual_seed(9)
    )
    b = rollout(
        model, tokenizer, prompts, 5, max_context=16, generator=torch.Generator().manual_seed(9)
    )
    np.testing.assert_array_equal(a.categorical, b.categorical)
    np.testing.assert_allclose(a.continuous, b.continuous)


def test_baseline_rollout_and_one_step(setup) -> None:
    tokenizer, encoded = setup
    baseline = MarkovBaseline.fit(encoded, tokenizer.layout, len(tokenizer.instruments))
    prompts = sample_prompts(encoded, context_length=4, count=4, seed=0)
    result = rollout(
        baseline, tokenizer, prompts, 20, max_context=4, generator=torch.Generator().manual_seed(0)
    )
    assert result.invalid_rate == 0.0
    fake, real = one_step_samples(baseline, tokenizer, encoded, context_length=4, count=50)
    assert len(fake) == len(real) == 50


def test_padded_prompts_are_rejected(setup) -> None:
    tokenizer, encoded = setup
    prompts = sample_prompts(encoded, context_length=4, count=2, seed=0)
    ctx = prompts.context
    padded = type(ctx)(
        ctx.categorical,
        ctx.continuous,
        ctx.continuous_mask,
        ctx.instrument,
        ctx.session_time,
        ctx.valid.clone().index_fill_(1, torch.tensor([3]), False),
    )
    with pytest.raises(ConfigError, match="unpadded"):
        rollout(
            tiny_model("hybrid", tokenizer),
            tokenizer,
            type(prompts)(padded, None, prompts.start_timestamp),
            2,
            max_context=4,
        )
