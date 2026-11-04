"""The context encoder must never see the event it predicts."""

from __future__ import annotations

import pytest
import torch

from eventflow.tokenization import EventBatch, EventTokenizer
from tests.unit.model_fixtures import KINDS, tiny_batch, tiny_model


@pytest.fixture(scope="module")
def setup(small_frame):
    tokenizer = EventTokenizer.fit(small_frame)
    return tokenizer, tiny_batch(small_frame, tokenizer, context=12, size=3)


def _perturb_from(batch: EventBatch, start: int) -> EventBatch:
    cat = batch.categorical.clone()
    cat[:, start:, 0] = (cat[:, start:, 0] + 1) % 3
    cont = batch.continuous.clone()
    cont[:, start:] += 5.0
    stime = batch.session_time.clone()
    stime[:, start:] += 100.0
    return EventBatch(cat, cont, batch.continuous_mask, batch.instrument, stime, batch.valid)


@pytest.mark.parametrize("kind", KINDS)
def test_encoder_outputs_ignore_future_events(kind, setup) -> None:
    tokenizer, batch = setup
    model = tiny_model(kind, tokenizer).eval()
    t = 6
    with torch.no_grad():
        base = model.encoder(batch)
        changed = model.encoder(_perturb_from(batch, t + 1))
    torch.testing.assert_close(base[:, : t + 1], changed[:, : t + 1])
    assert not torch.allclose(base[:, t + 1 :], changed[:, t + 1 :])


def test_padding_does_not_change_valid_positions(setup) -> None:
    tokenizer, batch = setup
    model = tiny_model("hybrid", tokenizer).eval()
    n = 7
    short = EventBatch(
        batch.categorical[:, :n],
        batch.continuous[:, :n],
        batch.continuous_mask[:, :n],
        batch.instrument,
        batch.session_time[:, :n],
        batch.valid[:, :n],
    )
    garbage = _perturb_from(batch, n)
    padded = EventBatch(
        garbage.categorical,
        garbage.continuous,
        garbage.continuous_mask,
        garbage.instrument,
        garbage.session_time,
        torch.arange(batch.length)[None].expand_as(batch.valid) < n,
    )
    with torch.no_grad():
        torch.testing.assert_close(model.encoder(short), model.encoder(padded)[:, :n])


def test_training_targets_are_shifted_by_one(setup) -> None:
    tokenizer, batch = setup
    cat, _, _, valid = batch.targets()
    assert torch.equal(cat, batch.categorical[:, 1:])
    assert torch.equal(batch.inputs().categorical, batch.categorical[:, :-1])
    assert valid.shape == (batch.batch_size, batch.length - 1)


def test_loss_does_not_depend_on_inputs_after_the_last_target(setup) -> None:
    """Perturbing the final event changes only its own target term, never earlier terms."""
    tokenizer, batch = setup
    model = tiny_model("transformer", tokenizer).eval()
    with torch.no_grad():
        h1, *_ = model.encode_targets(batch)
        h2, *_ = model.encode_targets(_perturb_from(batch, batch.length - 1))
    torch.testing.assert_close(h1, h2)
