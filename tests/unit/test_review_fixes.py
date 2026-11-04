"""Regression tests for issues found in code review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from eventflow.generation import one_step_samples
from eventflow.marketdata import SyntheticConfig, generate_synthetic
from eventflow.marketdata.lobster import (
    LOBSTER_EVENT_TYPES,
    LobsterDay,
    read_lobster,
    write_lobster_fixture,
)
from eventflow.marketdata.preprocess import SplitConfig, chronological_split
from eventflow.tokenization import LOG_GAP, EventTokenizer, EventWindowDataset
from eventflow.training import TrainConfig, Trainer, build_optimizer
from tests.unit.model_fixtures import tiny_model

MPS = torch.backends.mps.is_available()


def _lobster_days(tmp_path: Path, days: int) -> list[LobsterDay]:
    out = []
    for d in range(days):
        frame = generate_synthetic(
            SyntheticConfig(
                seed=d, instruments=("X",), sessions_per_instrument=1, events_per_session=60
            )
        )
        # Every day opens at the same time of day, as real LOBSTER days do.
        frame = frame.replace(
            timestamp=frame.timestamp + 34_200.0,
            event_type=np.array([0, 2, 3])[frame.event_type],
            event_types=LOBSTER_EVENT_TYPES,
        )
        message = tmp_path / f"day{d}_message.csv"
        write_lobster_fixture(frame, message, tmp_path / f"day{d}_orderbook.csv")
        out.append(LobsterDay("X", message, tmp_path / f"day{d}_orderbook.csv", name=f"d{d}"))
    return out


@pytest.mark.parametrize("unit", ["session", "time"])
def test_lobster_days_split_in_date_order(tmp_path: Path, unit: str) -> None:
    frame = read_lobster(_lobster_days(tmp_path, 10))
    assert frame.timestamp.max() < 86_400  # raw seconds after midnight are kept
    parts = chronological_split(frame, SplitConfig(0.6, 0.2, 0.2, unit=unit))
    names = {k: {s.name.split("@")[0] for s in p.sessions} for k, p in parts.items()}
    assert names["train"] == {f"d{i}" for i in range(6)}
    assert names["test"] == {"d8", "d9"}


def test_weight_decay_applies_to_linear_weights_only(small_frame) -> None:
    model = tiny_model("hybrid", EventTokenizer.fit(small_frame))
    opt = build_optimizer(model, TrainConfig(max_steps=2, warmup_steps=0, device="cpu"))
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(model.encoder.embedding.continuous.weight) in decayed
    assert id(model.encoder.embedding.position.weight) not in decayed
    assert all(
        id(m.weight) not in decayed
        for m in model.modules()
        if isinstance(m, nn.Embedding | nn.LayerNorm) and getattr(m, "weight", None) is not None
    )
    assert id(model.continuous_head.time_embedding.mlp[0].weight) in decayed


def test_accumulation_matches_one_large_batch(small_frame, tmp_path: Path) -> None:
    tokenizer = EventTokenizer.fit(small_frame)
    data = EventWindowDataset(tokenizer.encode(small_frame), context_length=16)
    results = []
    for batch, accumulation in ((4, 1), (2, 2)):
        torch.manual_seed(0)
        model = tiny_model("transformer", tokenizer)  # deterministic loss (no sampled noise)
        cfg = TrainConfig(
            max_steps=1,
            batch_size=batch,
            grad_accumulation=accumulation,
            optimizer="sgd",
            lr=0.1,
            momentum=0.0,
            warmup_steps=0,
            scheduler="constant",
            grad_clip=None,
            device="cpu",
        )
        Trainer(model, data, cfg, tmp_path / f"{batch}x{accumulation}").fit()
        results.append([p.detach().clone() for p in model.parameters()])
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_generated_gaps_are_floored_at_smallest_training_gap(small_frame) -> None:
    tokenizer = EventTokenizer.fit(small_frame)
    gaps = small_frame.interarrival
    assert tokenizer.min_positive_gap == pytest.approx(gaps[gaps > 0].min())
    cat = np.zeros((1, len(tokenizer.layout.categorical)), dtype=np.int64)
    cat[0, tokenizer.layout.cat_index("price")] = tokenizer.config.max_price_ticks
    cont = np.zeros((1, len(tokenizer.layout.continuous)))
    cont[0, tokenizer.layout.cont_index(LOG_GAP)] = -8.0
    decoded = tokenizer.decode(cat, cont, np.zeros(1, dtype=np.int64))
    assert decoded.interarrival[0] >= tokenizer.min_positive_gap
    restored = EventTokenizer.from_json(tokenizer.to_json())
    assert restored.min_positive_gap == tokenizer.min_positive_gap


def test_restore_best_loads_best_weights(small_frame, tmp_path: Path) -> None:
    tokenizer = EventTokenizer.fit(small_frame)
    data = EventWindowDataset(tokenizer.encode(small_frame), context_length=16)
    cfg = TrainConfig(
        max_steps=4,
        batch_size=4,
        warmup_steps=0,
        eval_every=2,
        eval_batches=1,
        checkpoint_every=10,
        device="cpu",
    )
    trainer = Trainer(tiny_model("hybrid", tokenizer), data, cfg, tmp_path, validation_data=data)
    trainer.fit()
    assert trainer.restore_best()


@pytest.mark.skipif(not MPS, reason="needs an accelerator (MPS) to exercise device paths")
def test_resume_and_one_step_on_accelerator(small_frame, tmp_path: Path) -> None:
    tokenizer = EventTokenizer.fit(small_frame)
    encoded = tokenizer.encode(small_frame)
    data = EventWindowDataset(encoded, context_length=16)
    cfg = TrainConfig(max_steps=2, batch_size=2, warmup_steps=0, checkpoint_every=1, device="mps")
    Trainer(tiny_model("hybrid", tokenizer), data, cfg, tmp_path).fit()
    resumed = Trainer(
        tiny_model("hybrid", tokenizer),
        data,
        TrainConfig.from_dict({**cfg.to_dict(), "max_steps": 3}),
        tmp_path,
    )
    assert resumed.resume()
    model = resumed.model
    fake, real = one_step_samples(model, tokenizer, encoded, 8, 4, device="mps")
    assert len(fake) == len(real) == 4
