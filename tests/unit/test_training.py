from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eventflow.errors import CheckpointError, ConfigError, NumericalError
from eventflow.models.generative import LossOutput
from eventflow.tokenization import EventTokenizer, EventWindowDataset
from eventflow.training import (
    MetricLogger,
    TrainConfig,
    Trainer,
    atomic_write_json,
    load_checkpoint,
    lr_lambda,
    save_checkpoint,
)
from eventflow.training.checkpoint import prune_checkpoints
from tests.unit.model_fixtures import tiny_model


@pytest.fixture(scope="module")
def data(small_frame):
    tokenizer = EventTokenizer.fit(small_frame)
    encoded = tokenizer.encode(small_frame)
    return tokenizer, EventWindowDataset(encoded, context_length=16)


def _cfg(**kw) -> TrainConfig:
    base = dict(
        max_steps=6,
        batch_size=4,
        warmup_steps=1,
        eval_every=3,
        eval_batches=2,
        checkpoint_every=3,
        log_every=1,
        device="cpu",
        seed=0,
    )
    base.update(kw)
    return TrainConfig(**base)


def _seeded_model(kind: str, tokenizer):
    torch.manual_seed(123)  # identical initialisation for runs that are compared
    return tiny_model(kind, tokenizer)


def _params(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def test_fit_logs_validates_checkpoints_and_completes(data, tmp_path: Path) -> None:
    tokenizer, ds = data
    trainer = Trainer(
        tiny_model("hybrid", tokenizer), ds, _cfg(grad_accumulation=2), tmp_path, validation_data=ds
    )
    result = trainer.fit()
    assert result.status == "completed" and result.step == 6
    records = MetricLogger(tmp_path / "metrics.jsonl").read()
    assert {r["split"] for r in records} == {"train", "validation"}
    assert all("grad_norm" in r for r in records if r["split"] == "train")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["environment"]["torch"]
    names = {p.name for p in (tmp_path / "checkpoints").iterdir()}
    assert {"best.pt", "final.pt", "step_000000006.pt"} <= names


def test_resume_reproduces_uninterrupted_training(data, tmp_path: Path) -> None:
    tokenizer, ds = data
    # A constant schedule makes the 3-step "interrupted" run follow exactly the
    # first half of the 6-step run, isolating checkpoint and RNG restoration.
    cfg = _cfg(scheduler="constant", warmup_steps=0)
    full = Trainer(_seeded_model("ddpm", tokenizer), ds, cfg, tmp_path / "full")
    full.fit()
    part = tmp_path / "part"
    short = TrainConfig.from_dict({**cfg.to_dict(), "max_steps": 3})
    Trainer(_seeded_model("ddpm", tokenizer), ds, short, part).fit()
    resumed = Trainer(_seeded_model("ddpm", tokenizer), ds, cfg, part)
    resumed.fit(resume=True)
    assert resumed.step == 6
    for a, b in zip(_params(full.model), _params(resumed.model), strict=True):
        torch.testing.assert_close(a, b)


def test_validation_is_deterministic(data, tmp_path: Path) -> None:
    tokenizer, ds = data
    trainer = Trainer(tiny_model("hybrid", tokenizer), ds, _cfg(), tmp_path, validation_data=ds)
    assert trainer.validate() == trainer.validate()


def test_early_stopping(data, tmp_path: Path) -> None:
    tokenizer, ds = data
    cfg = _cfg(
        max_steps=50,
        eval_every=1,
        early_stopping_patience=1,
        early_stopping_min_delta=1e9,
        checkpoint_every=100,
    )
    trainer = Trainer(tiny_model("transformer", tokenizer), ds, cfg, tmp_path, validation_data=ds)
    result = trainer.fit()
    assert result.status == "early_stopped" and result.step == 2


class _ExplodingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def loss(self, batch) -> LossOutput:
        total = self.w * float("nan")
        return LossOutput(total.sum(), {"total": float("nan")}, 1)


def test_nonfinite_losses_abort_and_mark_run_failed(data, tmp_path: Path) -> None:
    _, ds = data
    trainer = Trainer(_ExplodingModel(), ds, _cfg(max_nonfinite_steps=2), tmp_path)
    with pytest.raises(NumericalError):
        trainer.fit()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "NumericalError" in manifest["error"]
    assert (tmp_path / "checkpoints" / "failed.pt").exists()


def test_resume_rejects_mismatched_checkpoint(data, tmp_path: Path) -> None:
    tokenizer, ds = data
    Trainer(tiny_model("hybrid", tokenizer), ds, _cfg(max_steps=3), tmp_path).fit()
    other = Trainer(tiny_model("transformer", tokenizer), ds, _cfg(), tmp_path)
    with pytest.raises(CheckpointError, match="does not match"):
        other.resume()


def test_lr_schedule_shapes() -> None:
    cfg = _cfg(max_steps=100, warmup_steps=10, scheduler="cosine", min_lr_ratio=0.1)
    f = lr_lambda(cfg)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.1)
    linear = lr_lambda(_cfg(max_steps=100, warmup_steps=0, scheduler="linear", min_lr_ratio=0.0))
    assert linear(50) == pytest.approx(0.5)


def test_checkpoint_io_is_atomic_and_validated(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "c.pt", {"step": 1})
    assert load_checkpoint(path)["step"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "bad.pt")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "missing.pt")
    for step in range(5):
        save_checkpoint(tmp_path / f"step_{step:09d}.pt", {"step": step})
    removed = prune_checkpoints(tmp_path, keep=2)
    assert len(removed) == 3 and len(list(tmp_path.glob("step_*.pt"))) == 2
    atomic_write_json(tmp_path / "m.json", {"a": 1})
    assert json.loads((tmp_path / "m.json").read_text()) == {"a": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"warmup_steps": 6},
        {"precision": "fp8"},
        {"device": "tpu"},
        {"grad_clip": 0.0},
        {"betas": (0.9, 1.0)},
    ],
)
def test_train_config_validation(kwargs) -> None:
    with pytest.raises(ConfigError):
        _cfg(**kwargs)


def test_train_config_round_trip() -> None:
    cfg = _cfg(betas=(0.8, 0.9))
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
