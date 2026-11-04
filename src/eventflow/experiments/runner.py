"""Build every pipeline component from an :class:`ExperimentConfig` and run it.

Stages (each reusable on its own):

1. :func:`load_frame`          data source -> validated :class:`EventFrame`
2. :func:`prepare`             preprocess, chronological split, tokenizer fit
                               on the training partition only, windowed datasets
3. :func:`build_model`         neural model or statistical baseline
4. :func:`dry_run`             validate and initialise everything, run one CPU
                               forward/backward on a single tiny batch, report
5. :func:`run_experiment`      train -> generate -> evaluate -> write reports

``dry_run`` never trains: it is the compatibility check to run before
handing a configuration to a GPU machine.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eventflow.errors import ConfigError
from eventflow.evaluation.downstream import train_synthetic_test_real
from eventflow.evaluation.metrics import METRICS, EvalOptions
from eventflow.evaluation.report import EvaluationReport, evaluate
from eventflow.experiments.config import BaselineConfig, ExperimentConfig
from eventflow.generation.rollout import one_step_samples, rollout, sample_prompts
from eventflow.marketdata.io import read_csv, read_parquet
from eventflow.marketdata.lobster import LobsterDay, read_lobster
from eventflow.marketdata.preprocess import Partitions, chronological_split, fingerprint, preprocess
from eventflow.marketdata.schema import EventFrame
from eventflow.marketdata.synthetic import generate_synthetic
from eventflow.models.baseline import MarkovBaseline
from eventflow.models.config import ModelConfig
from eventflow.models.generative import EventGenerativeModel
from eventflow.tokenization.tokenizer import EncodedEvents, EventTokenizer
from eventflow.tokenization.windows import EventWindowDataset, collate_events
from eventflow.training.checkpoint import atomic_write_json
from eventflow.training.reproducibility import git_revision, seed_everything
from eventflow.training.trainer import Trainer


def load_frame(cfg: ExperimentConfig) -> EventFrame:
    data = cfg.data
    if data.source == "synthetic":
        return generate_synthetic(data.synthetic)
    if data.source == "csv":
        assert data.path is not None
        return read_csv(data.path, event_types=data.event_types, message_level=data.message_level)
    if data.source == "parquet":
        assert data.path is not None
        return read_parquet(
            data.path, event_types=data.event_types, message_level=data.message_level
        )
    days = [
        LobsterDay(d.instrument, Path(d.message), Path(d.orderbook) if d.orderbook else None)
        for d in data.lobster_days
    ]
    return read_lobster(days, tick_size=data.tick_size)


@dataclass(frozen=True)
class Prepared:
    partitions: Partitions
    tokenizer: EventTokenizer
    encoded: dict[str, EncodedEvents]
    datasets: dict[str, EventWindowDataset]
    fingerprints: dict[str, str]
    preprocess_log: tuple[dict[str, Any], ...]


def prepare(cfg: ExperimentConfig, frame: EventFrame | None = None) -> Prepared:
    frame = frame if frame is not None else load_frame(cfg)
    cleaned = preprocess(frame, cfg.preprocess)
    partitions = chronological_split(cleaned.frame, cfg.split)
    tokenizer = EventTokenizer.fit(partitions.train, cfg.tokenizer)
    encoded = {name: tokenizer.encode(part) for name, part in partitions.items()}
    datasets = {
        name: EventWindowDataset(enc, cfg.sequence.context_length, cfg.sequence.stride)
        for name, enc in encoded.items()
    }
    fingerprints = {
        "source": fingerprint(frame),
        **{name: fingerprint(part) for name, part in partitions.items()},
    }
    return Prepared(partitions, tokenizer, encoded, datasets, fingerprints, cleaned.log)


def build_model(cfg: ExperimentConfig, prepared: Prepared) -> EventGenerativeModel | MarkovBaseline:
    """Neural model (seeded initialisation) or a fitted statistical baseline."""
    if isinstance(cfg.model, BaselineConfig):
        return MarkovBaseline.fit(
            prepared.encoded["train"],
            prepared.tokenizer.layout,
            len(prepared.tokenizer.instruments),
            cfg.model.smoothing,
        )
    torch.manual_seed(cfg.seed)
    assert isinstance(cfg.model, ModelConfig)
    return EventGenerativeModel.from_tokenizer(cfg.model, prepared.tokenizer)


def _required_capabilities(cfg: ExperimentConfig) -> dict[str, list[str]]:
    names = cfg.evaluation.metrics if cfg.evaluation.metrics is not None else tuple(METRICS)
    return {n: sorted(c.value for c in METRICS[n].requires) for n in names}


def dry_run(cfg: ExperimentConfig, frame: EventFrame | None = None) -> dict[str, Any]:
    """Validate and initialise every component; one CPU forward/backward; no training.

    Raises on any incompatibility. Returns a JSON-serialisable report with
    dataset summaries and fingerprints, window counts, parameter counts,
    budgets and which metrics will be skipped for lack of capabilities.
    """
    started = time.time()
    prepared = prepare(cfg, frame)
    if len(prepared.datasets["train"]) == 0:
        raise ConfigError("the training partition yields no windows")
    model = build_model(cfg, prepared)
    report: dict[str, Any] = {
        "experiment": cfg.name,
        "model_kind": cfg.model_kind,
        "fingerprints": prepared.fingerprints,
        "partitions": {name: part.summary() for name, part in prepared.partitions.items()},
        "windows": {name: len(ds) for name, ds in prepared.datasets.items()},
        "targets_per_epoch": prepared.datasets["train"].num_targets(),
        "layout": prepared.tokenizer.layout.to_dict(),
        "training_token_budget": cfg.training_token_budget(),
        "sampling_evaluations_per_event": cfg.sampling_evaluations_per_event(),
        "preprocess_log": list(prepared.preprocess_log),
    }
    train_caps = prepared.partitions.train.capabilities
    report["skipped_metrics"] = {
        name: caps
        for name, caps in _required_capabilities(cfg).items()
        if not all(any(c == cap.value for cap in train_caps) for c in caps)
    }
    if isinstance(model, EventGenerativeModel):
        batch = collate_events(
            [prepared.datasets["train"][i] for i in range(min(2, len(prepared.datasets["train"])))]
        )
        model.train()
        out = model.loss(batch)
        if not torch.isfinite(out.total):
            raise ConfigError(f"initial loss is not finite: {out.components}")
        out.total.backward()
        disconnected = [n for n, p in model.named_parameters() if p.grad is None]
        if disconnected:
            raise ConfigError(f"parameters receive no gradient: {disconnected[:5]}")
        model.zero_grad(set_to_none=True)
        model.eval()
        nxt = model.sample_next(batch.inputs(), torch.Generator().manual_seed(cfg.seed))
        report.update(
            parameters=model.num_parameters(),
            initial_loss=out.components,
            sample_shapes={
                "categorical": list(nxt.categorical.shape),
                "continuous": list(nxt.continuous.shape),
            },
        )
    else:
        report.update(parameters=model.num_parameters(), baseline=model.summary())
    report["seconds"] = round(time.time() - started, 3)
    return report


def generate_and_evaluate(
    cfg: ExperimentConfig,
    prepared: Prepared,
    model: EventGenerativeModel | MarkovBaseline,
    device: torch.device | str = "cpu",
) -> tuple[EvaluationReport, Any]:
    """Rollouts and one-step samples from evaluation prompts, then all metrics."""
    eval_name = "test" if "test" in prepared.encoded else "validation"
    if eval_name not in prepared.encoded:
        raise ConfigError("evaluation needs a validation or test partition")
    encoded = prepared.encoded[eval_name]
    real = prepared.partitions.test if eval_name == "test" else prepared.partitions.validation
    assert real is not None
    if isinstance(model, EventGenerativeModel):
        model = model.to(device)
    s = cfg.sampling
    prompts = sample_prompts(
        encoded, cfg.sequence.context_length, s.num_rollouts, seed=s.prompt_seed
    )
    prompts = type(prompts)(prompts.context.to(device), None, prompts.start_timestamp)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    result = rollout(
        model,
        prepared.tokenizer,
        prompts,
        s.horizon,
        max_context=cfg.sequence.context_length,
        generator=generator,
        temperature=s.temperature,
    )
    options = EvalOptions(
        lags=cfg.evaluation.lags, bucket_size=cfg.evaluation.bucket_size, invalid=result.invalid
    )
    downstream = None
    if cfg.evaluation.downstream:
        downstream = train_synthetic_test_real(prepared.partitions.train, real, result.frame)
    report = evaluate(
        real,
        result.frame,
        metrics=cfg.evaluation.metrics,
        options=options,
        by_instrument=cfg.evaluation.by_instrument,
        downstream=downstream,
        metadata={
            "experiment": cfg.name,
            "evaluation_partition": eval_name,
            "mode": "rollout",
            "horizon": s.horizon,
        },
    )
    fake, real_next = one_step_samples(
        model,
        prepared.tokenizer,
        encoded,
        cfg.sequence.context_length,
        s.one_step_samples,
        seed=s.prompt_seed,
        temperature=s.temperature,
    )
    one_step = {
        "invalid_rate": float(fake.invalid.mean()),
        "event_type_tv": float(
            0.5
            * np.abs(
                np.bincount(fake.event_type, minlength=len(prepared.tokenizer.event_types))
                / len(fake)
                - np.bincount(real_next.event_type, minlength=len(prepared.tokenizer.event_types))
                / len(real_next)
            ).sum()
        ),
    }
    return report, {"rollout_invalid_rate": result.invalid_rate, "one_step": one_step}


def run_experiment(
    cfg: ExperimentConfig,
    run_dir: Path | None = None,
    *,
    resume: bool = False,
    frame: EventFrame | None = None,
) -> dict[str, Any]:
    """Full pipeline. Trains for ``train.max_steps``: use on suitable hardware."""
    run_dir = Path(run_dir or Path(cfg.output_dir) / cfg.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    prepared = prepare(cfg, frame)
    (run_dir / "tokenizer.json").write_text(prepared.tokenizer.to_json())
    model = build_model(cfg, prepared)
    manifest = {
        "experiment": cfg.to_dict(),
        "fingerprints": prepared.fingerprints,
        "git": git_revision(),
        "model_kind": cfg.model_kind,
        "parameters": model.num_parameters(),
    }
    device: torch.device | str = "cpu"
    if isinstance(model, EventGenerativeModel):
        trainer = Trainer(
            model,
            prepared.datasets["train"],
            cfg.train,
            run_dir,
            validation_data=prepared.datasets.get("validation"),
            manifest=manifest,
            checkpoint_extra={
                "experiment": cfg.to_dict(),
                "tokenizer": prepared.tokenizer.to_dict(),
            },
        )
        result = trainer.fit(resume=resume)
        device = trainer.device
        training = {
            "status": result.status,
            "step": result.step,
            "best_validation": result.best_validation,
        }
    else:
        atomic_write_json(run_dir / "manifest.json", {**manifest, "status": "fitted"})
        training = {"status": "fitted"}
    report, extras = generate_and_evaluate(cfg, prepared, model, device)
    (run_dir / "evaluation.json").write_text(report.to_json())
    summary = {
        "experiment": cfg.name,
        "training": training,
        **extras,
        "fingerprints": prepared.fingerprints,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary


def load_model_from_checkpoint(
    path: Path, map_location: str = "cpu"
) -> tuple[EventGenerativeModel, EventTokenizer, dict[str, Any]]:
    """Rebuild a trained model and its tokenizer from a self-describing checkpoint."""
    from eventflow.experiments.config import config_from_dict
    from eventflow.training.checkpoint import load_checkpoint

    payload = load_checkpoint(path, map_location=map_location)
    if "experiment" not in payload or "tokenizer" not in payload:
        raise ConfigError(f"{path} lacks the experiment config or tokenizer")
    tokenizer = EventTokenizer.from_dict(payload["tokenizer"])
    experiment = payload["experiment"]
    cfg_dict = {
        "experiment": {k: experiment[k] for k in ("name", "seed", "output_dir")},
        **{k: v for k, v in experiment.items() if k not in ("name", "seed", "output_dir")},
    }
    cfg = config_from_dict(json.loads(json.dumps(cfg_dict)))
    assert isinstance(cfg.model, ModelConfig)
    model = EventGenerativeModel.from_tokenizer(cfg.model, tokenizer)
    model.load_state_dict(payload["model"])
    return model, tokenizer, payload
