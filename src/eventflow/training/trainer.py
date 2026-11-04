"""Training loop with validation, checkpoint/resume, early stopping and failure recovery.

The trainer works with any ``nn.Module`` exposing ``loss(batch) -> LossOutput``
(:class:`~eventflow.models.generative.EventGenerativeModel`). It never
requires a GPU: on CPU it runs the same code path in fp32 (or bf16 autocast).

Run directory layout::

    run_dir/
        manifest.json          run description and status (atomic updates)
        metrics.jsonl          train / validation metrics, one JSON per line
        checkpoints/
            step_000001000.pt  periodic checkpoints (newest ``keep_checkpoints`` kept)
            best.pt            best validation loss so far
            final.pt           written when training ends normally

Failure recovery: any exception inside :meth:`Trainer.fit` marks the run
``failed`` in the manifest, writes ``checkpoints/failed.pt`` when the model
state is still finite, and re-raises. :meth:`Trainer.fit` with
``resume=True`` continues from the newest step checkpoint (model, optimiser,
scheduler, scaler, RNG states and early-stopping state are restored).
Non-finite losses skip the optimiser step; more than ``max_nonfinite_steps``
consecutive skips abort with :class:`~eventflow.errors.NumericalError`.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from eventflow.errors import CheckpointError, NumericalError
from eventflow.tokenization.windows import EventBatch, collate_events
from eventflow.training.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    step_checkpoint_name,
)
from eventflow.training.config import TrainConfig
from eventflow.training.reproducibility import environment_info, resolve_device, seed_everything
from eventflow.training.runlog import MetricLogger, RunManifest


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW/Adam/SGD with weight decay on ``nn.Linear`` weights only."""
    # Classify by owning module type, not by name: only nn.Linear weights decay.
    decay_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)}
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if id(param) in decay_ids else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, betas=cfg.betas)
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)


def lr_lambda(cfg: TrainConfig) -> Callable[[int], float]:
    """Multiplier on the base learning rate: linear warmup, then constant/cosine/linear decay."""

    def factor(step: int) -> float:
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.scheduler == "constant":
            return 1.0
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        floor = cfg.min_lr_ratio
        if cfg.scheduler == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return floor + (1 - floor) * (1 - progress)

    return factor


@dataclass(frozen=True)
class TrainResult:
    status: str
    step: int
    best_validation: float | None
    run_dir: Path


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset[Any],
        config: TrainConfig,
        run_dir: Path,
        *,
        validation_data: Dataset[Any] | None = None,
        manifest: dict[str, Any] | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.device = resolve_device(config.device)
        seed_everything(config.seed, deterministic=config.deterministic)
        self.model = model.to(self.device)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda(config))
        self.autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            config.precision
        ]
        # Loss scaling is only meaningful (and only supported) for fp16 on CUDA.
        use_scaler = config.precision == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.step = 0
        self.batches_consumed = 0
        self.best_validation: float | None = None
        self.evals_without_improvement = 0
        self.nonfinite_streak = 0
        self.checkpoint_extra = checkpoint_extra or {}
        self.logger = MetricLogger(self.run_dir / "metrics.jsonl")
        self.manifest = RunManifest(
            self.run_dir / "manifest.json",
            {
                **(manifest or {}),
                "status": "initialised",
                "created": time.time(),
                "environment": environment_info(self.device),
                "train_config": config.to_dict(),
            },
        )

    # ------------------------------------------------------------------ data
    def _loader(self, data: Dataset[Any], *, shuffle: bool, seed: int) -> DataLoader[Any]:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_events,
            num_workers=self.config.num_workers,
            generator=generator,
            drop_last=False,
        )

    def _batches(self) -> Iterator[EventBatch]:
        epoch = 0
        while True:
            # Seed per epoch so a resumed run sees a well-defined data order.
            yield from self._loader(self.train_data, shuffle=True, seed=self.config.seed + epoch)
            epoch += 1

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Target-weighted mean of every loss component over validation batches."""
        if self.validation_data is None:
            return {}
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        loader = self._loader(self.validation_data, shuffle=False, seed=0)
        devices = [self.device] if self.device.type == "cuda" else []
        # Stochastic losses (diffusion/flow) draw noise from the global RNG; a
        # forked, fixed seed makes validation comparable across evaluations
        # without perturbing the training RNG stream.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed + 1_000_003)
            for i, batch in enumerate(loader):
                if self.config.eval_batches is not None and i >= self.config.eval_batches:
                    break
                with self._autocast():
                    out = self.model.loss(batch.to(self.device))
                for key, value in out.components.items():
                    totals[key] = totals.get(key, 0.0) + value * out.num_targets
                count += out.num_targets
        self.model.train()
        return {k: v / max(count, 1) for k, v in totals.items()} | {"targets": float(count)}

    # ----------------------------------------------------------- checkpoints
    def _payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "batches_consumed": self.batches_consumed,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": rng_state(),
            "best_validation": self.best_validation,
            "evals_without_improvement": self.evals_without_improvement,
            "train_config": self.config.to_dict(),
            **self.checkpoint_extra,
        }

    def save(self, name: str | None = None) -> Path:
        path = self.checkpoint_dir / (name or step_checkpoint_name(self.step))
        save_checkpoint(path, self._payload())
        if name is None:
            prune_checkpoints(self.checkpoint_dir, self.config.keep_checkpoints)
        return path

    def resume(self, path: Path | None = None) -> bool:
        """Restore the newest (or given) checkpoint; returns False if none exists."""
        path = path or latest_checkpoint(self.checkpoint_dir)
        if path is None:
            return False
        payload = load_checkpoint(path, map_location=self.device)
        try:
            self.model.load_state_dict(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scheduler.load_state_dict(payload["scheduler"])
            self.scaler.load_state_dict(payload["scaler"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise CheckpointError(
                f"checkpoint {path} does not match this model/optimiser: {exc}"
            ) from exc
        restore_rng_state(payload["rng"])
        self.step = int(payload["step"])
        self.batches_consumed = int(payload.get("batches_consumed", self.step))
        self.best_validation = payload["best_validation"]
        self.evals_without_improvement = int(payload["evals_without_improvement"])
        self.manifest.update(resumed_from=str(path), resumed_step=self.step)
        return True

    def restore_best(self) -> bool:
        """Load ``best.pt`` model weights (if present) for evaluation; returns success."""
        path = self.checkpoint_dir / "best.pt"
        if not path.is_file():
            return False
        self.model.load_state_dict(load_checkpoint(path, map_location=self.device)["model"])
        return True

    # -------------------------------------------------------------- training
    def _train_step(self, batches: Iterator[EventBatch]) -> dict[str, float] | None:
        cfg = self.config
        self.optimizer.zero_grad(set_to_none=True)
        weighted: dict[str, float] = {}
        targets = 0
        reference: int | None = None
        finite = True
        for _ in range(cfg.grad_accumulation):
            batch = next(batches).to(self.device)
            self.batches_consumed += 1
            with self._autocast():
                out = self.model.loss(batch)
            if not torch.isfinite(out.total):
                finite = False
                break
            # Micro-batch losses are per-target means. Weighting each by its
            # target count (relative to the first micro-batch, to keep loss
            # magnitudes near 1 under fp16 loss scaling) and rescaling the
            # summed gradients below makes accumulation equal to one large
            # batch even when windows are padded or short.
            reference = reference or max(out.num_targets, 1)
            self.scaler.scale(out.total * (out.num_targets / reference)).backward()
            for key, value in out.components.items():
                weighted[key] = weighted.get(key, 0.0) + value * out.num_targets
            targets += out.num_targets
        if not finite:
            self.optimizer.zero_grad(set_to_none=True)
            self.nonfinite_streak += 1
            if self.nonfinite_streak > cfg.max_nonfinite_steps:
                raise NumericalError(
                    f"{self.nonfinite_streak} consecutive non-finite losses at step {self.step}"
                )
            return None
        self.nonfinite_streak = 0
        if targets and reference is not None and targets != reference:
            factor = reference / targets
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(factor)
        components = {k: v / max(targets, 1) for k, v in weighted.items()}
        self.scaler.unscale_(self.optimizer)
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if cfg.grad_clip is not None
            else torch.linalg.vector_norm(
                torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])
            )
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        return components | {
            "grad_norm": float(grad_norm),
            "lr": self.scheduler.get_last_lr()[0],
            "targets": float(targets),
        }

    def _after_validation(self, metrics: dict[str, float]) -> bool:
        """Track the best model; return True when early stopping triggers."""
        value = metrics.get("total")
        if value is None:
            return False
        if (
            self.best_validation is None
            or value < self.best_validation - self.config.early_stopping_min_delta
        ):
            self.best_validation = value
            self.evals_without_improvement = 0
            self.save("best.pt")
            return False
        self.evals_without_improvement += 1
        patience = self.config.early_stopping_patience
        return patience is not None and self.evals_without_improvement >= patience

    def fit(self, *, resume: bool = False) -> TrainResult:
        cfg = self.config
        if resume:
            self.resume()
        self.manifest.update(status="running")
        self.model.train()
        batches = self._batches()
        # Fast-forward the data stream so a resumed run does not replay batches
        # (counts batches, not steps: skipped non-finite steps also consume data).
        for _ in range(self.batches_consumed):
            next(batches)
        status = "completed"
        try:
            while self.step < cfg.max_steps:
                metrics = self._train_step(batches)
                if metrics is None:
                    self.logger.log("train", self.step, {"skipped_nonfinite": 1})
                    continue
                if self.step % cfg.log_every == 0:
                    self.logger.log("train", self.step, metrics)
                stop = False
                if self.validation_data is not None and self.step % cfg.eval_every == 0:
                    val = self.validate()
                    self.logger.log("validation", self.step, val)
                    stop = self._after_validation(val)
                if self.step % cfg.checkpoint_every == 0:
                    self.save()
                if stop:
                    status = "early_stopped"
                    break
        except BaseException as exc:
            self.manifest.update(
                status="failed", failed_step=self.step, error=f"{type(exc).__name__}: {exc}"
            )
            if all(torch.isfinite(p).all() for p in self.model.parameters()):
                with contextlib.suppress(Exception):
                    self.save("failed.pt")
            raise
        self.save("final.pt")
        self.manifest.update(
            status=status, final_step=self.step, best_validation=self.best_validation
        )
        return TrainResult(status, self.step, self.best_validation, self.run_dir)
