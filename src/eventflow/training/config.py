"""Training configuration and its validation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from eventflow.errors import ConfigError
from eventflow.models.config import build_dataclass

Precision = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and run-control settings.

    Budgets are expressed in optimiser steps; the effective batch per step is
    ``batch_size * grad_accumulation`` windows. ``eval_every`` and
    ``checkpoint_every`` count optimiser steps.
    """

    max_steps: int = 10_000
    batch_size: int = 32
    grad_accumulation: int = 1
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    momentum: float = 0.9
    scheduler: Literal["constant", "cosine", "linear"] = "cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    grad_clip: float | None = 1.0
    precision: Precision = "fp32"
    eval_every: int = 500
    eval_batches: int | None = 50
    checkpoint_every: int = 1000
    keep_checkpoints: int = 3
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    max_nonfinite_steps: int = 10
    log_every: int = 50
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    deterministic: bool = False

    def __post_init__(self) -> None:
        positive = (
            "max_steps",
            "batch_size",
            "grad_accumulation",
            "eval_every",
            "checkpoint_every",
            "keep_checkpoints",
            "log_every",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ConfigError(f"train.{name} must be >= 1")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ConfigError("train.lr must be positive and weight_decay non-negative")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ConfigError("train.betas must be two values in [0, 1)")
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise ConfigError(f"train.optimizer {self.optimizer!r} not supported")
        if self.scheduler not in ("constant", "cosine", "linear"):
            raise ConfigError(f"train.scheduler {self.scheduler!r} not supported")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ConfigError("train.warmup_steps must lie in [0, max_steps)")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ConfigError("train.min_lr_ratio must lie in [0, 1]")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError("train.grad_clip must be positive or null")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ConfigError("train.precision must be fp32, bf16 or fp16")
        if self.eval_batches is not None and self.eval_batches < 1:
            raise ConfigError("train.eval_batches must be >= 1 or null")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ConfigError("train.early_stopping_patience must be >= 1 or null")
        if self.max_nonfinite_steps < 0 or self.num_workers < 0:
            raise ConfigError("train.max_nonfinite_steps and num_workers must be >= 0")
        if self.device not in ("auto", "cpu", "mps") and not self.device.startswith("cuda"):
            raise ConfigError(f"train.device {self.device!r} must be auto, cpu, mps or cuda[:N]")

    @property
    def windows_per_step(self) -> int:
        return self.batch_size * self.grad_accumulation

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        return build_dataclass(cls, data, "train")
