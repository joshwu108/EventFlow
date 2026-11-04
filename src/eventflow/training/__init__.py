"""Training infrastructure: configuration, trainer, checkpoints, logging, reproducibility."""

from eventflow.training.checkpoint import (
    atomic_write_json,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from eventflow.training.config import TrainConfig
from eventflow.training.reproducibility import (
    environment_info,
    git_revision,
    resolve_device,
    seed_everything,
)
from eventflow.training.runlog import MetricLogger, RunManifest
from eventflow.training.trainer import Trainer, TrainResult, build_optimizer, lr_lambda

__all__ = [
    "MetricLogger",
    "RunManifest",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "atomic_write_json",
    "build_optimizer",
    "environment_info",
    "git_revision",
    "latest_checkpoint",
    "load_checkpoint",
    "lr_lambda",
    "resolve_device",
    "save_checkpoint",
    "seed_everything",
]
