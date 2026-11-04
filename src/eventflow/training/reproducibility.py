"""Seeding, device selection and environment capture for run manifests."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

import eventflow
from eventflow.errors import ConfigError


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all devices).

    ``deterministic=True`` additionally requests deterministic kernels; some
    CUDA operations then raise instead of running non-deterministically,
    which is the point. Bitwise reproducibility across hardware or library
    versions is not promised.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global state used by third-party code
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA, then MPS, then CPU; explicit requests are verified."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"device {name!r} requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("device 'mps' requested but MPS is not available")
    return torch.device(name)


def git_revision(path: Path | None = None) -> dict[str, Any]:
    """Current commit and whether the working tree is dirty (``None`` outside Git)."""
    cwd = path or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": dirty}


def environment_info(device: torch.device | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "eventflow": eventflow.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device is not None:
        info["device"] = str(device)
        if device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(device)
    return info
