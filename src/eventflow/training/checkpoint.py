"""Atomic checkpoint writing, loading and retention.

A checkpoint is a single ``torch.save`` file holding the model, optimiser,
scheduler and gradient-scaler states, all RNG states, the training step and
early-stopping bookkeeping, plus the JSON-serialisable experiment config and
tokenizer so a checkpoint is self-describing.

Atomicity: the payload is written to a temporary file in the destination
directory, flushed and ``fsync``-ed, then moved into place with
``os.replace`` (atomic on POSIX and Windows for same-filesystem moves). A
crash mid-write leaves the previous checkpoint intact; a stray ``*.tmp`` is
the only possible residue.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eventflow.errors import CheckpointError

CHECKPOINT_FORMAT = 1


def atomic_write_bytes(path: Path, write: Any) -> None:
    """Call ``write(file_handle)`` on a temp file, then atomically replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True, default=str).encode()
    atomic_write_bytes(path, lambda handle: handle.write(payload))


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - legacy global state is what we restore
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    # Checkpoints may be loaded with map_location on an accelerator; RNG states
    # must be CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    payload = {"format": CHECKPOINT_FORMAT, **payload}
    atomic_write_bytes(Path(path), lambda handle: torch.save(payload, handle))
    return Path(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint {path} does not exist")
    try:
        # weights_only=False: the payload holds RNG states (tuples, numpy arrays).
        # Only load checkpoints you produced yourself.
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # torch raises several unrelated types for corrupt files
        raise CheckpointError(f"checkpoint {path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"checkpoint {path} has unsupported format")
    return payload


def step_checkpoint_name(step: int) -> str:
    return f"step_{step:09d}.pt"


def prune_checkpoints(directory: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` step checkpoints; returns removed paths."""
    steps = sorted(Path(directory).glob("step_*.pt"))
    removed = steps[:-keep] if keep > 0 else steps
    for path in removed:
        path.unlink()
    return removed


def latest_checkpoint(directory: Path) -> Path | None:
    steps = sorted(Path(directory).glob("step_*.pt"))
    return steps[-1] if steps else None
