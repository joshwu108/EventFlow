"""Structured local metric logging (JSON Lines) and run manifests.

No external service is involved: metrics go to ``metrics.jsonl`` in the run
directory, one JSON object per line, flushed after every write so a crashed
run keeps everything logged before the crash.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from eventflow.training.checkpoint import atomic_write_json


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # JSON has no NaN/inf; keep them visible as strings
    return value


class MetricLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, split: str, step: int, metrics: dict[str, Any]) -> None:
        record = {
            "time": time.time(),
            "split": split,
            "step": step,
            **{k: _clean(v) for k, v in metrics.items()},
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


class RunManifest:
    """Mutable-on-disk, immutable-in-memory run description (``manifest.json``).

    Every update writes a fresh copy atomically, so readers never observe a
    partially written manifest.
    """

    def __init__(self, path: Path, initial: dict[str, Any]):
        self.path = Path(path)
        self._data = dict(initial)
        atomic_write_json(self.path, self._data)

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    def update(self, **changes: Any) -> dict[str, Any]:
        self._data = {**self._data, **changes, "updated": time.time()}
        atomic_write_json(self.path, self._data)
        return dict(self._data)

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)
