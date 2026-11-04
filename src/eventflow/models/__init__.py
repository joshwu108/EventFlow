"""Neural and statistical next-event models.

Heavy submodules are imported lazily so that configuration code (and the
diffusion / flow heads, which import :mod:`eventflow.models.config`) never
trigger a circular import.
"""

from __future__ import annotations

from typing import Any

from eventflow.models.config import PRESETS, DiffusionConfig, FlowConfig, ModelConfig

_LAZY = {
    "EventGenerativeModel": "eventflow.models.generative",
    "LossOutput": "eventflow.models.generative",
    "NextEvent": "eventflow.models.generative",
    "MarkovBaseline": "eventflow.models.baseline",
}

__all__ = ["PRESETS", "DiffusionConfig", "FlowConfig", "ModelConfig", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
