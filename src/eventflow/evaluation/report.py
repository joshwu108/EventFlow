"""Run metrics with capability gating and collect them into a report."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eventflow.errors import ConfigError
from eventflow.evaluation.metrics import (
    METRICS,
    EvalOptions,
    InsufficientData,
    Metric,
    MetricResult,
)
from eventflow.marketdata.schema import EventFrame

CATEGORIES: tuple[str, ...] = ("marginal", "conditional", "temporal", "structural", "downstream")


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[MetricResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_category(self) -> dict[str, list[MetricResult]]:
        return {c: [r for r in self.results if r.category == c] for c in CATEGORIES}

    def get(self, name: str, scope: str = "all") -> MetricResult:
        for result in self.results:
            if result.name == name and result.scope == scope:
                return result
        raise KeyError(f"{name} ({scope})")

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "results": [r.to_dict() for r in self.results]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def run_metric(
    metric: Metric,
    real: EventFrame,
    generated: EventFrame,
    options: EvalOptions,
    scope: str = "all",
) -> MetricResult:
    missing = sorted(c.value for c in metric.requires if not (real.has(c) and generated.has(c)))
    if missing:
        return MetricResult(
            metric.name,
            metric.category,
            "skipped",
            reason=f"requires capabilities {missing} in both frames",
            scope=scope,
        )
    try:
        values = metric.fn(real, generated, options)
    except InsufficientData as exc:
        return MetricResult(
            metric.name, metric.category, "insufficient_data", reason=str(exc), scope=scope
        )
    return MetricResult(metric.name, metric.category, "ok", values, scope=scope)


def evaluate(
    real: EventFrame,
    generated: EventFrame,
    *,
    metrics: Sequence[str] | None = None,
    options: EvalOptions | None = None,
    by_instrument: bool = False,
    downstream: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Compare ``generated`` against ``real`` on the selected metrics.

    ``by_instrument`` repeats every metric on each instrument present in both
    frames (scope ``instrument:<name>``). ``downstream`` lets callers attach a
    precomputed :func:`~eventflow.evaluation.downstream.train_synthetic_test_real`
    result, which needs separate real train/test frames.
    """
    if real.event_types != generated.event_types:
        raise ConfigError("real and generated frames use different event-type vocabularies")
    options = options or EvalOptions()
    names = list(metrics) if metrics is not None else list(METRICS)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ConfigError(f"unknown metrics {unknown}; available: {sorted(METRICS)}")
    selected = [METRICS[n] for n in names]
    results = [run_metric(m, real, generated, options) for m in selected]
    if by_instrument:
        for code, name in enumerate(real.instruments):
            if name not in generated.instruments:
                continue
            gen_code = generated.instruments.index(name)
            r_sub = real.take(real.instrument == code)
            g_sub = generated.take(generated.instrument == gen_code)
            if len(r_sub) == 0 or len(g_sub) == 0:
                continue
            # The rollout invalid matrix is not instrument-indexed; drop it per instrument.
            sub_options = EvalOptions(
                options.lags, options.bucket_size, options.regimes, options.regime_window, None
            )
            results += [
                run_metric(m, r_sub, g_sub, sub_options, f"instrument:{name}") for m in selected
            ]
    if downstream is not None:
        status = downstream.get("status", "ok")
        results.append(MetricResult("tstr_direction", "downstream", status, downstream))
    meta = {"real": real.summary(), "generated": generated.summary(), **(metadata or {})}
    return EvaluationReport(tuple(results), meta)
