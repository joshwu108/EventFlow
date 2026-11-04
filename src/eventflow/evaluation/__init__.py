"""Generative evaluation: marginal, conditional, temporal, structural and downstream metrics."""

from eventflow.evaluation.downstream import DirectionTask, train_synthetic_test_real
from eventflow.evaluation.metrics import METRICS, REGISTRY, EvalOptions, Metric, MetricResult
from eventflow.evaluation.report import CATEGORIES, EvaluationReport, evaluate, run_metric

__all__ = [
    "CATEGORIES",
    "METRICS",
    "REGISTRY",
    "DirectionTask",
    "EvalOptions",
    "EvaluationReport",
    "Metric",
    "MetricResult",
    "evaluate",
    "run_metric",
    "train_synthetic_test_real",
]
