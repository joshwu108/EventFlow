"""Experiment configuration (TOML), validation and fair-comparison checks.

One TOML file fully specifies a run::

    [experiment]  name, seed, output_dir
    [data]        source and adapter options (synthetic / csv / parquet / lobster)
    [preprocess]  eventflow.marketdata.preprocess.PreprocessConfig
    [split]       eventflow.marketdata.preprocess.SplitConfig
    [tokenizer]   eventflow.tokenization.TokenizerConfig
    [sequence]    context_length, stride
    [model]       eventflow.models.ModelConfig, or kind = "baseline"
    [train]       eventflow.training.TrainConfig
    [sampling]    rollout and one-step sampling budgets
    [evaluation]  metric selection and options

Unknown keys anywhere are errors, so a typo cannot silently fall back to a
default.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from eventflow.errors import ConfigError
from eventflow.marketdata.preprocess import PreprocessConfig, SplitConfig
from eventflow.marketdata.synthetic import SyntheticConfig
from eventflow.models.config import ModelConfig, build_dataclass
from eventflow.tokenization.tokenizer import TokenizerConfig
from eventflow.training.config import TrainConfig


@dataclass(frozen=True)
class LobsterDaySpec:
    instrument: str
    message: str
    orderbook: str | None = None
    day_index: int | None = None


@dataclass(frozen=True)
class DataConfig:
    source: Literal["synthetic", "csv", "parquet", "lobster"] = "synthetic"
    path: str | None = None
    event_types: tuple[str, ...] | None = None
    message_level: bool = False
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    lobster_days: tuple[LobsterDaySpec, ...] = ()
    tick_size: float = 0.01

    def __post_init__(self) -> None:
        if self.source not in ("synthetic", "csv", "parquet", "lobster"):
            raise ConfigError(f"data.source {self.source!r} not supported")
        if self.source in ("csv", "parquet") and not self.path:
            raise ConfigError(f"data.path is required for source={self.source!r}")
        if self.source == "lobster" and not self.lobster_days:
            raise ConfigError("data.lobster_days is required for source='lobster'")


@dataclass(frozen=True)
class SequenceConfig:
    context_length: int = 128
    stride: int | None = None

    def __post_init__(self) -> None:
        if self.context_length < 1 or (self.stride is not None and self.stride < 1):
            raise ConfigError("sequence.context_length and stride must be >= 1")


@dataclass(frozen=True)
class BaselineConfig:
    kind: Literal["baseline"] = "baseline"
    smoothing: float = 0.5


@dataclass(frozen=True)
class SamplingConfig:
    num_rollouts: int = 64
    horizon: int = 512
    temperature: float = 1.0
    one_step_samples: int = 2048
    prompt_seed: int = 0

    def __post_init__(self) -> None:
        if min(self.num_rollouts, self.horizon, self.one_step_samples) < 1:
            raise ConfigError("sampling budgets must be >= 1")
        if self.temperature <= 0:
            raise ConfigError("sampling.temperature must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...] | None = None
    by_instrument: bool = True
    lags: tuple[int, ...] = (1, 2, 5, 10, 20)
    bucket_size: int = 20
    downstream: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 0
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    model: ModelConfig | BaselineConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in "/\\ "):
            raise ConfigError("experiment.name must be non-empty without slashes or spaces")
        if (
            isinstance(self.model, ModelConfig)
            and self.model.max_context < self.sequence.context_length
        ):
            raise ConfigError(
                f"model.max_context={self.model.max_context} is shorter than "
                f"sequence.context_length={self.sequence.context_length}"
            )
        if self.evaluation.metrics is not None:
            from eventflow.evaluation.metrics import METRICS

            unknown = sorted(set(self.evaluation.metrics) - set(METRICS))
            if unknown:
                raise ConfigError(f"evaluation.metrics: unknown {unknown}")

    @property
    def is_baseline(self) -> bool:
        return isinstance(self.model, BaselineConfig)

    @property
    def model_kind(self) -> str:
        return self.model.kind

    def training_token_budget(self) -> int:
        """Upper bound on supervised next-event targets seen during training."""
        if self.is_baseline:
            return 0
        return self.train.max_steps * self.train.windows_per_step * self.sequence.context_length

    def sampling_evaluations_per_event(self) -> int:
        """Continuous-head network evaluations per generated event (sampling cost)."""
        if not isinstance(self.model, ModelConfig):
            return 0
        if self.model.continuous_head == "ddpm":
            return self.model.diffusion.sample_steps
        if self.model.continuous_head == "flow":
            from eventflow.flow.solvers import function_evaluations

            return function_evaluations(self.model.flow.solver, self.model.flow.sample_steps)
        return 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    data = dict(data)
    experiment = data.pop("experiment", {})
    if not isinstance(experiment, dict) or "name" not in experiment:
        raise ConfigError("[experiment] table with a name is required")
    allowed = {
        "data",
        "preprocess",
        "split",
        "tokenizer",
        "sequence",
        "model",
        "train",
        "sampling",
        "evaluation",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"unknown top-level tables {unknown}")
    extra_experiment = sorted(set(experiment) - {"name", "seed", "output_dir"})
    if extra_experiment:
        raise ConfigError(f"experiment: unknown keys {extra_experiment}")
    kwargs: dict[str, Any] = dict(experiment)
    data_table = dict(data.get("data", {}))
    if "lobster_days" in data_table:
        data_table["lobster_days"] = tuple(
            build_dataclass(LobsterDaySpec, d, "data.lobster_days")
            for d in data_table["lobster_days"]
        )
    if "synthetic" in data_table:
        data_table["synthetic"] = build_dataclass(
            SyntheticConfig, data_table["synthetic"], "data.synthetic"
        )
    builders: dict[str, Any] = {
        "preprocess": PreprocessConfig,
        "split": SplitConfig,
        "tokenizer": TokenizerConfig,
        "sequence": SequenceConfig,
        "train": TrainConfig,
        "sampling": SamplingConfig,
        "evaluation": EvaluationConfig,
    }
    kwargs["data"] = build_dataclass(DataConfig, data_table, "data")
    for name, cls in builders.items():
        if name in data:
            kwargs[name] = build_dataclass(cls, data[name], name)
    if "model" in data:
        model = dict(data["model"])
        if model.get("kind") == "baseline":
            kwargs["model"] = build_dataclass(BaselineConfig, model, "model")
        else:
            kwargs["model"] = ModelConfig.from_dict(model)
    try:
        return ExperimentConfig(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"experiment: {exc}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        # A baseline shares nothing with the neural [model] table it would inherit.
        if key == "model" and isinstance(value, dict) and value.get("kind") == "baseline":
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_config_table(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Parse a TOML config, resolving ``experiment.extends`` (relative path) recursively.

    The child's tables are deep-merged over the parent's, so shared settings
    for a comparison live in one base file and cannot drift apart.
    """
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"circular extends: {' -> '.join(str(p) for p in (*_seen, path))}")
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    parent = data.get("experiment", {}).pop("extends", None)
    if parent is None:
        return data
    base = read_config_table(path.parent / parent, (*_seen, path))
    base.get("experiment", {}).pop("name", None)
    return _deep_merge(base, data)


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(read_config_table(path))


# ------------------------------------------------------------------ fairness
FAIRNESS_SECTIONS = (
    "data",
    "preprocess",
    "split",
    "tokenizer",
    "sequence",
    "sampling",
    "evaluation",
)
NEURAL_TRAIN_KEYS = (
    "max_steps",
    "batch_size",
    "grad_accumulation",
    "optimizer",
    "lr",
    "weight_decay",
    "betas",
    "scheduler",
    "warmup_steps",
    "min_lr_ratio",
    "grad_clip",
    "precision",
    "eval_every",
    "early_stopping_patience",
)


def comparability_issues(
    configs: list[ExperimentConfig],
    parameter_counts: dict[str, int] | None = None,
    parameter_tolerance: float = 0.25,
) -> list[str]:
    """Differences that would make a comparison between ``configs`` unfair.

    Checks that data, preprocessing, partitions, tokenisation, context length,
    sampling budgets and evaluation are identical; that all neural runs share
    optimiser, schedule and training-token budgets; and (when
    ``parameter_counts`` is given, keyed by experiment name) that neural
    parameter counts lie within ``parameter_tolerance`` of their median.
    Different seeds are allowed (they are the replication axis).
    """
    issues: list[str] = []
    if len(configs) < 2:
        return issues
    ref = configs[0]
    for cfg in configs[1:]:
        for section in FAIRNESS_SECTIONS:
            if getattr(cfg, section) != getattr(ref, section):
                issues.append(f"{cfg.name}: [{section}] differs from {ref.name}")
    neural = [c for c in configs if not c.is_baseline]
    if neural:
        base = neural[0]
        for cfg in neural[1:]:
            for key in NEURAL_TRAIN_KEYS:
                if getattr(cfg.train, key) != getattr(base.train, key):
                    issues.append(f"{cfg.name}: train.{key} differs from {base.name}")
            if cfg.training_token_budget() != base.training_token_budget():
                issues.append(f"{cfg.name}: training-token budget differs from {base.name}")
            for key in ("d_model", "n_layers", "n_heads", "d_ff", "max_context"):
                if getattr(cfg.model, key) != getattr(base.model, key):
                    issues.append(
                        f"{cfg.name}: context encoder model.{key} differs from {base.name}"
                    )
    if parameter_counts:
        counts = {c.name: parameter_counts[c.name] for c in neural if c.name in parameter_counts}
        if counts:
            values = sorted(counts.values())
            median = values[len(values) // 2]
            for name, count in counts.items():
                if abs(count - median) > parameter_tolerance * median:
                    issues.append(
                        f"{name}: {count} parameters is outside +-{parameter_tolerance:.0%} "
                        f"of the median {median}"
                    )
    return issues
