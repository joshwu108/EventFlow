"""Model configuration with validation and presets for the compared model families.

The four neural families share one implementation
(:class:`eventflow.models.generative.EventGenerativeModel`) and differ only in
two switches, which makes every pairwise comparison a controlled ablation:

============  ==================  ================  ==================================
preset        categorical_mode    continuous_head   isolates
============  ==================  ================  ==================================
transformer   factorized          mdn               parametric mixture density baseline
ddpm          factorized          ddpm              diffusion vs. flow (vs. hybrid)
cfm           relaxed             flow              explicit categoricals (vs. hybrid)
hybrid        factorized          flow              primary architecture
============  ==================  ================  ==================================

``factorized``: categorical fields are sampled from softmax heads in a fixed
chain order and the continuous head conditions on the sampled values.
``relaxed``: categorical fields are embedded as ``+-1`` one-hot vectors and
generated jointly with the continuous fields by the continuous head, then
decoded by (constrained) argmax -- a deliberately naive continuous treatment
of discrete data, kept as a baseline.

Order-book context is intentionally *not* a model input. During free-running
generation the book would have to be updated from generated events, which
requires a matching-engine simulator that cannot be validated against the
available datasets; conditioning on recorded book state would leak real
future information into rollouts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

from eventflow.errors import ConfigError

CategoricalMode = Literal["factorized", "relaxed"]
ContinuousHeadKind = Literal["mdn", "ddpm", "flow"]

PRESETS: dict[str, tuple[CategoricalMode, ContinuousHeadKind]] = {
    "transformer": ("factorized", "mdn"),
    "ddpm": ("factorized", "ddpm"),
    "cfm": ("relaxed", "flow"),
    "hybrid": ("factorized", "flow"),
}


@dataclass(frozen=True)
class DiffusionConfig:
    train_steps: int = 1000
    schedule: Literal["linear", "cosine"] = "cosine"
    sample_steps: int = 100
    x0_clip: float = 6.0  # standardised units; bounds the clean estimate during sampling
    prediction: Literal["epsilon"] = "epsilon"

    def __post_init__(self) -> None:
        if self.train_steps < 2 or not 1 <= self.sample_steps <= self.train_steps:
            raise ConfigError(
                "diffusion: need train_steps >= 2 and 1 <= sample_steps <= train_steps"
            )
        if self.schedule not in ("linear", "cosine"):
            raise ConfigError(f"diffusion.schedule must be linear or cosine, got {self.schedule!r}")
        if self.x0_clip <= 0:
            raise ConfigError("diffusion.x0_clip must be positive")


@dataclass(frozen=True)
class FlowConfig:
    sigma_min: float = 1e-4
    time_sampling: Literal["uniform", "logit_normal"] = "uniform"
    solver: Literal["euler", "heun", "rk4"] = "heun"
    sample_steps: int = 32

    def __post_init__(self) -> None:
        if not 0 <= self.sigma_min < 1:
            raise ConfigError("flow.sigma_min must lie in [0, 1)")
        if self.time_sampling not in ("uniform", "logit_normal"):
            raise ConfigError("flow.time_sampling must be uniform or logit_normal")
        if self.solver not in ("euler", "heun", "rk4"):
            raise ConfigError("flow.solver must be euler, heun or rk4")
        if self.sample_steps < 1:
            raise ConfigError("flow.sample_steps must be >= 1")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of a neural event model (see the module docstring)."""

    kind: str = "hybrid"
    categorical_mode: CategoricalMode = "factorized"
    continuous_head: ContinuousHeadKind = "flow"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_context: int = 256
    time_frequencies: int = 8
    head_width: int = 256
    head_depth: int = 3
    mdn_components: int = 8
    mdn_width: int | None = None  # defaults to head_width; lets MDN budgets match DDPM/flow heads
    loss_repeats: int = 4
    continuous_loss_weight: float = 1.0
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

    def __post_init__(self) -> None:
        if self.kind in PRESETS and PRESETS[self.kind] != (
            self.categorical_mode,
            self.continuous_head,
        ):
            raise ConfigError(
                f"model.kind={self.kind!r} implies {PRESETS[self.kind]}, got "
                f"({self.categorical_mode!r}, {self.continuous_head!r}); "
                "use kind='custom' for ablations"
            )
        if self.categorical_mode not in ("factorized", "relaxed"):
            raise ConfigError("model.categorical_mode must be factorized or relaxed")
        if self.continuous_head not in ("mdn", "ddpm", "flow"):
            raise ConfigError("model.continuous_head must be mdn, ddpm or flow")
        if self.categorical_mode == "relaxed" and self.continuous_head == "mdn":
            raise ConfigError("relaxed categoricals need a ddpm or flow head, not an MDN")
        if self.d_model % self.n_heads:
            raise ConfigError("model.d_model must be divisible by model.n_heads")
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "max_context",
            "head_width",
            "head_depth",
            "mdn_components",
            "loss_repeats",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"model.{name} must be >= 1")
        if self.mdn_width is not None and self.mdn_width < 1:
            raise ConfigError("model.mdn_width must be >= 1 or null")
        if not 0 <= self.dropout < 1:
            raise ConfigError("model.dropout must lie in [0, 1)")
        if self.time_frequencies < 0:
            raise ConfigError("model.time_frequencies must be >= 0")
        if self.continuous_loss_weight <= 0:
            raise ConfigError("model.continuous_loss_weight must be positive")

    @classmethod
    def preset(cls, kind: str, **overrides: Any) -> ModelConfig:
        if kind not in PRESETS:
            raise ConfigError(f"unknown preset {kind!r}; choose from {sorted(PRESETS)}")
        mode, head = PRESETS[kind]
        return cls(kind=kind, categorical_mode=mode, continuous_head=head, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        data = dict(data)
        if "kind" in data and data["kind"] in PRESETS:
            mode, head = PRESETS[data["kind"]]
            data.setdefault("categorical_mode", mode)
            data.setdefault("continuous_head", head)
        return build_dataclass(cls, data, "model")


def build_dataclass(cls: Any, data: dict[str, Any], where: str) -> Any:
    """Construct a (possibly nested) frozen dataclass, rejecting unknown keys.

    Nested dataclass fields are built recursively from dicts; lists become
    tuples so configs stay hashable and immutable.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a table, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = fields[name].default_factory  # type: ignore[misc]
        if isinstance(value, dict) and default is not dataclasses.MISSING:
            sample = default()
            if dataclasses.is_dataclass(sample):
                kwargs[name] = build_dataclass(type(sample), value, f"{where}.{name}")
                continue
        kwargs[name] = _freeze(value)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value
