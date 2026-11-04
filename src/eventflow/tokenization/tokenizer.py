"""Mixed discrete/continuous event tokenizer.

Representation of one event
---------------------------

Categorical fields (in this fixed order, which is also the order in which
factorised models sample them):

``event_type``  index into the dataset's event-type vocabulary
``side``        buy / sell / unknown                       (if SIDE)
``price``       :class:`TickCoder` bin of the price change  (if PRICE)
``zero_gap``    1 if the interarrival time is (numerically) zero, else 0

Continuous fields (standardised per instrument with training statistics):

``log_quantity``  log of the quantity                          (if QUANTITY)
``log_gap``       log of the interarrival time; *masked* when ``zero_gap``
                  is 1 or when the gap is undefined (first event of a session)

Why split the gap into ``zero_gap`` and ``log_gap``: message data has a point
mass at zero elapsed time (simultaneous messages). A continuous density
cannot represent an atom, and ``log(0)`` is undefined; modelling the atom as
a Bernoulli and the positive part as a continuous variable is exact.

Conditioning available at generation time
-----------------------------------------
To predict event ``i + 1`` a model may use, for events ``0..i`` of the same
session: every encoded field, the instrument id, and ``session_time`` (seconds
since the session's first event, a function of past timestamps only). The
timestamp of event ``i + 1`` is *not* available: it is the target
``zero_gap``/``log_gap``. Order-book state is not part of the encoding;
see :mod:`eventflow.models.config` for why book context is rejected in
free-running generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eventflow.errors import CapabilityError, TokenizationError
from eventflow.marketdata.schema import (
    SIDE_UNKNOWN,
    Capability,
    EventFrame,
)
from eventflow.tokenization.transforms import GroupStandardizer, TickCoder, safe_log

EVENT_TYPE, SIDE, PRICE, ZERO_GAP = "event_type", "side", "price", "zero_gap"
LOG_QUANTITY, LOG_GAP = "log_quantity", "log_gap"
CATEGORICAL_ORDER: tuple[str, ...] = (EVENT_TYPE, SIDE, PRICE, ZERO_GAP)
TOKENIZER_VERSION = 1


@dataclass(frozen=True)
class TokenizerConfig:
    max_price_ticks: int = 8
    price_resolution: float = 1.0
    zero_gap_threshold: float = 0.0
    per_instrument: bool = True
    std_floor: float = 1e-3
    clip_std: float = 8.0

    def __post_init__(self) -> None:
        if self.zero_gap_threshold < 0:
            raise TokenizationError("zero_gap_threshold must be non-negative")
        if self.clip_std <= 0 or self.std_floor <= 0:
            raise TokenizationError("clip_std and std_floor must be positive")
        TickCoder(self.max_price_ticks, self.price_resolution)


@dataclass(frozen=True)
class FieldLayout:
    """Names and cardinalities of the encoded fields."""

    categorical: tuple[tuple[str, int], ...]
    continuous: tuple[str, ...]

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.categorical)

    @property
    def categorical_sizes(self) -> tuple[int, ...]:
        return tuple(size for _, size in self.categorical)

    def cat_index(self, name: str) -> int:
        return self.categorical_names.index(name)

    def cont_index(self, name: str) -> int:
        return self.continuous.index(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical": [list(c) for c in self.categorical],
            "continuous": list(self.continuous),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldLayout:
        return cls(
            tuple((str(n), int(s)) for n, s in data["categorical"]), tuple(data["continuous"])
        )


@dataclass(frozen=True)
class EncodedEvents:
    """Encoded events as NumPy arrays, row-aligned with the source frame."""

    categorical: np.ndarray  # [N, K] int64
    continuous: np.ndarray  # [N, D] float32
    continuous_mask: np.ndarray  # [N, D] bool, True where the value is defined
    instrument: np.ndarray  # [N] int64
    session: np.ndarray  # [N] int64
    session_time: np.ndarray  # [N] float64, seconds since session start
    timestamp: np.ndarray  # [N] float64
    price_tail: np.ndarray  # [N] bool, price change clipped to a tail bin

    def __len__(self) -> int:
        return int(self.categorical.shape[0])


@dataclass(frozen=True)
class DecodedEvents:
    """Events mapped back to physical units, with constraint bookkeeping."""

    event_type: np.ndarray
    side: np.ndarray
    price_change: np.ndarray
    quantity: np.ndarray
    interarrival: np.ndarray
    instrument: np.ndarray
    invalid: np.ndarray
    corrections: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.event_type.shape[0])


@dataclass(frozen=True)
class EventTokenizer:
    """A fitted tokenizer. Create with :meth:`fit`; immutable afterwards."""

    config: TokenizerConfig
    layout: FieldLayout
    event_types: tuple[str, ...]
    instruments: tuple[str, ...]
    capabilities: frozenset[Capability]
    standardizers: dict[str, GroupStandardizer]
    allowed: dict[str, tuple[tuple[bool, ...], ...]]
    integer_quantities: bool
    min_quantity: float
    time_scale: float

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, frame: EventFrame, config: TokenizerConfig | None = None) -> EventTokenizer:
        """Fit vocabularies, normalisation statistics and constraints on ``frame``.

        ``frame`` must be the *training* partition; fitting on evaluation data
        leaks its statistics into the model inputs.
        """
        config = config or TokenizerConfig()
        frame.require(Capability.TIMESTAMPS, purpose="tokenization")
        coder = TickCoder(config.max_price_ticks, config.price_resolution)
        categorical: list[tuple[str, int]] = [(EVENT_TYPE, len(frame.event_types))]
        if frame.has(Capability.SIDE):
            categorical.append((SIDE, 3))
        if frame.has(Capability.PRICE):
            categorical.append((PRICE, coder.num_bins))
        categorical.append((ZERO_GAP, 2))
        continuous = [LOG_QUANTITY] if frame.has(Capability.QUANTITY) else []
        continuous.append(LOG_GAP)
        layout = FieldLayout(tuple(categorical), tuple(continuous))

        groups = frame.instrument if config.per_instrument else np.zeros(len(frame), dtype=np.int64)
        num_groups = len(frame.instruments) if config.per_instrument else 1
        standardizers: dict[str, GroupStandardizer] = {}
        if LOG_QUANTITY in continuous:
            standardizers[LOG_QUANTITY] = GroupStandardizer.fit(
                safe_log(_finite(frame.quantity, "quantity")),
                groups,
                num_groups,
                std_floor=config.std_floor,
            )
        gap = frame.interarrival
        positive = np.isfinite(gap) & (gap > config.zero_gap_threshold)
        log_gap = np.full(len(frame), np.nan)
        log_gap[positive] = np.log(gap[positive])
        standardizers[LOG_GAP] = GroupStandardizer.fit(
            log_gap, groups, num_groups, std_floor=config.std_floor
        )

        provisional = cls(
            config,
            layout,
            frame.event_types,
            frame.instruments,
            frame.capabilities,
            standardizers,
            {},
            True,
            1.0,
            1.0,
        )
        encoded = provisional._encode_categorical(frame)
        allowed: dict[str, tuple[tuple[bool, ...], ...]] = {}
        types = encoded[:, 0]
        for k, (name, size) in enumerate(layout.categorical[1:], start=1):
            matrix = np.zeros((len(frame.event_types), size), dtype=bool)
            matrix[types, encoded[:, k]] = True
            allowed[name] = tuple(tuple(bool(v) for v in row) for row in matrix)

        quantities = frame.quantity[np.isfinite(frame.quantity)]
        integer = bool(quantities.size and np.all(quantities == np.round(quantities)))
        durations = [frame.timestamp[b - 1] - frame.timestamp[a] for a, b in frame.session_bounds()]
        time_scale = float(max(np.median(durations), 1e-6)) if durations else 1.0
        return cls(
            config,
            layout,
            frame.event_types,
            frame.instruments,
            frame.capabilities,
            standardizers,
            allowed,
            integer,
            float(quantities.min()) if quantities.size else 1.0,
            time_scale,
        )

    # ----------------------------------------------------------------- encoding
    @property
    def tick_coder(self) -> TickCoder:
        return TickCoder(self.config.max_price_ticks, self.config.price_resolution)

    def _groups(self, instrument: np.ndarray) -> np.ndarray:
        instrument = np.asarray(instrument, dtype=np.int64)
        return instrument if self.config.per_instrument else np.zeros_like(instrument)

    def _check_compatible(self, frame: EventFrame) -> None:
        if frame.event_types != self.event_types:
            raise TokenizationError(
                f"event-type vocabulary {frame.event_types} differs from fitted {self.event_types}"
            )
        if frame.instruments != self.instruments:
            raise TokenizationError(
                f"instrument vocabulary {frame.instruments} differs from fitted {self.instruments}"
            )
        needed = {SIDE: Capability.SIDE, PRICE: Capability.PRICE}
        for name, capability in needed.items():
            if name in self.layout.categorical_names and not frame.has(capability):
                raise CapabilityError(
                    f"tokenizer encodes '{name}' but the frame lacks {capability}"
                )
        if LOG_QUANTITY in self.layout.continuous and not frame.has(Capability.QUANTITY):
            raise CapabilityError("tokenizer encodes quantity but the frame lacks it")

    def _encode_categorical(self, frame: EventFrame) -> np.ndarray:
        columns = [frame.event_type.astype(np.int64)]
        names = self.layout.categorical_names
        if SIDE in names:
            columns.append(frame.side.astype(np.int64))
        if PRICE in names:
            codes, _ = self.tick_coder.encode(_finite(frame.price_change, "price_change"))
            columns.append(codes)
        gap = frame.interarrival
        columns.append(
            (np.isfinite(gap) & (gap <= self.config.zero_gap_threshold)).astype(np.int64)
        )
        return np.stack(columns, axis=1)

    def encode(self, frame: EventFrame) -> EncodedEvents:
        """Encode a frame. Missing values must have been resolved by preprocessing."""
        self._check_compatible(frame)
        categorical = self._encode_categorical(frame)
        groups = self._groups(frame.instrument)
        n = len(frame)
        continuous = np.zeros((n, len(self.layout.continuous)), dtype=np.float32)
        mask = np.zeros_like(continuous, dtype=bool)
        if LOG_QUANTITY in self.layout.continuous:
            k = self.layout.cont_index(LOG_QUANTITY)
            z = self.standardizers[LOG_QUANTITY].forward(
                safe_log(_finite(frame.quantity, "quantity")), groups
            )
            continuous[:, k] = z
            mask[:, k] = True
        k = self.layout.cont_index(LOG_GAP)
        gap = frame.interarrival
        positive = np.isfinite(gap) & (gap > self.config.zero_gap_threshold)
        z_gap = self.standardizers[LOG_GAP].forward(np.log(np.where(positive, gap, 1.0)), groups)
        continuous[:, k] = np.where(positive, z_gap, 0.0)
        mask[:, k] = positive

        session_time = np.zeros(n)
        for a, b in frame.session_bounds():
            session_time[a:b] = frame.timestamp[a:b] - frame.timestamp[a]
        tail = (
            self.tick_coder.encode(frame.price_change)[1]
            if PRICE in self.layout.categorical_names
            else np.zeros(n, dtype=bool)
        )
        return EncodedEvents(
            categorical,
            continuous,
            mask,
            frame.instrument.astype(np.int64),
            frame.session.astype(np.int64),
            session_time,
            frame.timestamp.copy(),
            tail,
        )

    # ----------------------------------------------------------------- decoding
    def decode(
        self,
        categorical: np.ndarray,
        continuous: np.ndarray,
        instrument: np.ndarray,
    ) -> DecodedEvents:
        """Map encoded fields back to physical units and enforce constraints.

        Constraints enforced (each application is counted in ``corrections``):

        * standardised continuous values are clipped to ``+-clip_std``;
        * quantities are positive, rounded to whole units when training
          quantities were integral, and at least the smallest training size;
        * interarrival times are exactly zero when ``zero_gap`` is set and
          strictly positive otherwise.

        Events that remain invalid (non-finite values, out-of-range codes or
        categorical combinations never observed in training) are flagged in
        ``invalid`` rather than repaired.
        """
        categorical = np.asarray(categorical, dtype=np.int64)
        continuous = np.asarray(continuous, dtype=np.float64)
        instrument = np.asarray(instrument, dtype=np.int64)
        n = categorical.shape[0]
        names = self.layout.categorical_names
        invalid = np.zeros(n, dtype=bool)
        corrections: dict[str, int] = {}
        sizes = np.asarray(self.layout.categorical_sizes)
        out_of_range = np.any((categorical < 0) | (categorical >= sizes), axis=1)
        invalid |= out_of_range
        safe_cat = np.clip(categorical, 0, sizes - 1)

        etype = safe_cat[:, 0]
        for name in names[1:]:
            table = np.asarray(self.allowed[name], dtype=bool)
            invalid |= ~table[etype, safe_cat[:, self.layout.cat_index(name)]]
        side = safe_cat[:, names.index(SIDE)] if SIDE in names else np.full(n, SIDE_UNKNOWN)
        price = (
            self.tick_coder.decode(safe_cat[:, names.index(PRICE)])
            if PRICE in names
            else np.full(n, np.nan)
        )

        nonfinite = ~np.all(np.isfinite(continuous), axis=1)
        invalid |= nonfinite
        clip = self.config.clip_std
        clipped = np.clip(np.nan_to_num(continuous, nan=0.0), -clip, clip)
        corrections["clipped_continuous"] = int(np.sum(np.abs(np.nan_to_num(continuous)) > clip))
        groups = self._groups(instrument)

        quantity = np.full(n, np.nan)
        if LOG_QUANTITY in self.layout.continuous:
            k = self.layout.cont_index(LOG_QUANTITY)
            quantity = np.exp(self.standardizers[LOG_QUANTITY].inverse(clipped[:, k], groups))
            if self.integer_quantities:
                quantity = np.round(quantity)
            below = quantity < self.min_quantity
            corrections["quantity_floor"] = int(below.sum())
            quantity = np.where(below, self.min_quantity, quantity)

        zero = safe_cat[:, names.index(ZERO_GAP)] == 1
        k = self.layout.cont_index(LOG_GAP)
        gap = np.exp(self.standardizers[LOG_GAP].inverse(clipped[:, k], groups))
        floor = max(self.config.zero_gap_threshold, 0.0)
        not_above = (~zero) & (gap <= floor)
        corrections["gap_floor"] = int(not_above.sum())
        gap = np.where(not_above, np.nextafter(floor, np.inf), gap)
        gap = np.where(zero, 0.0, gap)
        return DecodedEvents(
            event_type=etype,
            side=side.astype(np.int64),
            price_change=price,
            quantity=quantity,
            interarrival=gap,
            instrument=instrument,
            invalid=invalid,
            corrections=corrections,
        )

    def categorical_constraints(self) -> dict[str, np.ndarray]:
        """``allowed[field][event_type, value]`` matrices for constrained sampling."""
        return {name: np.asarray(m, dtype=bool) for name, m in self.allowed.items()}

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TOKENIZER_VERSION,
            "config": self.config.__dict__,
            "layout": self.layout.to_dict(),
            "event_types": list(self.event_types),
            "instruments": list(self.instruments),
            "capabilities": sorted(c.value for c in self.capabilities),
            "standardizers": {k: v.to_dict() for k, v in self.standardizers.items()},
            "allowed": {k: [list(r) for r in v] for k, v in self.allowed.items()},
            "integer_quantities": self.integer_quantities,
            "min_quantity": self.min_quantity,
            "time_scale": self.time_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventTokenizer:
        if data.get("version") != TOKENIZER_VERSION:
            raise TokenizationError(f"unsupported tokenizer version {data.get('version')}")
        return cls(
            config=TokenizerConfig(**data["config"]),
            layout=FieldLayout.from_dict(data["layout"]),
            event_types=tuple(data["event_types"]),
            instruments=tuple(data["instruments"]),
            capabilities=frozenset(Capability(c) for c in data["capabilities"]),
            standardizers={
                k: GroupStandardizer.from_dict(v) for k, v in data["standardizers"].items()
            },
            allowed={
                k: tuple(tuple(bool(x) for x in r) for r in v) for k, v in data["allowed"].items()
            },
            integer_quantities=bool(data["integer_quantities"]),
            min_quantity=float(data["min_quantity"]),
            time_scale=float(data["time_scale"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> EventTokenizer:
        return cls.from_dict(json.loads(text))


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        raise TokenizationError(
            f"{name} contains missing values; resolve them with preprocessing before encoding"
        )
    return values
