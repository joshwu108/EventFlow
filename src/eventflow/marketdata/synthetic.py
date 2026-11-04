"""Deterministic synthetic market-event generator.

The generator is a small, fully specified stochastic process designed to
exercise every part of the pipeline with *known* ground truth. It is not a
market simulator and makes no claim of realism. Its stylised properties are
chosen so that evaluation metrics have predictable outcomes:

* **Regimes.** A two-state Markov chain (calm / volatile) with persistence
  ``regime_persistence`` modulates price-move probability, move size and
  event intensity. This produces volatility clustering: the autocorrelation
  of ``|price_change|`` is positive over many lags.
* **Event types.** A first-order Markov chain over ``EVENT_TYPES`` with a
  fixed transition matrix (rows sum to one).
* **Side.** With probability ``side_persistence`` an event repeats the side of
  the previous event (order-splitting style persistence).
* **Price changes.** Only trades and cancels move the price. A move is signed
  by side (buy = up, sell = down) and has geometric magnitude in ticks.
* **Quantities.** Log-normal body with a Pareto tail mixture, rounded up to a
  whole lot: strictly positive integers with a heavy right tail.
* **Interarrival times.** With probability ``zero_gap_prob`` the gap is
  exactly zero (a point mass, as observed in real message data); otherwise it
  is exponential with a regime- and burst-dependent rate. A simple
  self-excitation term raises the rate after short gaps.

Every draw comes from one ``numpy.random.Generator`` seeded from
``SyntheticConfig.seed``, so identical configs produce bit-identical frames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from eventflow.errors import ConfigError
from eventflow.marketdata.schema import (
    SIDE_BUY,
    SIDE_SELL,
    Capability,
    EventFrame,
    Provenance,
    SessionInfo,
)

EVENT_TYPES: tuple[str, ...] = ("limit_add", "cancel", "trade")
LIMIT_ADD, CANCEL, TRADE = 0, 1, 2
_MAX_EXCITATION = 4.0

# Rows: previous type; columns: next type. Trades cluster mildly.
DEFAULT_TRANSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.55, 0.30, 0.15),
    (0.50, 0.35, 0.15),
    (0.40, 0.25, 0.35),
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic process. All probabilities are per event."""

    seed: int = 0
    instruments: tuple[str, ...] = ("SYN_A", "SYN_B")
    sessions_per_instrument: int = 2
    events_per_session: int = 2000
    regime_persistence: float = 0.995
    side_persistence: float = 0.7
    move_prob: tuple[float, float] = (0.25, 0.6)  # calm, volatile
    move_geometric_p: tuple[float, float] = (0.7, 0.4)  # larger p -> smaller moves
    base_rate: float = 5.0  # events per second in the calm regime
    volatile_rate_multiplier: float = 3.0
    self_excitation: float = 0.5
    zero_gap_prob: float = 0.15
    quantity_log_mean: tuple[float, float, float] = (4.0, 3.5, 3.0)  # per event type
    quantity_log_std: float = 0.8
    quantity_tail_prob: float = 0.03
    quantity_tail_alpha: float = 1.5
    lot_size: int = 1
    session_gap: float = 3600.0
    with_book: bool = True
    transitions: tuple[tuple[float, ...], ...] = DEFAULT_TRANSITIONS

    def __post_init__(self) -> None:
        if not self.instruments:
            raise ConfigError("synthetic: at least one instrument is required")
        if self.sessions_per_instrument < 1 or self.events_per_session < 2:
            raise ConfigError("synthetic: need >=1 session per instrument and >=2 events")
        for name in (
            "regime_persistence",
            "side_persistence",
            "zero_gap_prob",
            "quantity_tail_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"synthetic: {name}={value} must lie in [0, 1]")
        for p in (*self.move_prob, *self.move_geometric_p):
            if not 0.0 < p <= 1.0:
                raise ConfigError("synthetic: move probabilities must lie in (0, 1]")
        if self.base_rate <= 0 or self.volatile_rate_multiplier <= 0 or self.lot_size < 1:
            raise ConfigError("synthetic: rates and lot size must be positive")
        matrix = np.asarray(self.transitions, dtype=np.float64)
        if matrix.shape != (len(EVENT_TYPES), len(EVENT_TYPES)):
            raise ConfigError("synthetic: transitions must be a 3x3 matrix")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
            raise ConfigError("synthetic: transition rows must be probability vectors")
        if len(self.quantity_log_mean) != len(EVENT_TYPES):
            raise ConfigError("synthetic: quantity_log_mean needs one entry per event type")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class _SessionDraw:
    timestamp: np.ndarray
    event_type: np.ndarray
    side: np.ndarray
    price_change: np.ndarray
    quantity: np.ndarray
    regime: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    extras: dict[str, float] = field(default_factory=dict)


def _draw_session(cfg: SyntheticConfig, rng: np.random.Generator, start: float) -> _SessionDraw:
    n = cfg.events_per_session
    transitions = np.asarray(cfg.transitions, dtype=np.float64)
    regime = np.empty(n, dtype=np.int64)
    etype = np.empty(n, dtype=np.int64)
    side = np.empty(n, dtype=np.int64)
    dprice = np.zeros(n, dtype=np.float64)
    gaps = np.zeros(n, dtype=np.float64)

    regime[0] = rng.integers(2)
    etype[0] = rng.integers(len(EVENT_TYPES))
    side[0] = rng.integers(2)
    excitation = 0.0
    for i in range(n):
        if i > 0:
            stay = rng.random() < cfg.regime_persistence
            regime[i] = regime[i - 1] if stay else 1 - regime[i - 1]
            etype[i] = rng.choice(len(EVENT_TYPES), p=transitions[etype[i - 1]])
            keep = rng.random() < cfg.side_persistence
            side[i] = side[i - 1] if keep else rng.integers(2)
            if rng.random() < cfg.zero_gap_prob:
                gaps[i] = 0.0
            else:
                rate = cfg.base_rate * (cfg.volatile_rate_multiplier if regime[i] else 1.0)
                rate *= 1.0 + excitation
                gaps[i] = rng.exponential(1.0 / rate)
            # Hawkes-style excitation: jumps by ``self_excitation`` per event and
            # decays exponentially in elapsed time. The cap keeps the process
            # stationary even when zero gaps (no decay) occur in long runs.
            decay = np.exp(-gaps[i] * cfg.base_rate)
            excitation = min((excitation + cfg.self_excitation) * decay, _MAX_EXCITATION)
        if etype[i] in (TRADE, CANCEL) and rng.random() < cfg.move_prob[regime[i]]:
            magnitude = rng.geometric(cfg.move_geometric_p[regime[i]])
            dprice[i] = magnitude if side[i] == SIDE_BUY else -magnitude

    log_mean = np.asarray(cfg.quantity_log_mean)[etype]
    body = np.exp(rng.normal(log_mean, cfg.quantity_log_std))
    tail = np.exp(log_mean) * (1.0 + rng.pareto(cfg.quantity_tail_alpha, size=n)) * 5.0
    raw_qty = np.where(rng.random(n) < cfg.quantity_tail_prob, tail, body)
    quantity = np.maximum(np.ceil(raw_qty / cfg.lot_size), 1.0) * cfg.lot_size

    timestamp = start + np.cumsum(gaps)
    # Book: mid follows cumulative price changes; spread of 1-3 ticks, wider when volatile.
    mid = 10_000.0 + np.cumsum(dprice)
    half_spread = 0.5 * (1 + rng.integers(0, 2, size=n) + regime)
    bid = np.floor(mid - half_spread)
    ask = np.maximum(np.ceil(mid + half_spread), bid + 1)
    bid_size = np.ceil(np.exp(rng.normal(5.0, 0.7, size=n)))
    ask_size = np.ceil(np.exp(rng.normal(5.0, 0.7, size=n)))
    return _SessionDraw(
        timestamp, etype, side, dprice, quantity, regime, bid, ask, bid_size, ask_size
    )


def generate_synthetic(cfg: SyntheticConfig | None = None) -> EventFrame:
    """Generate a validated :class:`EventFrame` from ``cfg`` (deterministic)."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    parts: list[_SessionDraw] = []
    sessions: list[SessionInfo] = []
    instrument_ids: list[np.ndarray] = []
    session_ids: list[np.ndarray] = []
    clock = 0.0
    session = 0
    # Sessions are emitted day-major so that a chronological split keeps every
    # instrument represented in each partition.
    for day in range(cfg.sessions_per_instrument):
        for inst, name in enumerate(cfg.instruments):
            draw = _draw_session(cfg, rng, clock)
            parts.append(draw)
            sessions.append(SessionInfo(session, inst, f"{name}/day{day}", open_time=clock))
            instrument_ids.append(np.full(cfg.events_per_session, inst))
            session_ids.append(np.full(cfg.events_per_session, session))
            clock = float(draw.timestamp[-1]) + cfg.session_gap
            session += 1

    def cat(attr: str) -> np.ndarray:
        return np.concatenate([getattr(p, attr) for p in parts])

    book = (
        {
            "bid": cat("bid"),
            "ask": cat("ask"),
            "bid_size": cat("bid_size"),
            "ask_size": cat("ask_size"),
        }
        if cfg.with_book
        else None
    )
    capabilities = [
        Capability.TIMESTAMPS,
        Capability.SIDE,
        Capability.PRICE,
        Capability.QUANTITY,
        Capability.MULTI_EVENT_TYPE,
        Capability.MESSAGE_LEVEL,
        Capability.LABELS,
    ]
    if cfg.with_book:
        capabilities.append(Capability.BOOK_TOP)
    return EventFrame.from_columns(
        timestamp=cat("timestamp"),
        instrument=np.concatenate(instrument_ids),
        event_type=cat("event_type"),
        side=cat("side"),
        price_change=cat("price_change"),
        quantity=cat("quantity"),
        session=np.concatenate(session_ids),
        event_types=EVENT_TYPES,
        instruments=cfg.instruments,
        provenance=Provenance(
            source=f"synthetic:{cfg.fingerprint()[:12]}",
            adapter="eventflow.marketdata.synthetic",
            parameters=asdict(cfg),
            notes="Deterministic synthetic process; not a market simulator.",
        ),
        capabilities=capabilities,
        sessions=sessions,
        book=book,
        labels={"regime": cat("regime")},
    )


__all__ = [
    "CANCEL",
    "EVENT_TYPES",
    "LIMIT_ADD",
    "SIDE_BUY",
    "SIDE_SELL",
    "TRADE",
    "SyntheticConfig",
    "generate_synthetic",
]
