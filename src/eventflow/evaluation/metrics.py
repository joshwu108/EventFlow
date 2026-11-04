"""Generative-model metrics grouped by what they test.

Categories (kept separate in every report, never averaged together):

``marginal``     per-field distributions, ignoring time order
``conditional``  distributions conditional on the event type or on an
                 observable volatility regime
``temporal``     dependence across events: return autocorrelation,
                 volatility clustering, interarrival autocorrelation
``structural``   validity of generated events (constraint violations,
                 unseen categorical combinations, invalid-event rates)
``downstream``   usefulness for a predictive task (see :mod:`.downstream`)

Each metric declares the :class:`~eventflow.marketdata.schema.Capability`
values it needs; the runner refuses to compute it unless *both* the real and
the generated frame provide them. The statistics are inspired by the
LOB-Bench methodology (Nagy et al., 2025) but are not an implementation of
LOB-Bench and are not numerically comparable with its scores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eventflow.evaluation.statistics import (
    autocorrelation,
    bucket_sums,
    categorical_distribution,
    excess_kurtosis,
    hill_tail_index,
    integer_histograms,
    jensen_shannon,
    ks_statistic,
    quantile_table,
    total_variation,
    wasserstein_1d,
)
from eventflow.marketdata.schema import Capability, EventFrame

MIN_SAMPLES = 20


@dataclass(frozen=True)
class EvalOptions:
    lags: tuple[int, ...] = (1, 2, 5, 10, 20)
    bucket_size: int = 20
    regimes: int = 3
    regime_window: int = 50
    invalid: np.ndarray | None = None  # [B, H] flags from a rollout, if available


@dataclass(frozen=True)
class Metric:
    name: str
    category: str
    requires: frozenset[Capability]
    fn: Callable[[EventFrame, EventFrame, EvalOptions], dict[str, Any]]
    description: str = ""


@dataclass(frozen=True)
class MetricResult:
    name: str
    category: str
    status: str  # "ok", "skipped" or "insufficient_data"
    values: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    scope: str = "all"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "values": self.values,
            "reason": self.reason,
            "scope": self.scope,
        }


class InsufficientData(Exception):
    pass


def _need(*arrays: np.ndarray) -> None:
    if any(np.asarray(a).size < MIN_SAMPLES for a in arrays):
        raise InsufficientData(f"fewer than {MIN_SAMPLES} samples")


def _cat_compare(x: np.ndarray, y: np.ndarray, support: int) -> dict[str, Any]:
    _need(x, y)
    p, q = categorical_distribution(x, support), categorical_distribution(y, support)
    return {
        "tv": total_variation(p, q),
        "js": jensen_shannon(p, q),
        "real": p.tolist(),
        "generated": q.tolist(),
    }


def _continuous_compare(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    _need(x, y)
    return {
        "wasserstein1": wasserstein_1d(x, y),
        "ks": ks_statistic(x, y),
        "mean": [float(np.mean(x)), float(np.mean(y))],
        "std": [float(np.std(x)), float(np.std(y))],
        "quantiles": quantile_table(x, y),
    }


def _positive_gaps(frame: EventFrame) -> np.ndarray:
    g = frame.interarrival
    return g[np.isfinite(g) & (g > 0)]


# ------------------------------------------------------------------- marginal
def event_type_marginal(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    return _cat_compare(real.event_type, gen.event_type, len(real.event_types))


def side_marginal(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    return _cat_compare(real.side, gen.side, 3)


def price_change_marginal(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    x, y = real.price_change, gen.price_change
    _need(x, y)
    p, q = integer_histograms(x, y)
    return {
        "tv_ticks": total_variation(p, q),
        **_continuous_compare(x, y),
        "nonzero_fraction": [float(np.mean(x != 0)), float(np.mean(y != 0))],
    }


def quantity_marginal(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    return {"log_quantity": _continuous_compare(np.log(real.quantity), np.log(gen.quantity))}


def interarrival_marginal(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    gr, gg = real.interarrival, gen.interarrival
    gr, gg = gr[np.isfinite(gr)], gg[np.isfinite(gg)]
    _need(gr, gg)
    pr, pg = _positive_gaps(real), _positive_gaps(gen)

    def cv(g: np.ndarray) -> float:
        return float(g.std() / g.mean()) if g.mean() > 0 else float("nan")

    return {
        "zero_fraction": [float(np.mean(gr == 0)), float(np.mean(gg == 0))],
        "mean": [float(gr.mean()), float(gg.mean())],
        "coefficient_of_variation": [cv(gr), cv(gg)],
        "log_positive_gap": _continuous_compare(np.log(pr), np.log(pg)),
    }


def tail_statistics(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pairs = {"abs_price_change": (np.abs(real.price_change), np.abs(gen.price_change))}
    if real.has(Capability.QUANTITY) and gen.has(Capability.QUANTITY):
        pairs["quantity"] = (real.quantity, gen.quantity)
    for name, (x, y) in pairs.items():
        _need(x, y)
        out[name] = {
            "hill_alpha": [hill_tail_index(x), hill_tail_index(y)],
            "excess_kurtosis": [excess_kurtosis(x), excess_kurtosis(y)],
            "q99": [float(np.quantile(x, 0.99)), float(np.quantile(y, 0.99))],
            "q999": [float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999))],
        }
    return out


# ---------------------------------------------------------------- conditional
def type_transitions(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    """Row-wise TV between event-type transition matrices, weighted by real row mass."""
    V = len(real.event_types)

    def matrix(frame: EventFrame) -> np.ndarray:
        same = frame.session[1:] == frame.session[:-1]
        m = np.zeros((V, V))
        np.add.at(m, (frame.event_type[:-1][same], frame.event_type[1:][same]), 1.0)
        return m

    mr, mg = matrix(real), matrix(gen)
    _need(np.zeros(int(mr.sum())), np.zeros(int(mg.sum())))
    weights = mr.sum(1) / mr.sum()
    rows = []
    for i in range(V):
        if mr[i].sum() == 0 or mg[i].sum() == 0:
            rows.append(float("nan"))
            continue
        rows.append(total_variation(mr[i] / mr[i].sum(), mg[i] / mg[i].sum()))
    rows_arr = np.array(rows)
    ok = np.isfinite(rows_arr)
    weighted = (
        float(np.sum(weights[ok] * rows_arr[ok]) / weights[ok].sum()) if ok.any() else float("nan")
    )
    return {"weighted_tv": weighted, "row_tv": rows}


def _per_type(
    real: EventFrame,
    gen: EventFrame,
    compare: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    field_of: Callable[[EventFrame], np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for code, name in enumerate(real.event_types):
        x, y = field_of(real)[real.event_type == code], field_of(gen)[gen.event_type == code]
        if x.size >= MIN_SAMPLES and y.size >= MIN_SAMPLES:
            out[name] = compare(x, y)
    if not out:
        raise InsufficientData("no event type has enough samples in both frames")
    return out


def quantity_given_type(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    return _per_type(real, gen, _continuous_compare, lambda f: np.log(f.quantity))


def side_given_type(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    return _per_type(real, gen, lambda x, y: _cat_compare(x, y, 3), lambda f: f.side)


def price_given_type(real: EventFrame, gen: EventFrame, _: EvalOptions) -> dict[str, Any]:
    def compare(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        p, q = integer_histograms(x, y)
        return {"tv_ticks": total_variation(p, q), "wasserstein1": wasserstein_1d(x, y)}

    return _per_type(real, gen, compare, lambda f: f.price_change)


def observable_regime(frame: EventFrame, window: int, thresholds: np.ndarray) -> np.ndarray:
    """Volatility regime from the trailing mean ``|price change|`` over ``window`` events.

    Uses only past and current events of the same session, so it is defined
    identically for real and generated data. Events with fewer than
    ``window`` predecessors in their session get regime ``-1``.
    """
    out = np.full(len(frame), -1, dtype=np.int64)
    a_abs = np.abs(np.nan_to_num(frame.price_change))
    for a, b in frame.session_bounds():
        if b - a < window:
            continue
        csum = np.concatenate([[0.0], np.cumsum(a_abs[a:b])])
        trailing = (csum[window:] - csum[:-window]) / window
        out[a + window - 1 : b] = np.searchsorted(thresholds, trailing, side="right")
    return out


def regime_thresholds(frame: EventFrame, window: int, regimes: int) -> np.ndarray:
    probe = observable_regime(frame, window, np.array([np.inf]))
    a_abs = np.abs(np.nan_to_num(frame.price_change))
    values = []
    for a, b in frame.session_bounds():
        if b - a >= window:
            csum = np.concatenate([[0.0], np.cumsum(a_abs[a:b])])
            values.append((csum[window:] - csum[:-window]) / window)
    if not values or (probe >= 0).sum() < MIN_SAMPLES:
        raise InsufficientData("too few events with a full regime window")
    pooled = np.concatenate(values)
    return np.quantile(pooled, np.linspace(0, 1, regimes + 1)[1:-1])


def price_given_regime(real: EventFrame, gen: EventFrame, opts: EvalOptions) -> dict[str, Any]:
    """Price-change distributions within observable volatility regimes.

    Regime thresholds are terciles (by default) of the *real* trailing
    volatility and are applied unchanged to the generated frame.
    """
    thresholds = regime_thresholds(real, opts.regime_window, opts.regimes)
    rr = observable_regime(real, opts.regime_window, thresholds)
    rg = observable_regime(gen, opts.regime_window, thresholds)
    out: dict[str, Any] = {"thresholds": thresholds.tolist()}
    for k in range(opts.regimes):
        x, y = real.price_change[rr == k], gen.price_change[rg == k]
        entry: dict[str, Any] = {
            "occupancy": [
                float(np.mean(rr[rr >= 0] == k)),
                float(np.mean(rg[rg >= 0] == k)) if (rg >= 0).any() else float("nan"),
            ]
        }
        if x.size >= MIN_SAMPLES and y.size >= MIN_SAMPLES:
            p, q = integer_histograms(x, y)
            entry["tv_ticks"] = total_variation(p, q)
            entry["wasserstein1"] = wasserstein_1d(x, y)
        out[f"regime_{k}"] = entry
    return out


# ------------------------------------------------------------------- temporal
def _acf_compare(
    xr: np.ndarray, gr: np.ndarray, xg: np.ndarray, gg: np.ndarray, lags: tuple[int, ...]
) -> dict[str, Any]:
    _need(xr, xg)
    ar, ag = autocorrelation(xr, gr, lags), autocorrelation(xg, gg, lags)
    diffs = [abs(a - b) for a, b in zip(ar, ag, strict=True) if np.isfinite(a) and np.isfinite(b)]
    return {
        "lags": list(lags),
        "real": ar,
        "generated": ag,
        "max_abs_diff": float(max(diffs)) if diffs else float("nan"),
    }


def return_autocorrelation(real: EventFrame, gen: EventFrame, opts: EvalOptions) -> dict[str, Any]:
    return _acf_compare(real.price_change, real.session, gen.price_change, gen.session, opts.lags)


def volatility_clustering(real: EventFrame, gen: EventFrame, opts: EvalOptions) -> dict[str, Any]:
    """ACF of ``|sum of price changes|`` over buckets of ``bucket_size`` events."""
    rr, sr = bucket_sums(real.price_change, real.session, opts.bucket_size)
    rg, sg = bucket_sums(gen.price_change, gen.session, opts.bucket_size)
    lags = tuple(lag for lag in opts.lags if lag < 10)
    return _acf_compare(np.abs(rr), sr, np.abs(rg), sg, lags or (1,))


def interarrival_autocorrelation(
    real: EventFrame, gen: EventFrame, opts: EvalOptions
) -> dict[str, Any]:
    """ACF of ``log(1 + gap / median gap)``: bursts of activity show as positive values."""

    def series(frame: EventFrame) -> np.ndarray:
        g = frame.interarrival
        scale = np.nanmedian(g[g > 0]) if np.any(g > 0) else 1.0
        return np.log1p(g / scale)

    return _acf_compare(series(real), real.session, series(gen), gen.session, opts.lags)


# ----------------------------------------------------------------- structural
def structural_validity(real: EventFrame, gen: EventFrame, opts: EvalOptions) -> dict[str, Any]:
    """Constraint violations in the generated frame and combinations unseen in real data."""

    def combos(frame: EventFrame) -> set[tuple[int, int, float]]:
        return set(
            zip(
                frame.event_type.tolist(),
                frame.side.tolist(),
                np.nan_to_num(frame.price_change).tolist(),
                strict=True,
            )
        )

    seen = combos(real)
    gen_keys = list(
        zip(
            gen.event_type.tolist(),
            gen.side.tolist(),
            np.nan_to_num(gen.price_change).tolist(),
            strict=True,
        )
    )
    unseen = float(np.mean([k not in seen for k in gen_keys])) if gen_keys else float("nan")
    out: dict[str, Any] = {
        "nonpositive_quantity_rate": float(np.mean(~(gen.quantity > 0)))
        if gen.has(Capability.QUANTITY)
        else float("nan"),
        "negative_gap_rate": float(np.mean(np.nan_to_num(gen.interarrival) < 0)),
        "unseen_combination_rate": unseen,
    }
    if opts.invalid is not None and opts.invalid.size:
        out["invalid_event_rate"] = float(opts.invalid.mean())
        out["invalid_rate_by_step"] = opts.invalid.mean(axis=0).tolist()
    return out


_P, _Q, _S, _T, _M = (
    Capability.PRICE,
    Capability.QUANTITY,
    Capability.SIDE,
    Capability.TIMESTAMPS,
    Capability.MULTI_EVENT_TYPE,
)

REGISTRY: tuple[Metric, ...] = (
    Metric("event_type_marginal", "marginal", frozenset({_M}), event_type_marginal),
    Metric("side_marginal", "marginal", frozenset({_S}), side_marginal),
    Metric("price_change_marginal", "marginal", frozenset({_P}), price_change_marginal),
    Metric("quantity_marginal", "marginal", frozenset({_Q}), quantity_marginal),
    Metric("interarrival_marginal", "marginal", frozenset({_T}), interarrival_marginal),
    Metric("tail_statistics", "marginal", frozenset({_P}), tail_statistics),
    Metric("type_transitions", "conditional", frozenset({_M}), type_transitions),
    Metric("quantity_given_type", "conditional", frozenset({_Q, _M}), quantity_given_type),
    Metric("side_given_type", "conditional", frozenset({_S, _M}), side_given_type),
    Metric("price_given_type", "conditional", frozenset({_P, _M}), price_given_type),
    Metric("price_given_regime", "conditional", frozenset({_P}), price_given_regime),
    Metric("return_autocorrelation", "temporal", frozenset({_P}), return_autocorrelation),
    Metric("volatility_clustering", "temporal", frozenset({_P}), volatility_clustering),
    Metric(
        "interarrival_autocorrelation", "temporal", frozenset({_T}), interarrival_autocorrelation
    ),
    Metric("structural_validity", "structural", frozenset(), structural_validity),
)

METRICS: dict[str, Metric] = {m.name: m for m in REGISTRY}
