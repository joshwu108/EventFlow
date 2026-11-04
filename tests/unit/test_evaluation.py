"""Metric framework behaviour on synthetic fixtures with known outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from eventflow.errors import ConfigError
from eventflow.evaluation import (
    DirectionTask,
    EvalOptions,
    evaluate,
    train_synthetic_test_real,
)
from eventflow.marketdata import Provenance, SyntheticConfig, generate_synthetic
from eventflow.marketdata.io import frame_from_table
from eventflow.marketdata.preprocess import SplitConfig, chronological_split

BASE = dict(sessions_per_instrument=2, events_per_session=2000)


@pytest.fixture(scope="module")
def real():
    return generate_synthetic(SyntheticConfig(seed=10, **BASE))


@pytest.fixture(scope="module")
def same_process():
    return generate_synthetic(SyntheticConfig(seed=11, **BASE))


def test_identical_frames_have_zero_distances(real) -> None:
    report = evaluate(real, real)
    assert all(r.status == "ok" for r in report.results), [
        (r.name, r.reason) for r in report.results if r.status != "ok"
    ]
    assert report.get("event_type_marginal").values["tv"] == 0
    assert report.get("price_change_marginal").values["wasserstein1"] == 0
    assert report.get("quantity_marginal").values["log_quantity"]["ks"] == 0
    assert report.get("return_autocorrelation").values["max_abs_diff"] == 0
    assert report.get("type_transitions").values["weighted_tv"] == 0
    assert report.get("structural_validity").values["unseen_combination_rate"] == 0


def test_shifted_quantities_are_detected(real, same_process) -> None:
    shifted = generate_synthetic(
        SyntheticConfig(seed=11, quantity_log_mean=(5.0, 4.5, 4.0), **BASE)
    )
    near = evaluate(real, same_process, metrics=["quantity_marginal", "quantity_given_type"])
    far = evaluate(real, shifted, metrics=["quantity_marginal", "quantity_given_type"])
    w_near = near.get("quantity_marginal").values["log_quantity"]["wasserstein1"]
    w_far = far.get("quantity_marginal").values["log_quantity"]["wasserstein1"]
    assert w_near < 0.1 and w_far == pytest.approx(1.0, abs=0.15)


def test_missing_volatility_clustering_is_detected() -> None:
    big = dict(sessions_per_instrument=2, events_per_session=5000)
    clustered = generate_synthetic(SyntheticConfig(seed=12, **big))
    no_regimes = generate_synthetic(SyntheticConfig(seed=13, regime_persistence=0.5, **big))
    opts = EvalOptions(lags=(1, 2, 3, 5))
    report = evaluate(clustered, no_regimes, metrics=["volatility_clustering"], options=opts)
    values = report.get("volatility_clustering").values
    # Persistent regimes give clearly positive bucketed |return| autocorrelation;
    # switching every other event removes it (about +-0.03 sampling noise here).
    assert np.mean(values["real"]) > 0.1
    assert abs(np.mean(values["generated"])) < 0.05


def test_regime_conditioning_uses_real_thresholds(real, same_process) -> None:
    report = evaluate(real, same_process, metrics=["price_given_regime"])
    values = report.get("price_given_regime").values
    assert len(values["thresholds"]) == 2
    assert {"regime_0", "regime_1", "regime_2"} <= set(values)


def test_metrics_requiring_missing_capabilities_are_skipped(real) -> None:
    table = {
        "timestamp": np.arange(100.0),
        "instrument": ["SYN_A"] * 100,
        "event_type": ["trade"] * 100,
        "price_change": np.tile([0.0, 1.0, -1.0, 0.0], 25),
    }
    trades = frame_from_table(
        table, provenance=Provenance("prints", "test"), event_types=real.event_types
    )
    report = evaluate(trades, trades)
    status = {r.name: r for r in report.results}
    for name in ("side_marginal", "quantity_marginal", "event_type_marginal", "type_transitions"):
        assert status[name].status == "skipped", name
        assert "requires" in status[name].reason
    assert status["price_change_marginal"].status == "ok"


def test_per_instrument_scopes_and_json(real, same_process) -> None:
    report = evaluate(real, same_process, metrics=["side_marginal"], by_instrument=True)
    scopes = {r.scope for r in report.results}
    assert scopes == {"all", "instrument:SYN_A", "instrument:SYN_B"}
    assert '"side_marginal"' in report.to_json()


def test_unknown_metric_and_vocabulary_mismatch(real) -> None:
    with pytest.raises(ConfigError, match="unknown metrics"):
        evaluate(real, real, metrics=["nope"])
    other = real.replace(event_types=("a", "b", "c"))
    with pytest.raises(ConfigError, match="vocabularies"):
        evaluate(real, other)


def test_invalid_rates_are_reported(real) -> None:
    flags = np.zeros((2, 5), dtype=bool)
    flags[:, -1] = True
    report = evaluate(
        real, real, metrics=["structural_validity"], options=EvalOptions(invalid=flags)
    )
    values = report.get("structural_validity").values
    assert values["invalid_event_rate"] == pytest.approx(0.2)
    assert values["invalid_rate_by_step"][-1] == 1.0


def test_tstr_matches_trtr_on_same_process_data(real, same_process) -> None:
    parts = chronological_split(real, SplitConfig(0.5, 0.0, 0.5))
    assert parts.test is not None
    task = DirectionTask(lags=3, horizon=5)
    result = train_synthetic_test_real(parts.train, parts.test, same_process, task)
    assert result["status"] == "ok"
    # Side persistence makes direction predictable well above chance.
    assert result["trtr"]["accuracy"] > result["trtr"]["majority_accuracy"] + 0.05
    assert abs(result["accuracy_gap"]) < 0.05
    report = evaluate(real, same_process, metrics=[], downstream=result)
    assert report.get("tstr_direction").category == "downstream"
