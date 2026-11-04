"""Hand-checkable values for every distance and statistic."""

from __future__ import annotations

import numpy as np
import pytest

from eventflow.evaluation.statistics import (
    autocorrelation,
    bucket_sums,
    categorical_distribution,
    excess_kurtosis,
    hill_tail_index,
    integer_histograms,
    jensen_shannon,
    ks_statistic,
    total_variation,
    wasserstein_1d,
)


def test_total_variation_and_js_extremes() -> None:
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert total_variation(p, p) == 0 and total_variation(p, q) == 1
    assert jensen_shannon(p, p) == 0
    assert jensen_shannon(p, q) == pytest.approx(np.log(2))


def test_categorical_distribution_ignores_out_of_support() -> None:
    np.testing.assert_allclose(categorical_distribution(np.array([0, 0, 1, 5]), 2), [2 / 3, 1 / 3])


def test_wasserstein_of_a_shift_equals_the_shift() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert wasserstein_1d(x, x + 0.7) == pytest.approx(0.7)
    assert wasserstein_1d(np.array([0.0, 1.0]), np.array([0.0, 3.0])) == pytest.approx(1.0)


def test_ks_statistic() -> None:
    assert ks_statistic(np.arange(10.0), np.arange(10.0)) == 0
    assert ks_statistic(np.zeros(5), np.ones(5)) == 1
    assert ks_statistic(np.array([0.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)


def test_integer_histograms_align_supports() -> None:
    p, q = integer_histograms(np.array([-1, 0, 0]), np.array([0, 2]))
    np.testing.assert_allclose(p, [1 / 3, 2 / 3, 0, 0])
    np.testing.assert_allclose(q, [0, 0.5, 0, 0.5])


def test_hill_estimator_recovers_pareto_index() -> None:
    rng = np.random.default_rng(1)
    x = 1 + rng.pareto(2.5, size=200_000)  # Pareto(alpha=2.5) on [1, inf)
    assert hill_tail_index(x, k=2000) == pytest.approx(2.5, rel=0.08)
    assert np.isnan(hill_tail_index(np.ones(3)))


def test_excess_kurtosis_of_gaussian_and_laplace() -> None:
    rng = np.random.default_rng(2)
    assert excess_kurtosis(rng.normal(size=200_000)) == pytest.approx(0.0, abs=0.05)
    assert excess_kurtosis(rng.laplace(size=200_000)) == pytest.approx(3.0, abs=0.2)


def test_autocorrelation_of_ar1_and_session_isolation() -> None:
    rng = np.random.default_rng(3)
    phi, n = 0.6, 100_000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    acf = autocorrelation(x, np.zeros(n), (1, 2, 3))
    np.testing.assert_allclose(acf, [phi, phi**2, phi**3], atol=0.02)
    # Alternating constant sessions: correlation at lag 1 only crosses boundaries -> NaN.
    groups = np.arange(10)
    assert np.isnan(autocorrelation(np.arange(10.0), groups, (1,))[0])


def test_bucket_sums_respect_groups() -> None:
    sums, owners = bucket_sums(np.ones(7), np.array([0, 0, 0, 0, 0, 1, 1]), 2)
    np.testing.assert_allclose(sums, [2, 2, 2])
    np.testing.assert_array_equal(owners, [0, 0, 1])
