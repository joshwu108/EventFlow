"""Distribution distances and time-series statistics used by the metrics.

All functions take NumPy arrays, ignore NaNs only where documented, and
return plain floats so results serialise to JSON directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def categorical_distribution(values: np.ndarray, support: int) -> np.ndarray:
    """Empirical probabilities of integer codes ``0..support-1``."""
    values = np.asarray(values, dtype=np.int64)
    counts = np.bincount(values[(values >= 0) & (values < support)], minlength=support)
    total = counts.sum()
    return counts / total if total else np.full(support, np.nan)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """``TV(p, q) = 0.5 * sum |p - q|`` in ``[0, 1]``."""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats, in ``[0, log 2]``."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / m[nz])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def integer_histograms(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aligned empirical distributions of two integer-valued samples."""
    x = np.round(np.asarray(x)).astype(np.int64)
    y = np.round(np.asarray(y)).astype(np.int64)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    return (
        np.bincount(x - lo, minlength=hi - lo + 1) / x.size,
        np.bincount(y - lo, minlength=hi - lo + 1) / y.size,
    )


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between two empirical distributions.

    ``W1 = integral |F_x(t) - F_y(t)| dt``, evaluated on the merged support.
    """
    x, y = np.sort(np.asarray(x, dtype=np.float64)), np.sort(np.asarray(y, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([x, y]))
    widths = np.diff(grid)
    fx = np.searchsorted(x, grid[:-1], side="right") / x.size
    fy = np.searchsorted(y, grid[:-1], side="right") / y.size
    return float(np.sum(np.abs(fx - fy) * widths))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup_t |F_x(t) - F_y(t)|``."""
    x, y = np.sort(np.asarray(x, dtype=np.float64)), np.sort(np.asarray(y, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.concatenate([x, y])
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def quantile_table(
    x: np.ndarray, y: np.ndarray, qs: Sequence[float] = DEFAULT_QUANTILES
) -> dict[str, list[float]]:
    return {
        "quantiles": list(qs),
        "real": np.quantile(x, qs).tolist(),
        "generated": np.quantile(y, qs).tolist(),
    }


def hill_tail_index(x: np.ndarray, k: int | None = None) -> float:
    """Hill estimator of the tail index ``alpha`` from the ``k`` largest positive values.

    ``alpha_hat = k / sum_{i<k} log(X_(i) / X_(k))`` with order statistics
    ``X_(0) >= X_(1) >= ...``. Default ``k = ceil(sqrt(n))``. Returns NaN with
    fewer than 10 positive values; the estimate is biased for distributions
    that are not regularly varying.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x[np.isfinite(x) & (x > 0)])[::-1]
    if x.size < 10:
        return float("nan")
    k = k or int(np.ceil(np.sqrt(x.size)))
    k = min(max(k, 2), x.size - 1)
    logs = np.log(x[:k] / x[k])
    denom = logs.sum()
    return float(k / denom) if denom > 0 else float("nan")


def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4 or x.std() == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    return float(np.mean(z**4) - 3.0)


def autocorrelation(x: np.ndarray, groups: np.ndarray, lags: Sequence[int]) -> list[float]:
    """Pooled within-group autocorrelation at each lag.

    Pairs ``(x_t, x_{t+l})`` are formed only inside each group (session), so
    no correlation is computed across session boundaries. The mean and
    variance are pooled over all values.
    """
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups)
    mean, var = np.nanmean(x), np.nanvar(x)
    out = []
    for lag in lags:
        if lag <= 0 or lag >= x.size or var == 0:
            out.append(float("nan"))
            continue
        same = groups[lag:] == groups[:-lag]
        a, b = x[lag:][same] - mean, x[:-lag][same] - mean
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(float(np.mean(a[ok] * b[ok]) / var) if ok.any() else float("nan"))
    return out


def bucket_sums(x: np.ndarray, groups: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``x`` over consecutive blocks of ``size`` events within each group.

    Incomplete trailing blocks are dropped. Returns ``(sums, group_of_block)``.
    """
    sums, owners = [], []
    groups = np.asarray(groups)
    change = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    for a, b in zip(
        np.concatenate([[0], change]), np.concatenate([change, [groups.size]]), strict=True
    ):
        n_blocks = (b - a) // size
        if n_blocks:
            block = np.asarray(x[a : a + n_blocks * size], dtype=np.float64).reshape(n_blocks, size)
            sums.append(block.sum(axis=1))
            owners.append(np.full(n_blocks, groups[a]))
    if not sums:
        return np.zeros(0), np.zeros(0, dtype=groups.dtype)
    return np.concatenate(sums), np.concatenate(owners)
