"""Elementary reversible transforms with serialisable state.

Each transform is a frozen dataclass whose fitted state is plain Python data,
so tokenizer metadata can be stored as JSON next to a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eventflow.errors import TokenizationError


@dataclass(frozen=True)
class TickCoder:
    """Signed price change in ticks <-> categorical bin index.

    ``code = clip(round(change / resolution), -max_ticks, max_ticks) + max_ticks``.
    Decoding is exact for ``|change| < max_ticks * resolution`` on the
    resolution grid; the two outermost bins are *tail* bins that decode to
    ``+-max_ticks * resolution`` and are flagged as lossy.
    """

    max_ticks: int = 8
    resolution: float = 1.0

    def __post_init__(self) -> None:
        if self.max_ticks < 1 or self.resolution <= 0:
            raise TokenizationError("TickCoder needs max_ticks >= 1 and resolution > 0")

    @property
    def num_bins(self) -> int:
        return 2 * self.max_ticks + 1

    def encode(self, change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(codes, is_tail)``; NaN changes are rejected."""
        change = np.asarray(change, dtype=np.float64)
        if np.any(~np.isfinite(change)):
            raise TokenizationError("price change contains non-finite values")
        ticks = np.round(change / self.resolution)
        is_tail = np.abs(ticks) >= self.max_ticks
        codes = np.clip(ticks, -self.max_ticks, self.max_ticks).astype(np.int64) + self.max_ticks
        return codes, is_tail

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        if np.any((codes < 0) | (codes >= self.num_bins)):
            raise TokenizationError("price code outside the bin range")
        return (codes - self.max_ticks).astype(np.float64) * self.resolution

    def off_grid(self, change: np.ndarray) -> np.ndarray:
        """Mask of changes that do not lie on the resolution grid (lossy to encode)."""
        ratio = np.asarray(change, dtype=np.float64) / self.resolution
        return ~np.isclose(ratio, np.round(ratio), atol=1e-6)


@dataclass(frozen=True)
class GroupStandardizer:
    """Per-group affine standardisation ``z = (y - mean[g]) / std[g]``.

    Groups are instrument ids. Statistics come from the training partition
    only. Groups never seen during fitting fall back to the pooled
    statistics, which is recorded rather than silently assumed: see
    :meth:`unseen_groups`.
    """

    means: tuple[float, ...]
    stds: tuple[float, ...]
    pooled_mean: float
    pooled_std: float

    @classmethod
    def fit(
        cls, values: np.ndarray, groups: np.ndarray, num_groups: int, *, std_floor: float = 1e-3
    ) -> GroupStandardizer:
        values = np.asarray(values, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.int64)
        finite = np.isfinite(values)
        if not finite.any():
            raise TokenizationError("cannot fit a standardizer on no finite values")
        pooled_mean = float(values[finite].mean())
        pooled_std = float(max(values[finite].std(), std_floor))
        means, stds = [], []
        for g in range(num_groups):
            sel = finite & (groups == g)
            if sel.sum() >= 2:
                means.append(float(values[sel].mean()))
                stds.append(float(max(values[sel].std(), std_floor)))
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
        return cls(tuple(means), tuple(stds), pooled_mean, pooled_std)

    def _params(self, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.asarray(self.means, dtype=np.float64)
        stds = np.asarray(self.stds, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.int64)
        if groups.size and (groups.min() < 0 or groups.max() >= len(means)):
            raise TokenizationError("group id outside the fitted range")
        mean = means[groups] if groups.size else np.zeros(0)
        std = stds[groups] if groups.size else np.ones(0)
        mean = np.where(np.isnan(mean), self.pooled_mean, mean)
        std = np.where(np.isnan(std), self.pooled_std, std)
        return mean, std

    def forward(self, values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        mean, std = self._params(groups)
        return (np.asarray(values, dtype=np.float64) - mean) / std

    def inverse(self, z: np.ndarray, groups: np.ndarray) -> np.ndarray:
        mean, std = self._params(groups)
        return np.asarray(z, dtype=np.float64) * std + mean

    def unseen_groups(self) -> list[int]:
        return [g for g, m in enumerate(self.means) if np.isnan(m)]

    def to_dict(self) -> dict[str, Any]:
        def clean(xs: tuple[float, ...]) -> list[float | None]:
            return [None if np.isnan(x) else x for x in xs]

        return {
            "means": clean(self.means),
            "stds": clean(self.stds),
            "pooled_mean": self.pooled_mean,
            "pooled_std": self.pooled_std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupStandardizer:
        def restore(xs: list[float | None]) -> tuple[float, ...]:
            return tuple(float("nan") if x is None else float(x) for x in xs)

        return cls(
            restore(data["means"]),
            restore(data["stds"]),
            float(data["pooled_mean"]),
            float(data["pooled_std"]),
        )


def safe_log(x: np.ndarray) -> np.ndarray:
    """``log(x)`` for strictly positive finite ``x``; raises otherwise.

    Quantities and non-zero gaps are positive by construction, so a
    non-positive value here is a pipeline bug, not data to be clipped.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise TokenizationError("log transform needs strictly positive finite values")
    return np.log(x)
