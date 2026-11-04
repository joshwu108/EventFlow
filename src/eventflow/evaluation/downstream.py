"""Downstream predictive usefulness: train-on-synthetic, test-on-real (TSTR).

Task: predict the direction (up / down) of the summed price change over the
next ``horizon`` events of the same session from features of the most recent
``lags`` events. Windows whose future sum is exactly zero are excluded.

Features per anchor event ``t`` (all observable at ``t``):

* the last ``lags`` price changes (clipped to ``+-clip`` ticks);
* the last ``lags`` sides encoded as ``+1`` buy, ``-1`` sell, ``0`` unknown;
* a one-hot of the event type at ``t``;
* ``log1p`` of the interarrival time at ``t``.

The classifier is L2-regularised logistic regression fitted by Newton's
method (IRLS) in NumPy, so results do not depend on an external library.
Report TRTR (train real, test real) next to TSTR: only the *gap* between them
is informative, since absolute accuracy depends on how predictable the data
is in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eventflow.marketdata.schema import SIDE_BUY, SIDE_SELL, Capability, EventFrame


@dataclass(frozen=True)
class DirectionTask:
    lags: int = 5
    horizon: int = 10
    clip: float = 5.0
    l2: float = 1.0
    iterations: int = 25


def direction_dataset(frame: EventFrame, task: DirectionTask) -> tuple[np.ndarray, np.ndarray]:
    frame.require(Capability.PRICE, Capability.SIDE, purpose="direction task")
    V = len(frame.event_types)
    rows, labels = [], []
    price = np.clip(np.nan_to_num(frame.price_change), -task.clip, task.clip)
    side = np.where(frame.side == SIDE_BUY, 1.0, np.where(frame.side == SIDE_SELL, -1.0, 0.0))
    gap = np.log1p(np.nan_to_num(frame.interarrival))
    for a, b in frame.session_bounds():
        csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(frame.price_change[a:b]))])
        for t in range(a + task.lags - 1, b - task.horizon):
            future = csum[t - a + 1 + task.horizon] - csum[t - a + 1]
            if future == 0:
                continue
            onehot = np.zeros(V)
            onehot[frame.event_type[t]] = 1.0
            window = slice(t - task.lags + 1, t + 1)
            rows.append(np.concatenate([price[window], side[window], onehot, [gap[t]]]))
            labels.append(1.0 if future > 0 else 0.0)
    if not rows:
        return np.zeros((0, 2 * task.lags + V + 1)), np.zeros(0)
    return np.asarray(rows), np.asarray(labels)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, iterations: int) -> np.ndarray:
    """Newton/IRLS for L2-regularised logistic regression; returns ``[bias, weights]``."""
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(X.shape[1])
    penalty = np.full(X.shape[1], l2)
    penalty[0] = 0.0  # the intercept is not regularised
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) + penalty * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(penalty) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _scores(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([np.ones(len(x)), x]) @ w, -30, 30)))
    eps = 1e-12
    return {
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        "log_loss": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
    }


def train_synthetic_test_real(
    real_train: EventFrame,
    real_test: EventFrame,
    generated: EventFrame,
    task: DirectionTask | None = None,
) -> dict[str, Any]:
    """Compare TRTR and TSTR scores on the same real test set."""
    task = task or DirectionTask()
    x_tr, y_tr = direction_dataset(real_train, task)
    x_te, y_te = direction_dataset(real_test, task)
    x_gen, y_gen = direction_dataset(generated, task)
    out: dict[str, Any] = {
        "samples": {"real_train": len(y_tr), "real_test": len(y_te), "generated": len(y_gen)}
    }
    if min(len(y_tr), len(y_te), len(y_gen)) < 20 or len(np.unique(y_gen)) < 2:
        out["status"] = "insufficient_data"
        return out
    out["trtr"] = _scores(fit_logistic(x_tr, y_tr, task.l2, task.iterations), x_te, y_te)
    out["tstr"] = _scores(fit_logistic(x_gen, y_gen, task.l2, task.iterations), x_te, y_te)
    out["accuracy_gap"] = out["trtr"]["accuracy"] - out["tstr"]["accuracy"]
    out["status"] = "ok"
    return out
