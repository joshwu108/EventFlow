"""Fixed-step ODE solvers for ``dx/dt = v(x, t)`` on ``t in [t0, t1]``.

Deliberately simple and dependency-free: explicit Euler (order 1), Heun's
method (explicit trapezoid, order 2) and classical Runge-Kutta (order 4).
Each step evaluates ``v`` 1, 2 and 4 times respectively; compare sampling
budgets in *function evaluations* (:func:`function_evaluations`), not steps.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_EVALS_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


def function_evaluations(method: str, steps: int) -> int:
    return _EVALS_PER_STEP[method] * steps


def _t(value: float, x: torch.Tensor) -> torch.Tensor:
    return torch.full((x.shape[0],), value, dtype=x.dtype, device=x.device)


def euler_step(v: VectorField, x: torch.Tensor, t: float, h: float) -> torch.Tensor:
    return x + h * v(x, _t(t, x))


def heun_step(v: VectorField, x: torch.Tensor, t: float, h: float) -> torch.Tensor:
    k1 = v(x, _t(t, x))
    k2 = v(x + h * k1, _t(t + h, x))
    return x + 0.5 * h * (k1 + k2)


def rk4_step(v: VectorField, x: torch.Tensor, t: float, h: float) -> torch.Tensor:
    k1 = v(x, _t(t, x))
    k2 = v(x + 0.5 * h * k1, _t(t + 0.5 * h, x))
    k3 = v(x + 0.5 * h * k2, _t(t + 0.5 * h, x))
    k4 = v(x + h * k3, _t(t + h, x))
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


_STEPS = {"euler": euler_step, "heun": heun_step, "rk4": rk4_step}


def integrate(
    v: VectorField,
    x0: torch.Tensor,
    *,
    steps: int,
    method: str = "heun",
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Integrate from ``t0`` to ``t1`` with ``steps`` equal steps of ``method``."""
    if method not in _STEPS:
        raise ValueError(f"unknown ODE method {method!r}; choose from {sorted(_STEPS)}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    step = _STEPS[method]
    h = (t1 - t0) / steps
    x = x0
    for i in range(steps):
        x = step(v, x, t0 + i * h, h)
    return x
