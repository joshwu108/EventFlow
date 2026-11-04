"""Analytic checks of the flow-matching path, ODE solvers and head."""

from __future__ import annotations

import math

import pytest
import torch

from eventflow.errors import ConfigError
from eventflow.flow import FlowMatchingHead, OTPath, function_evaluations, integrate, sample_times
from eventflow.models.config import FlowConfig


def test_path_endpoints_and_target_is_time_derivative() -> None:
    path = OTPath(sigma_min=0.01)
    x0, x1 = torch.randn(10, 3, dtype=torch.float64), torch.randn(10, 3, dtype=torch.float64)
    torch.testing.assert_close(path.interpolate(x0, x1, torch.zeros(10, dtype=torch.float64)), x0)
    torch.testing.assert_close(
        path.interpolate(x0, x1, torch.ones(10, dtype=torch.float64)), x1 + 0.01 * x0
    )
    t, h = torch.full((10,), 0.3, dtype=torch.float64), 1e-6
    numeric = (path.interpolate(x0, x1, t + h) - path.interpolate(x0, x1, t - h)) / (2 * h)
    torch.testing.assert_close(numeric, path.target(x0, x1), atol=1e-6, rtol=1e-6)


def test_time_sampling_ranges() -> None:
    g = torch.Generator().manual_seed(0)
    for kind in ("uniform", "logit_normal"):
        t = sample_times(10_000, kind, generator=g)
        assert t.min() >= 0 and t.max() <= 1
    assert sample_times(10_000, "logit_normal", generator=g).std() < 0.25
    with pytest.raises(ValueError):
        sample_times(3, "beta")


@pytest.mark.parametrize(("method", "order"), [("euler", 1), ("heun", 2), ("rk4", 4)])
def test_solver_convergence_order_on_linear_ode(method: str, order: int) -> None:
    # dx/dt = -x, x(0) = 1  =>  x(1) = e^-1
    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    x0 = torch.ones(1, 1, dtype=torch.float64)
    errors = [abs(integrate(v, x0, steps=n, method=method).item() - math.exp(-1)) for n in (8, 16)]
    observed = math.log2(errors[0] / errors[1])
    assert observed == pytest.approx(order, abs=0.25)


def test_solvers_handle_time_dependent_field() -> None:
    # dx/dt = 2t => x(1) = x(0) + 1, exact for Heun and RK4 (polynomial of degree 1 in t).
    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 2 * t[:, None].to(x.dtype).expand_as(x)

    x0 = torch.zeros(4, 2, dtype=torch.float64)
    for method in ("heun", "rk4"):
        torch.testing.assert_close(integrate(v, x0, steps=3, method=method), x0 + 1)
    assert function_evaluations("rk4", 10) == 40
    with pytest.raises(ValueError):
        integrate(v, x0, steps=0)


def test_marginal_gaussian_flow_transports_noise_to_data() -> None:
    """With sigma_min=0 and independent coupling, x_t ~ N(t mu, (1-t)^2 + t^2 s^2) and

    v(x, t) = mu + (t s^2 - (1 - t)) / ((1 - t)^2 + t^2 s^2) (x - t mu).
    Integrating this exact marginal field from N(0, 1) must yield N(mu, s^2).
    """
    mu, s = -1.0, 0.4

    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t[:, None].to(x.dtype)
        var = (1 - t) ** 2 + t**2 * s**2
        return mu + (t * s**2 - (1 - t)) / var * (x - t * mu)

    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(50_000, 1, generator=g, dtype=torch.float64)
    x1 = integrate(v, x0, steps=50, method="rk4")
    assert x1.mean().item() == pytest.approx(mu, abs=0.01)
    assert x1.std().item() == pytest.approx(s, abs=0.01)


def test_head_loss_masks_and_sample_shapes() -> None:
    head = FlowMatchingHead(
        cond_dim=4, dim=2, config=FlowConfig(sample_steps=4), width=16, depth=2, loss_repeats=2
    )
    cond, x = torch.randn(6, 4), torch.randn(6, 2)
    mask = torch.tensor([[True, False]] * 6)
    a = head.loss(cond, x, mask, generator=torch.Generator().manual_seed(0))
    b = head.loss(
        cond, x + torch.tensor([0.0, 9.0]), mask, generator=torch.Generator().manual_seed(0)
    )
    torch.testing.assert_close(a, b)
    out = head.sample(cond, generator=torch.Generator().manual_seed(0))
    assert out.shape == (6, 2) and torch.isfinite(out).all()
    # Zero-initialised output layer: the initial field is zero, so the ODE
    # leaves the source sample unchanged.
    noise = torch.randn((6, 2), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out, noise)


@pytest.mark.slow
def test_head_learns_a_conditional_gaussian() -> None:
    torch.manual_seed(0)
    head = FlowMatchingHead(
        cond_dim=1, dim=1, config=FlowConfig(sigma_min=0.0, sample_steps=20), width=64, depth=2
    )
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(600):
        c = torch.randint(0, 2, (256, 1), generator=g).float()
        x = (4 * c - 2) + 0.3 * torch.randn(256, 1, generator=g)
        loss = head.loss(c, x, torch.ones_like(x, dtype=torch.bool), generator=g).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for value, target in ((0.0, -2.0), (1.0, 2.0)):
        samples = head.sample(torch.full((4000, 1), value), generator=g)
        assert samples.mean().item() == pytest.approx(target, abs=0.2)
        assert samples.std().item() == pytest.approx(0.3, abs=0.15)


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        FlowConfig(sigma_min=1.0)
    with pytest.raises(ConfigError):
        FlowConfig(solver="dopri5")  # type: ignore[arg-type]
