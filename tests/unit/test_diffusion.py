"""Mathematical checks of the DDPM schedule, posterior, sampler and loss."""

from __future__ import annotations

import pytest
import torch

from eventflow.diffusion import DDPMHead, NoiseSchedule, cosine_betas, linear_betas
from eventflow.errors import ConfigError
from eventflow.models.config import DiffusionConfig


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedules_are_valid_and_monotone(kind: str) -> None:
    sched = NoiseSchedule.create(kind, 1000)
    abar = sched.alphas_cumprod
    assert torch.all((sched.betas > 0) & (sched.betas < 1))
    assert torch.all(abar[1:] < abar[:-1])
    assert abar[0] > 0.99 and abar[-1] < 1e-3


def test_linear_schedule_matches_ho_et_al_endpoints() -> None:
    betas = linear_betas(1000)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)
    # Rescaling keeps the total noise comparable at other lengths.
    assert linear_betas(100)[-1].item() == pytest.approx(0.2)


def test_cosine_schedule_follows_closed_form() -> None:
    import math

    T, s = 50, 0.008
    abar = torch.cumprod(1 - cosine_betas(T, s), 0)
    t = torch.arange(1, T + 1, dtype=torch.float64) / T
    f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
    expected = f(t) / f(torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(abar[:-1], expected[:-1])  # final beta is clipped


def test_q_sample_moments() -> None:
    sched = NoiseSchedule.create("cosine", 100)
    g = torch.Generator().manual_seed(0)
    x0 = torch.full((200_000, 1), 2.0)
    t = torch.full((200_000,), 40)
    xt = sched.q_sample(x0, t, torch.randn(x0.shape, generator=g))
    abar = sched.alphas_cumprod[40].item()
    assert xt.mean().item() == pytest.approx(2.0 * abar**0.5, abs=0.01)
    assert xt.var().item() == pytest.approx(1 - abar, abs=0.01)


def test_predict_x0_inverts_q_sample_with_true_noise() -> None:
    sched = NoiseSchedule.create("linear", 200)
    x0 = torch.randn(64, 3, dtype=torch.float64)
    noise = torch.randn(64, 3, dtype=torch.float64)
    t = torch.randint(0, 200, (64,))
    xt = sched.q_sample(x0, t, noise)
    torch.testing.assert_close(sched.predict_x0(xt, t, noise), x0)


def test_posterior_matches_gaussian_product_rule() -> None:
    """q(x_{t-1}|x_t,x_0) ∝ q(x_t|x_{t-1}) q(x_{t-1}|x_0), computed independently."""
    sched = NoiseSchedule.create("cosine", 100)
    abar = sched.alphas_cumprod
    t = 37
    beta = sched.betas[t].double()
    x0 = torch.tensor([[0.7]], dtype=torch.float64)
    xt = torch.tensor([[-0.3]], dtype=torch.float64)
    # Prior N(sqrt(abar_{t-1}) x0, 1-abar_{t-1}); likelihood N(x_t; sqrt(alpha_t) x, beta_t).
    prior_mean, prior_var = abar[t - 1].sqrt() * x0, 1 - abar[t - 1]
    precision = 1 / prior_var + (1 - beta) / beta
    mean = (prior_mean / prior_var + (1 - beta).sqrt() * xt / beta) / precision
    got_mean, got_var = sched.posterior(x0, xt, torch.tensor([t]))
    torch.testing.assert_close(got_mean, mean)
    torch.testing.assert_close(got_var.squeeze(), (1 / precision).squeeze())


def test_posterior_at_first_step_returns_x0() -> None:
    sched = NoiseSchedule.create("linear", 100)
    x0, xt = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    mean, var = sched.posterior(x0, xt, torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(mean, x0)
    assert torch.all(var == 0)


def test_respacing_preserves_alpha_bar_at_kept_steps() -> None:
    sched = NoiseSchedule.create("cosine", 1000)
    sub = sched.respace(50)
    kept = sched.alphas_cumprod[sub.timesteps]
    # The last respaced beta hits MAX_BETA (abar ~ 1e-9 there), so only its
    # endpoint is distorted, and only within pure-noise territory.
    torch.testing.assert_close(sub.alphas_cumprod[:-1], kept[:-1])
    assert sub.alphas_cumprod[-1] < 1e-5
    assert sub.timesteps[0] == 0 and sub.timesteps[-1] == 999
    full = sched.respace(1000)
    torch.testing.assert_close(full.betas.double(), sched.betas.double())
    with pytest.raises(ConfigError):
        sched.respace(0)


def _oracle_head(mu: float, s: float, steps: int, sample_steps: int) -> DDPMHead:
    """DDPM head whose noise predictor is the exact posterior mean for x0 ~ N(mu, s^2)."""
    cfg = DiffusionConfig(train_steps=steps, sample_steps=sample_steps, x0_clip=50.0)
    head = DDPMHead(cond_dim=1, dim=1, config=cfg, width=8, depth=1)
    abar_all = head.schedule.alphas_cumprod

    def oracle(x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        abar = abar_all[t].to(x_t.dtype)[:, None]
        # E[eps | x_t] for Gaussian x0: sqrt(1-abar) (x_t - sqrt(abar) mu) / (abar s^2 + 1 - abar)
        return (1 - abar).sqrt() * (x_t - abar.sqrt() * mu) / (abar * s**2 + 1 - abar)

    head.predict_noise = oracle  # type: ignore[method-assign]
    return head


@pytest.mark.parametrize("sample_steps", [1000, 100])
def test_ancestral_sampler_with_oracle_recovers_gaussian(sample_steps: int) -> None:
    mu, s = 1.5, 0.5
    head = _oracle_head(mu, s, 1000, sample_steps)
    g = torch.Generator().manual_seed(1)
    x = head.sample(torch.zeros(20_000, 1), generator=g)
    assert x.mean().item() == pytest.approx(mu, abs=0.03)
    assert x.std().item() == pytest.approx(s, abs=0.03)


def test_loss_ignores_masked_dimensions_and_is_finite() -> None:
    cfg = DiffusionConfig(train_steps=50, sample_steps=10)
    head = DDPMHead(cond_dim=4, dim=2, config=cfg, width=16, depth=2, loss_repeats=3)
    cond = torch.randn(8, 4)
    x = torch.randn(8, 2)
    mask = torch.tensor([[True, False]] * 8)
    a = head.loss(cond, x, mask, generator=torch.Generator().manual_seed(0))
    b = head.loss(
        cond, x + torch.tensor([0.0, 100.0]), mask, generator=torch.Generator().manual_seed(0)
    )
    assert a.shape == (8,)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(head.sample(cond)).all()


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        DiffusionConfig(train_steps=10, sample_steps=20)
    with pytest.raises(ConfigError):
        DiffusionConfig(schedule="sigmoid")  # type: ignore[arg-type]
