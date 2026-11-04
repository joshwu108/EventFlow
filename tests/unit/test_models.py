from __future__ import annotations

import math

import pytest
import torch

from eventflow.errors import ConfigError
from eventflow.models.config import ModelConfig
from eventflow.models.heads import CategoricalChain, MixtureDensityHead
from eventflow.tokenization import PRICE, EventTokenizer
from tests.unit.model_fixtures import KINDS, tiny_batch, tiny_config, tiny_model


@pytest.fixture(scope="module")
def tokenizer(small_frame):
    return EventTokenizer.fit(small_frame)


@pytest.fixture(scope="module")
def batch(small_frame, tokenizer):
    return tiny_batch(small_frame, tokenizer)


@pytest.mark.parametrize("kind", KINDS)
def test_loss_is_finite_and_every_parameter_gets_gradient(kind, tokenizer, batch) -> None:
    torch.manual_seed(0)
    model = tiny_model(kind, tokenizer)
    out = model.loss(batch)
    assert torch.isfinite(out.total)
    assert out.num_targets == int(batch.targets()[3].sum())
    out.total.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    # Position embeddings beyond the window length legitimately receive zero
    # gradient but still get a grad tensor; nothing may be disconnected.
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kind", KINDS)
def test_sampling_is_deterministic_given_a_generator(kind, tokenizer, batch) -> None:
    model = tiny_model(kind, tokenizer).eval()
    context = batch.inputs()
    a = model.sample_next(context, torch.Generator().manual_seed(3))
    b = model.sample_next(context, torch.Generator().manual_seed(3))
    assert torch.equal(a.categorical, b.categorical)
    torch.testing.assert_close(a.continuous, b.continuous)
    assert a.categorical.shape == (batch.batch_size, len(tokenizer.layout.categorical))
    assert a.continuous.shape == (batch.batch_size, len(tokenizer.layout.continuous))


@pytest.mark.parametrize("kind", KINDS)
def test_sampled_categories_respect_constraints(kind, tokenizer, batch) -> None:
    model = tiny_model(kind, tokenizer).eval()
    context = batch.inputs()
    allowed = torch.as_tensor(tokenizer.categorical_constraints()[PRICE])
    k = tokenizer.layout.cat_index(PRICE)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        nxt = model.sample_next(context, g)
        assert allowed[nxt.categorical[:, 0], nxt.categorical[:, k]].all()
        zero_gap = nxt.categorical[:, tokenizer.layout.cat_index("zero_gap")] == 1
        gap = tokenizer.layout.cont_index("log_gap")
        assert torch.equal(nxt.continuous_mask[:, gap], ~zero_gap)


def test_continuous_component_can_be_sampled_given_true_categoricals(tokenizer, batch) -> None:
    model = tiny_model("hybrid", tokenizer).eval()
    given = batch.categorical[:, -1]
    nxt = model.sample_next(batch.inputs(), torch.Generator().manual_seed(0), categorical=given)
    assert torch.equal(nxt.categorical, given)
    cats = model.sample_categorical(batch.inputs(), torch.Generator().manual_seed(0))
    assert cats.shape == given.shape
    relaxed = tiny_model("cfm", tokenizer).eval()
    with pytest.raises(ValueError):
        relaxed.sample_next(batch.inputs(), categorical=given)


def test_relaxation_round_trip(tokenizer, batch) -> None:
    model = tiny_model("cfm", tokenizer)
    cat = batch.categorical[:, 0]
    assert torch.equal(model.unrelax(model.relax(cat)), cat)
    assert model.categorical is None  # relaxed models carry no unused softmax heads


def test_mdn_matches_closed_form_gaussian_and_masks() -> None:
    head = MixtureDensityHead(cond_dim=2, dim=2, components=1, width=8)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net.net[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0, math.log(2.0), 0.0]))
    x = torch.tensor([[2.0, 5.0]])
    nll = head.loss(torch.zeros(1, 2), x, torch.tensor([[True, False]]))
    # Only dimension 0: N(2; mu=1, sigma=2)
    expected = 0.5 * ((2 - 1) / 2) ** 2 + math.log(2) + 0.5 * math.log(2 * math.pi)
    assert nll.item() == pytest.approx(expected, rel=1e-6)


def test_mdn_gradients_pass_gradcheck() -> None:
    torch.manual_seed(0)
    head = MixtureDensityHead(cond_dim=3, dim=2, components=2, width=6).double()
    cond = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(4, 2, dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False], [False, True], [True, True]])
    assert torch.autograd.gradcheck(lambda c: head.loss(c, x, mask), (cond,))


def test_categorical_chain_gradients_pass_gradcheck() -> None:
    torch.manual_seed(0)
    chain = CategoricalChain(("a", "b"), (3, 4), d_model=5, width=6).double()
    h = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 1], [2, 3], [1, 0]])
    assert torch.autograd.gradcheck(lambda v: sum(chain.loss(v, targets).values()), (h,))


def test_config_presets_and_serialisation() -> None:
    cfg = tiny_config("ddpm")
    restored = ModelConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    with pytest.raises(ConfigError, match="implies"):
        ModelConfig(kind="hybrid", continuous_head="mdn")
    with pytest.raises(ConfigError, match="relaxed"):
        ModelConfig(kind="custom", categorical_mode="relaxed", continuous_head="mdn")
    with pytest.raises(ConfigError, match="unknown keys"):
        ModelConfig.from_dict({"kind": "hybrid", "width": 3})
    with pytest.raises(ConfigError, match="divisible"):
        ModelConfig.preset("hybrid", d_model=30, n_heads=4)
    custom = ModelConfig(kind="custom", categorical_mode="relaxed", continuous_head="ddpm")
    assert custom.continuous_head == "ddpm"


def test_equal_head_budgets_for_ddpm_and_flow(tokenizer) -> None:
    ddpm, hybrid = tiny_model("ddpm", tokenizer), tiny_model("hybrid", tokenizer)
    assert ddpm.num_parameters() == hybrid.num_parameters()
