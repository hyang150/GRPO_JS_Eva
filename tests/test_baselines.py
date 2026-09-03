"""Unit tests for the baseline estimators."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

from grpo_eva.baselines import (
    BASELINE_REGISTRY, compute_advantages, get_baseline_fn,
    js_fixed, js_loo, js_pooled, shrink_factor, to_verl_style,
)

ALL = sorted(BASELINE_REGISTRY)
torch.manual_seed(0)


# ---------------------------------------------------------------- contracts
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shape", [(4, 2), (8, 4), (16, 8), (32, 16)])
def test_shape_and_finiteness(name, shape):
    r = torch.rand(shape)
    b = get_baseline_fn(name)(r)
    assert b.shape == r.shape
    assert torch.isfinite(b).all()


@pytest.mark.parametrize("name", ALL)
def test_baseline_is_within_reward_range(name):
    """A baseline is an estimate of a mean reward; it cannot leave the hull."""
    r = torch.rand(16, 8)
    b = get_baseline_fn(name)(r)
    assert (b >= r.min() - 1e-6).all() and (b <= r.max() + 1e-6).all()


@pytest.mark.parametrize("name", ALL)
def test_all_rewards_equal_gives_zero_advantage(name):
    """If every completion scores the same there is nothing to learn."""
    r = torch.full((8, 4), 0.7)
    adv = compute_advantages(r, baseline=name)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


# ------------------------------------------------------------ known formulas
def test_vanilla_advantages_sum_to_zero_per_group():
    r = torch.rand(8, 4)
    adv = compute_advantages(r, baseline="vanilla")
    assert torch.allclose(adv.sum(dim=1), torch.zeros(8), atol=1e-6)


def test_global_baseline_is_constant():
    r = torch.rand(8, 4)
    b = get_baseline_fn("global")(r)
    assert torch.allclose(b, torch.full_like(b, r.mean()))


def test_rloo_excludes_self():
    """b[k,i] must not depend on r[k,i] -- that is what keeps PG unbiased."""
    r = torch.rand(6, 5)
    b0 = get_baseline_fn("rloo")(r)
    r2 = r.clone()
    r2[2, 3] += 10.0                       # perturb one sample
    b1 = get_baseline_fn("rloo")(r2)
    assert torch.allclose(b0[2, 3], b1[2, 3])          # its own baseline: unchanged
    assert not torch.allclose(b0[2, 1], b1[2, 1])      # its neighbours': changed


# ------------------------------------------------------------- js edge cases
@pytest.mark.parametrize("fn", [js_fixed, js_pooled])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_js_falls_back_to_group_mean_when_k_below_4(fn, k):
    """Stein's result needs K >= 4 when shrinking toward the grand mean.

    GRPO.md returns the *global* mean here, which silently stops being GRPO.
    """
    r = torch.rand(k, 8)
    assert torch.allclose(fn(r), r.mean(dim=1, keepdim=True).expand(-1, 8))


@pytest.mark.parametrize("name", ALL)
def test_no_nan_when_all_group_means_are_identical(name):
    """S = 0 -> the doc divides by zero."""
    r = torch.tensor([[0.0, 1.0]] * 8)     # every group mean is exactly 0.5
    b = get_baseline_fn(name)(r)
    assert torch.isfinite(b).all()
    assert torch.allclose(b.mean(dim=1), torch.full((8,), 0.5), atol=1e-6)


@pytest.mark.parametrize("name", ALL)
def test_single_generation_does_not_crash(name):
    assert torch.isfinite(get_baseline_fn(name)(torch.rand(8, 1))).all()


@pytest.mark.parametrize("fn", [js_fixed, js_pooled])
def test_shrinkage_is_a_convex_combination(fn):
    """Positive-part JS: the baseline lies between X_k and the grand mean."""
    r = torch.rand(16, 4)
    x = r.mean(dim=1)
    x_bar = x.mean()
    b = fn(r).mean(dim=1)
    lo, hi = torch.minimum(x, x_bar), torch.maximum(x, x_bar)
    assert (b >= lo - 1e-6).all() and (b <= hi + 1e-6).all()


def test_js_loo_closed_form_matches_naive_loop():
    """Guard the O(K) algebra that replaced the O(K^2) double sum."""
    torch.manual_seed(7)
    r = torch.rand(9, 5, dtype=torch.float64)
    k = r.shape[0]
    mu = r.mean(dim=1)

    naive = torch.empty(k, dtype=torch.float64)
    for i in range(k):
        others = torch.cat([mu[:i], mu[i + 1:]])
        naive[i] = ((others - others.mean()) ** 2).sum() / (k - 1)

    q = (mu ** 2).sum()
    mubar_loo = (mu.sum() - mu) / (k - 1)
    closed = ((q - mu ** 2) - (k - 1) * mubar_loo ** 2) / (k - 1)
    assert torch.allclose(naive, closed, atol=1e-12)


# ------------------------------------------- the claims the project must test
def test_degenerate_group_gets_signal_from_shrinkage_but_not_from_grpo():
    """The vanishing-gradient case EBPO (arXiv:2602.05165) targets.

    A group where every sample is correct gets zero advantage under GRPO --
    the rollouts are wasted.  Shrinkage keeps a non-zero signal.
    """
    r = torch.zeros(8, 4)
    r[0] = 1.0                                   # group 0: all correct
    r[1:, :2] = 1.0                              # the rest: half correct

    assert torch.allclose(compute_advantages(r, baseline="vanilla")[0],
                          torch.zeros(4), atol=1e-8)
    for name in ("js_fixed", "js_pooled", "js_loo", "global"):
        assert compute_advantages(r, baseline=name)[0].abs().max() > 1e-3, name


def test_js_fixed_over_shrinks_relative_to_js_pooled_on_binary_rewards():
    """The core defect: V = 1/N assumes sigma^2 = 1, but 0/1 rewards have
    sigma_k^2 = p_k(1-p_k) <= 0.25, so c is >= 4x too large."""
    torch.manual_seed(3)
    p = torch.rand(32) * 0.8 + 0.1
    r = (torch.rand(32, 8) < p.unsqueeze(1)).float()

    s_fixed = shrink_factor(r, "js_fixed").mean()
    s_pooled = shrink_factor(r, "js_pooled").mean()
    assert s_fixed < s_pooled, (s_fixed.item(), s_pooled.item())


# ------------------------------------------------------------- verl adapter
def test_verl_adapter_matches_direct_call_under_shuffling():
    """verl hands rows in arbitrary order with an `index` array."""
    torch.manual_seed(11)
    k, n, length = 6, 4, 3
    mat = torch.rand(k, n)

    index = np.repeat(np.arange(k), n)
    flat = mat.reshape(-1).clone()
    perm = np.random.permutation(k * n)          # shuffle rows
    index, flat = index[perm], flat[perm]

    tok = torch.zeros(k * n, length)
    tok[:, 0] = flat
    mask = torch.ones(k * n, length)

    adv, ret = to_verl_style("js_loo")(tok, mask, index)
    assert torch.allclose(adv, ret)

    expected = (mat - js_loo(mat)).reshape(-1)[perm]
    assert torch.allclose(adv[:, 0], expected, atol=1e-6)
