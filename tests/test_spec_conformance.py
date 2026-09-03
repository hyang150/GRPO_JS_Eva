"""Does `js_fixed` still implement GRPO.md section 3.2?

This file transcribes the spec's ten steps literally and asserts the shipped
estimator agrees.  It is the regression test for "we changed the method while
refactoring".

Two documented departures, both asserted below:

* **Step 5 vs the pseudocode.** Step 5 says "K < 4 -> return the original
  means", i.e. the per-group means X.  The pseudocode in 3.3 returns the
  *global* mean X_bar instead, which is a different estimator (and no longer
  GRPO).  We follow the prose.
* **Step 7.** "S = 0 -> shrink_factor = 1".  We clamp to 0 instead, which is
  observationally identical: S = 0 implies diff = 0, so
  mu = X_bar + f * 0 = X_bar = X_k for every f.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from grpo_eva.baselines import compute_advantages, js_fixed, shrunk_group_means

EPS = 1e-8


def spec_mu_shrunk(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO.md 3.2, steps 1-9, transcribed. Returns mu_shrunk, shape (K,)."""
    k, n = rewards.shape
    x = rewards.mean(dim=1)                                   # 1
    if k < 4:                                                 # 5 (prose)
        return x
    x_bar = x.mean()                                          # 2
    diff = x - x_bar                                          # 3
    s = (diff ** 2).sum()                                     # 3
    c = (k - 3) / n                                           # 4
    shrink = 1.0 - c / max(float(s), EPS)                     # 6
    if float(s) == 0.0:                                       # 7
        shrink = 1.0
    shrink = max(0.0, shrink)                                 # 8
    return x_bar + shrink * diff                              # 9


def spec_advantages(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - spec_mu_shrunk(rewards).unsqueeze(1)     # 10


@pytest.mark.parametrize("k,n", [(4, 2), (8, 4), (16, 8), (32, 16), (64, 4)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_spec_on_continuous_rewards(k, n, seed):
    torch.manual_seed(seed)
    r = torch.rand(k, n, dtype=torch.float64)
    assert torch.allclose(shrunk_group_means(r), spec_mu_shrunk(r), atol=1e-12)
    assert torch.allclose(compute_advantages(r, baseline="js_fixed"),
                          spec_advantages(r), atol=1e-12)


@pytest.mark.parametrize("k,n", [(8, 4), (32, 8), (16, 16)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_spec_on_binary_rewards(k, n, seed):
    """The case that actually occurs on GSM8K."""
    torch.manual_seed(seed)
    r = (torch.rand(k, n) < 0.35).double()
    assert torch.allclose(compute_advantages(r, baseline="js_fixed"),
                          spec_advantages(r), atol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_step5_follows_the_prose_not_the_pseudocode(k):
    """K < 4 returns the per-group means, per step 5.

    GRPO.md 3.3 line 128 returns `rewards - X_bar.unsqueeze(1)` -- the global
    mean -- which also raises, since X_bar is 0-d and unsqueeze(1) is out of
    range for it.
    """
    r = torch.rand(k, 8, dtype=torch.float64)
    assert torch.allclose(shrunk_group_means(r), r.mean(dim=1))

    x_bar = r.mean(dim=1).mean()
    with pytest.raises(IndexError):          # the pseudocode as written
        _ = r - x_bar.unsqueeze(1)


def test_step7_zero_S_is_observationally_identical():
    """Clamping to 0 instead of 1 cannot change mu when every diff is 0."""
    r = torch.tensor([[0.0, 1.0]] * 8, dtype=torch.float64)   # all group means 0.5
    assert torch.allclose(shrunk_group_means(r), torch.full((8,), 0.5, dtype=torch.float64))
    assert torch.allclose(compute_advantages(r, baseline="js_fixed"),
                          spec_advantages(r), atol=1e-12)


def test_output_contract_is_the_one_the_spec_states():
    """3.1: input (K, N) rewards -> output mu_shrunk of shape (K,)."""
    r = torch.rand(16, 8)
    mu = shrunk_group_means(r)
    assert mu.shape == (16,)
    assert torch.allclose(r - mu.unsqueeze(1), compute_advantages(r, baseline="js_fixed"))


@pytest.mark.parametrize("scale", ["group", "batch"])
def test_optional_std_normalisation_variant(scale):
    """GRPO.md line 110 / 145: dividing by a std afterwards is an allowed variant."""
    torch.manual_seed(0)
    r = (torch.rand(16, 8) < 0.4).float()
    plain = compute_advantages(r, baseline="js_fixed", scale="none")
    scaled = compute_advantages(r, baseline="js_fixed", scale=scale)
    std = (r.std(dim=1, keepdim=True, unbiased=True) if scale == "group"
           else r.std(unbiased=True))
    assert torch.allclose(scaled, plain / (std + 1e-4), atol=1e-6)
    assert torch.isfinite(scaled).all()          # degenerate groups: std = 0


def test_group_std_scaling_does_not_explode_on_degenerate_groups():
    """GRPO.md line 110's variant is a footgun combined with shrinkage.

    Plain GRPO survives `adv / (std + 1e-4)` on an all-correct group only
    because its advantage is exactly 0 there.  Every shrinkage baseline gives
    such a group a non-zero advantage on purpose -- that is the point -- so
    the naive expression scales it by 1e4.
    """
    r = torch.zeros(8, 4)
    r[0] = 1.0                       # degenerate: std = 0
    r[1:, :2] = 1.0

    naive = ((r - shrunk_group_means(r).unsqueeze(1))
             / (r.std(dim=1, keepdim=True, unbiased=True) + 1e-4))
    assert naive.abs().max() > 1e3, "the hazard this test guards has vanished"

    guarded = compute_advantages(r, baseline="js_fixed", scale="group")
    assert guarded.abs().max() < 10.0
    assert torch.isfinite(guarded).all()
    # non-degenerate groups are still normalised the usual way
    assert not torch.allclose(guarded[1:], compute_advantages(r, baseline="js_fixed")[1:])


def test_js_pooled_is_the_spec_with_its_precondition_honoured():
    """GRPO.md assumes rewards were rescaled so the per-sample variance is
    ~1 (that is where V = 1/N comes from).  Dividing 0/1 rewards by the
    pooled within-group std and then running the spec's ten steps is
    algebraically js_pooled: the shrink factor 1 - (K-3)/(N S') with
    S' = S/sigma^2 equals 1 - V_hat (K-3)/S with V_hat = sigma^2/N."""
    from grpo_eva.baselines import js_pooled, pooled_std

    for seed in range(5):
        torch.manual_seed(seed)
        r = (torch.rand(16, 8) < 0.35).double()
        sigma = pooled_std(r)
        assert sigma > 0                                   # else the spec divides by 0
        mu_spec_rescaled = sigma * spec_mu_shrunk(r / sigma)
        assert torch.allclose(js_pooled(r).mean(dim=1), mu_spec_rescaled, atol=1e-12)


# ------------------------------------------------ the pseudocode, verbatim
def spec_pseudocode_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO.md 3.3 transcribed character for character (not used anywhere else)."""
    K, N = rewards.shape
    X = rewards.mean(dim=1)
    if K < 4:
        X_bar = X.mean()
        return rewards - X_bar.unsqueeze(1)
    X_bar = X.mean()
    diff = X - X_bar
    S = (diff ** 2).sum()
    c = (K - 3) / N
    eps = 1e-8
    shrink_factor = 1.0 - c / max(S, eps)
    shrink_factor = torch.clamp(shrink_factor, min=0.0)
    mu_shrunk = X_bar + shrink_factor * diff
    return rewards - mu_shrunk.unsqueeze(1)


def test_pseudocode_agrees_with_js_fixed_off_the_edge_cases():
    torch.manual_seed(4)
    r = (torch.rand(8, 8) < 0.35).double()
    assert torch.allclose(spec_pseudocode_advantages(r),
                          compute_advantages(r, baseline="js_fixed"), atol=1e-12)


def test_pseudocode_crashes_on_S_equal_zero_which_step7_claims_to_handle():
    """Python's max(S, eps) hands back the float eps when S < eps, so
    torch.clamp receives a float and raises.  Step 7 ("S = 0 -> factor 1")
    is never reached.  Ours returns the group means, finite."""
    r = torch.tensor([[0.0, 1.0]] * 8, dtype=torch.float64)      # every group mean 0.5
    with pytest.raises(TypeError):
        spec_pseudocode_advantages(r)
    assert torch.isfinite(compute_advantages(r, baseline="js_fixed")).all()


def test_pseudocode_crashes_below_K_equal_4_and_would_center_globally():
    r = torch.rand(3, 8, dtype=torch.float64)
    with pytest.raises(IndexError):
        spec_pseudocode_advantages(r)
    # what the branch *meant* (fixing only the unsqueeze) is global centering,
    # i.e. the `global` arm -- not GRPO; step 5's prose returns the group means
    meant = r - r.mean(dim=1).mean()
    assert torch.allclose(meant, compute_advantages(r, baseline="global"))
    assert torch.allclose(compute_advantages(r, baseline="js_fixed"),
                          compute_advantages(r, baseline="vanilla"))
