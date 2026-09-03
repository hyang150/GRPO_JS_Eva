"""GRPO baseline (advantage) estimators.

The registry mirrors verl's ``register_adv_est``
(``verl/trainer/ppo/core_algos.py``) so an estimator written here can be
registered into verl through a thin adapter -- see ``to_verl_style``.

Convention
----------
A baseline function takes a reward matrix of shape ``(K, N)`` -- K prompts
("groups"), N sampled completions each -- and returns a *per-sample*
baseline of the same shape.

Per-sample rather than per-group is required: leave-one-out estimators
hand a different baseline to every sample inside a group, and that
independence between ``b[k, i]`` and ``r[k, i]`` is exactly what keeps the
policy-gradient estimator unbiased.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8

BASELINE_REGISTRY: dict[str, callable] = {}


def register_baseline(name: str):
    """Decorator mirroring verl's ``register_adv_est``."""

    def deco(fn):
        if name in BASELINE_REGISTRY:
            raise ValueError(f"baseline already registered: {name}")
        BASELINE_REGISTRY[name] = fn
        fn.baseline_name = name
        return fn

    return deco


def get_baseline_fn(name: str):
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name]


def _expand(mu: torch.Tensor, n: int) -> torch.Tensor:
    """(K,) group-level baseline -> (K, N) per-sample baseline."""
    return mu.unsqueeze(1).expand(-1, n).contiguous()


# --------------------------------------------------------------------------
# 1. vanilla GRPO -- the thing we are trying to beat
# --------------------------------------------------------------------------
@register_baseline("vanilla")
def vanilla_group_mean(rewards: torch.Tensor) -> torch.Tensor:
    """b[k, i] = mean_j r[k, j].  Standard GRPO."""
    k, n = rewards.shape
    return _expand(rewards.mean(dim=1), n)


# --------------------------------------------------------------------------
# 2. global batch mean -- the s=0 extreme, our key ablation
# --------------------------------------------------------------------------
@register_baseline("global")
def global_mean(rewards: torch.Tensor) -> torch.Tensor:
    """b[k, i] = mean over the whole batch.

    This is what any shrinkage estimator collapses to when it over-shrinks.
    Without this control we cannot tell "James-Stein works" apart from
    "any baseline other than the group mean works".
    """
    return rewards.mean().expand_as(rewards).contiguous()


# --------------------------------------------------------------------------
# 3. RLOO -- unbiased leave-one-out, the honest comparison point
# --------------------------------------------------------------------------
@register_baseline("rloo")
def rloo(rewards: torch.Tensor) -> torch.Tensor:
    """b[k, i] = mean of the group *excluding* sample i."""
    k, n = rewards.shape
    if n < 2:
        return torch.zeros_like(rewards)
    total = rewards.sum(dim=1, keepdim=True)
    return (total - rewards) / (n - 1)


# --------------------------------------------------------------------------
# 4. James-Stein, V fixed to 1/N -- GRPO.md as written
# --------------------------------------------------------------------------
@register_baseline("js_fixed")
def js_fixed(rewards: torch.Tensor) -> torch.Tensor:
    """James-Stein shrinkage toward the grand mean with V hard-coded to 1/N.

    This is GRPO.md section 3.3 verbatim, with two corrections:

    * the K < 4 fallback returns the *group* mean (plain GRPO), not the
      global mean -- the doc's version silently stops being GRPO;
    * ``X_bar.unsqueeze(1)`` in the doc raises on a 0-d tensor.

    Its assumption sigma^2 ~= 1 is false for 0/1 rewards, where
    sigma_k^2 = p_k(1-p_k) <= 0.25.  Kept as a faithful control so the
    over-shrinkage can be measured rather than argued about.
    """
    k, n = rewards.shape
    x = rewards.mean(dim=1)
    if k < 4:  # Stein's result needs K >= 4 when shrinking toward the mean
        return _expand(x, n)

    x_bar = x.mean()
    diff = x - x_bar
    s = (diff ** 2).sum()

    v = 1.0 / n
    c = v * (k - 3)
    shrink = torch.clamp(1.0 - c / torch.clamp(s, min=EPS), min=0.0)
    return _expand(x_bar + shrink * diff, n)


# --------------------------------------------------------------------------
# 5. James-Stein, V estimated from the within-group variance
# --------------------------------------------------------------------------
def pooled_std(rewards: torch.Tensor) -> torch.Tensor:
    """sqrt(mean_k s_k^2), s_k^2 the unbiased within-group variance.

    The one rescaling of a 0/1 reward matrix that establishes GRPO.md's
    sigma^2 ~= 1 precondition without dividing a degenerate group by zero.
    """
    return rewards.var(dim=1, unbiased=True).mean().sqrt()


@register_baseline("js_pooled")
def js_pooled(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO.md with its own precondition honoured.

    Section 2 of the spec assumes the rewards were rescaled so the per-sample
    variance is ~1 -- that is where V = 1/N comes from -- but its pseudocode
    takes the raw matrix.  Dividing by ``pooled_std`` establishes the
    precondition, and running the spec's ten steps on r/sigma then scaling
    back is algebraically this function: the shrink factor 1 - (K-3)/(N S')
    with S' = S/sigma^2 equals 1 - V_hat (K-3)/S with V_hat = sigma^2/N.
    ``tests/test_spec_conformance.py`` pins the identity to 1e-12.

    So this is not a different method: it is the spec applied as the spec
    says.  A per-group std is not usable for the rescaling -- 30-40% of
    GSM8K groups at N=8 have s_k = 0.  For 0/1 rewards V_hat tracks
    p(1-p)/N instead of the 4x-too-large 1/N.
    """
    k, n = rewards.shape
    x = rewards.mean(dim=1)
    if k < 4 or n < 2:
        return _expand(x, n)

    x_bar = x.mean()
    diff = x - x_bar
    s = (diff ** 2).sum()

    v = rewards.var(dim=1, unbiased=True).mean() / n
    c = v * (k - 3)
    shrink = torch.clamp(1.0 - c / torch.clamp(s, min=EPS), min=0.0)
    return _expand(x_bar + shrink * diff, n)


# --------------------------------------------------------------------------
# 6. Two-level leave-one-out shrinkage (arXiv:2511.03710, "JS2")
# --------------------------------------------------------------------------
@register_baseline("js_loo")
def js_loo(rewards: torch.Tensor, lambda_correction: bool = True) -> torch.Tensor:
    """Shrinkage baseline of "Shrinking the Variance" (arXiv:2511.03710).

        b[i, j] = (1 - lam_i) * mu_i^{-j} + lam_i * mubar_{-i}
        lam_i   = (n-1)/n * v_{-i} / (v_{-i} + s_{-i})

    In the paper's notation n = number of prompts and m = number of
    generations.  This file uses ``(k, n) = rewards.shape`` instead, so
    paper-n maps to code-``k`` and paper-m maps to code-``n``:

        v_{-i} = 1/(n-1) sum_{k != i} [ 1/(m(m-1)) sum_j (r_kj - mu_k)^2 ]
        s_{-i} = 1/(n-1) sum_{k != i} (mu_k - mubar_{-i})^2

    Both levels are leave-one-out: the group mean excludes sample j, the
    shrinkage target excludes prompt i.  That is what keeps the baseline
    independent of the reward it is subtracted from.

    Checked against the arXiv PDF (v1, Sec. 3.3): eq. 10 defines the two
    leave-one-out means, eq. 13-14 the plug-in v_{-i} and s_{-i}, eq. 15 the
    coefficient *with* the (n-1)/n factor, eq. 16 the baseline.  The
    ``lambda_correction`` switch is kept so the factor's effect can be
    ablated; ``True`` is the paper.
    """
    k, n = rewards.shape
    if k < 2 or n < 2:
        return _expand(rewards.mean(dim=1), n)

    mu = rewards.mean(dim=1)  # (K,)

    # within-group leave-one-out mean, mu_i^{-j}
    total = rewards.sum(dim=1, keepdim=True)
    mu_loo = (total - rewards) / (n - 1)  # (K, N)

    # per-group estimate of Var(mu_k) = s_k^2 / m
    ss = ((rewards - mu.unsqueeze(1)) ** 2).sum(dim=1)  # (K,)
    var_mu = ss / (n * (n - 1))  # (K,)

    # leave-one-prompt-out pooled sampling variance
    v_loo = (var_mu.sum() - var_mu) / (k - 1)  # (K,)

    # leave-one-prompt-out shrinkage target
    mubar_loo = (mu.sum() - mu) / (k - 1)  # (K,)

    # s_{-i}, closed form: sum_{k!=i}(mu_k - mubar_-i)^2 = (Q - mu_i^2) - (K-1)*mubar_-i^2
    q = (mu ** 2).sum()
    s_loo = ((q - mu ** 2) - (k - 1) * mubar_loo ** 2) / (k - 1)
    s_loo = torch.clamp(s_loo, min=0.0)

    lam = v_loo / torch.clamp(v_loo + s_loo, min=EPS)
    if lambda_correction:
        lam = lam * (k - 1) / k
    lam = torch.clamp(lam, 0.0, 1.0).unsqueeze(1)  # (K, 1)

    return (1.0 - lam) * mu_loo + lam * mubar_loo.unsqueeze(1)


# --------------------------------------------------------------------------
# advantage
# --------------------------------------------------------------------------
def compute_advantages(
    rewards: torch.Tensor,
    baseline: str = "vanilla",
    scale: str = "none",
    eps: float = 1e-4,
) -> torch.Tensor:
    """A = (r - b) optionally divided by a std.

    ``scale``:
      ``"none"``  -- no division (Dr.GRPO / recommended: no length or
                     difficulty bias introduced)
      ``"group"`` -- divide by the within-group std (original GRPO)
      ``"batch"`` -- divide by the batch std (verl's ``scale_rewards="batch"``)

    Note ``"group"`` is *not* a neutral choice here: dividing by a
    within-group std that goes to zero on degenerate groups is precisely
    the term Dr.GRPO (arXiv:2503.20783) identifies as a difficulty bias.
    """
    b = get_baseline_fn(baseline)(rewards)
    adv = rewards - b
    if scale == "group":
        std = rewards.std(dim=1, keepdim=True, unbiased=True)
        # A degenerate group (every completion scored the same) has no spread
        # to normalise by.  The usual `std + 1e-4` is safe under plain GRPO
        # only because its advantage is identically 0 there -- 0/1e-4 = 0.
        # A shrinkage baseline gives those groups a *non-zero* advantage by
        # design, so the same expression multiplies it by 1e4: measured
        # adv_var 1.07e5 and grad_norm 1.7e3 on GSM8K at K=8, N=8.
        # Leave such groups unscaled instead.
        adv = adv / torch.where(std > 1e-6, std + eps, torch.ones_like(std))
    elif scale == "batch":
        adv = adv / (rewards.std(unbiased=True) + eps)
    elif scale != "none":
        raise ValueError(f"scale must be none|group|batch, got {scale!r}")
    return adv


def shrunk_group_means(rewards: torch.Tensor,
                       baseline: str = "js_fixed") -> torch.Tensor:
    """The (K,) output contract GRPO.md 3.1 states: mu_shrunk per prompt.

    The registry works in per-sample (K, N) baselines because leave-one-out
    estimators need it; this is the group-level view, i.e. the mean over the
    N samples of a group.  For the group-level estimators (``vanilla``,
    ``global``, ``js_fixed``, ``js_pooled``) that is exact -- the baseline is
    constant within a group.  For ``rloo``/``js_loo`` it is a summary: the
    per-sample values differ, and averaging them is not what those estimators
    subtract.
    """
    return get_baseline_fn(baseline)(rewards).mean(dim=1)


def shrink_factor(rewards: torch.Tensor, baseline: str) -> torch.Tensor:
    """Recover the implied shrinkage 1 - lambda, for diagnostics.

    1.0 means "no shrinkage" (= vanilla GRPO), 0.0 means "fully collapsed
    onto the global mean".  Watching this distribution is how we catch
    over-shrinkage during training.

    A group whose mean coincides with the grand mean says nothing about the
    shrinkage (0/0) and comes back as NaN; reduce with ``nanmean``.  Earlier
    versions returned 1.0 there, which is why the seed-0 sweep logs a mean
    "shrink" of 0.025 for ``global`` -- by definition it is 0 -- and slightly
    inflates every other arm on steps where such a group occurs.
    """
    x = rewards.mean(dim=1)
    x_bar = x.mean()
    b = get_baseline_fn(baseline)(rewards).mean(dim=1)
    diff = x - x_bar
    ok = diff.abs() > 1e-6
    out = torch.full_like(diff, float("nan"))
    out[ok] = (b[ok] - x_bar) / diff[ok]
    return out


# --------------------------------------------------------------------------
# verl adapter
# --------------------------------------------------------------------------
def to_verl_style(name: str):
    """Wrap a (K, N) baseline into verl's flat ``index``-based signature.

    verl hands out ``token_level_rewards (bs, L)``, ``response_mask (bs, L)``
    and ``index (bs,)`` giving the prompt id of each row, and expects
    ``(advantages, returns)``.  Registering this into verl is then::

        register_adv_est("james_stein")(to_verl_style("js_loo"))
    """
    fn = get_baseline_fn(name)

    def estimator(token_level_rewards, response_mask, index, epsilon=1e-6,
                  norm_adv_by_std_in_grpo: bool = False, config=None, **kw):
        scores = token_level_rewards.sum(dim=-1)  # (bs,)
        idx = np.asarray(index)
        groups = list(dict.fromkeys(idx.tolist()))  # preserve order
        rows = [np.flatnonzero(idx == g) for g in groups]
        sizes = {len(r) for r in rows}
        if len(sizes) != 1:
            raise ValueError(f"ragged groups not supported: sizes={sorted(sizes)}")

        order = torch.as_tensor(np.concatenate(rows), device=scores.device)
        mat = scores[order].view(len(groups), sizes.pop())
        adv_mat = mat - fn(mat)

        flat = torch.empty_like(scores)
        flat[order] = adv_mat.reshape(-1)
        if norm_adv_by_std_in_grpo:
            std = mat.std(dim=1, keepdim=True, unbiased=True).expand_as(mat)
            flat[order] = flat[order] / (std.reshape(-1) + epsilon)
        out = flat.unsqueeze(-1) * response_mask
        return out, out

    estimator.__name__ = f"compute_{name}_outcome_advantage"
    return estimator
