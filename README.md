# GRPO_EVA — shrinkage baselines for GRPO on GSM8K

Replacing GRPO's per-group mean baseline with a James–Stein / empirical-Bayes
shrinkage estimator, and measuring whether it actually helps.

Method spec: [`GRPO.md`](GRPO.md).
Prior work: [arXiv:2511.03710](https://arxiv.org/abs/2511.03710) (shrinkage
baselines for RLVR), [arXiv:2602.05165](https://arxiv.org/abs/2602.05165) (EBPO).
Neither has released code.

## CLI

```bash
python main.py smoke                                    # GPU arch gate
python main.py synthetic --n 2 4 8 16                   # phase 1 + figure
python main.py train --baseline js_pooled --steps 200   # phase 2
python main.py sweep  --baselines vanilla js_fixed js_pooled global
python main.py gradvar --batches 200                    # phase 3
python main.py eval --n-problems 200
python main.py prompt                                   # compare system prompts
python main.py figures                                  # rebuild every figure
```

`train`/`sweep` expose the full GRPO knob set: `--beta` (KL to a frozen
reference, k3 estimator), `--num-iterations` (mu, inner updates per rollout),
`--scale` (`none`/`group`/`batch`), `--epsilon` (PPO clip), `--loss-norm`,
`--lr`, `--k`, `--n`, `--temperature`. With `--beta 0.04` the reference model
brings peak memory to 11.9 GiB of 15.9.

`sweep` runs one arm per baseline from the same seed, so every arm sees the
same prompt stream — a paired A/B rather than four independent runs.

## Baselines

Registered in `src/grpo_eva/baselines.py`, interface mirroring verl's
`register_adv_est` so any of them can be dropped into verl via `to_verl_style`.

| name | baseline `b[k,i]` | role |
|---|---|---|
| `vanilla` | group mean | GRPO, the incumbent |
| `rloo` | leave-one-out group mean | unbiased variance-reduction control |
| `global` | batch mean | **ablation**: what over-shrinkage degenerates to |
| `js_fixed` | JS toward grand mean, `V = 1/N` | GRPO.md as written |
| `js_pooled` | JS toward grand mean, `V` estimated | proposed fix |
| `js_loo` | two-level leave-one-out shrinkage | arXiv:2511.03710 "JS2" |

## Phase 1 result — synthetic, ground truth known

```bash
python main.py synthetic --k 32 --n 2 4 8 16 --trials 2000
```

`p_k ~ Beta(1.2, 2.2)` (mean 0.35, matching a small model's GSM8K pass-rate
spread), rewards `~ Bernoulli(p_k)`, K=32 prompts, 2000 trials.
MSE of the baseline against the true `p_k`, relative to `vanilla`:

| N | degenerate groups | `global` | `js_fixed` | `js_pooled` | `js_loo` |
|---|---|---|---|---|---|
| 2  | 64.6% | −39.3% | **−39.3%** | −54.1% | −51.3% |
| 4  | 36.5% | +16.9% | **+16.9%** | −40.7% | −40.0% |
| 8  | 18.5% | +129.9% | **+129.9%** | −26.4% | −26.8% |
| 16 | 8.9%  | +356.2% | **+260.2%** | −15.9% | −16.1% |

Two findings:

1. **`js_fixed` is numerically identical to `global` for N ≤ 8.** Its implied
   shrinkage factor is 0.004 — it discards the group structure entirely and
   stops being GRPO. `V = 1/N` assumes σ²≈1, but 0/1 rewards have
   σ²=p(1−p)≤0.25, so `c` is ≥4× too large. For N ≥ 4 it is *worse* than
   plain GRPO.
2. **Estimating `V` fixes it**, and the gain grows as N shrinks
   (−15.9% at N=16 → −54.1% at N=2), which is the regime the method claims.

`rloo` has exactly the same MSE as `vanilla` — leave-one-out changes the
*independence* of the baseline, not the point estimate. Its effect shows up in
`E[(r−b)²]`, not here.

![Figure 1](results/fig1_synthetic_mse.png)

## Phase 2 — GRPO on GSM8K

```bash
python main.py train --baseline js_pooled --steps 200
```

Qwen2.5-0.5B-Instruct, full fine-tune (no LoRA), K=8 prompts x N=8 generations,
512 new tokens, bf16 + gradient checkpointing on one RTX 5080.
**11.0 GiB peak / 14.5 GiB, ~25 s/step.**

Two things that were not free:

* **The prompt.** `experiments/select_prompt.py` compares four. A "reason step
  by step + \boxed{}" prompt rambles past the token budget on 38% of problems;
  the chosen terse variant truncates 13% at 512 tokens and roughly doubles
  accuracy. One-shot exemplars did not help. Truncated correct reasoning scores
  0, which is reward noise the estimator would otherwise have to absorb.
* **KV cache.** `gradient_checkpointing_enable()` sets `use_cache=False` on the
  config, which silently makes `generate()` quadratic in length. Toggling
  checkpointing off around the rollout and back on for the backward pass took a
  step from ~57 s to ~25 s.

Also note the measurement scale problem: over 60 test problems at
temperature 1.0 the same prompt scored 28.3% and 18.3% on two runs
(SE ~5pp). Accuracy at this scale cannot resolve the ~1pp effect the paper
reports — which is why Phase 3 measures the gradient directly.

## Phase 3 — paired gradient variance

```bash
python main.py gradvar --batches 200
```

Every baseline scores the **same rollouts**, so rollout randomness cancels and
the baseline is the only thing that differs. The policy is frozen: this
measures the estimator, not a training outcome. Reports `tr Cov(g) / ||E[g]||^2` — scale-invariant, because the baselines
produce gradients whose magnitude differs ~5x and raw `tr Cov` would just rank
them by scale. `tr Cov` comes from Hutchinson probes; `||E[g]||^2` from the
exact per-batch norms minus it. Intervals are a paired bootstrap over batches
(same resample for every arm, so the shared rollout noise cancels).

**Result, 200 batches, Qwen2.5-0.5B-Instruct, K=8 x N=8, mean accuracy 29.1%:**

| baseline | noise/signal | vs GRPO | 95% CI | |
|---|---|---|---|---|
| `vanilla` | 11.5 | — | — | |
| `rloo` | 11.5 | −0.3% | [−2.5, +6.9] | no difference — sanity check |
| **`js_loo`** | **9.1** | **−21.4%** | **[−51.5, −7.8]** | **significant** |
| `js_pooled` | 9.9 | −14.4% | [−38.2, +0.8] | marginal |
| `js_fixed` | 6.0 | −48.1% | [−81.3, −15.8] | biased — see below |
| `global` | 5.8 | −49.7% | [−83.5, −15.6] | biased — see below |

`js_loo` reproduces the paper's claim: −21.4% sits inside the 11.2–67.1%
band it reports, and the interval excludes zero.

**`js_fixed` and `global` look best on this metric and that reading is wrong.**
Only leave-one-out baselines are independent of the reward they are subtracted
from, so only `rloo` and `js_loo` are unbiased for the same gradient. The other
two target a *different* `E[g]`: giving an all-correct group a positive
advantage inflates `||E[g]||^2` with a component that merely pushes up easy
prompts. A larger denominator is not more signal.

`rloo ≈ vanilla` to within [−2.5, +6.9] is the check that the instrument works:
leave-one-out changes the baseline's *independence*, not its accuracy, which is
also what the synthetic study found.

`js_fixed ≈ global` for the fourth time (6.0 vs 5.8), now on real gradients.

![Figure 2](results/fig2_grad_variance.png)

### What this does not show

* 48 batches was **not** enough — every interval crossed zero. The result above
  needed 200. Do not read effect sizes off short runs here.
* `js_loo`'s formula, in particular the `(n-1)/n` factor, was transcribed from
  the paper's arXiv HTML by automated extraction and is **unverified against
  the PDF** (OpenReview blocks automated access). `baselines.py` exposes a
  `lambda_correction` switch. That `js_pooled` — derived independently — lands
  close by (9.9 vs 9.1) is reassuring but not a substitute.
* End-to-end training accuracy is not measured. At this scale it cannot be:
  the same prompt scored 28.3% and 18.3% on two 60-problem evals.

## Conformance with GRPO.md

`tests/test_spec_conformance.py` transcribes section 3.2's ten steps literally
and asserts `js_fixed` agrees to 1e-12 on both continuous and 0/1 rewards, and
that `shrunk_group_means` honours the `(K,) -> mu_shrunk` contract of 3.1.

Two departures, both asserted:

* **Step 5 contradicts the pseudocode.** The prose says K<4 returns "the
  original means" (the per-group means X); the 3.3 pseudocode returns the
  *global* mean instead — a different estimator, and one that raises, since
  `X_bar` is 0-d and `unsqueeze(1)` is out of range for it. We follow the prose.
* **Step 7.** "S = 0 -> shrink_factor = 1"; we clamp to 0. Observationally
  identical: S = 0 implies diff = 0, so mu = X_bar either way.

### The std-normalisation variant is a footgun

Line 110 offers "divide by the within-group std afterwards" as an option. Under
plain GRPO that is safe on a degenerate group only because the advantage is
exactly 0 there, so `0/(0+1e-4) = 0`. **Every shrinkage baseline gives such a
group a non-zero advantage by design** — that is the point of the method — so
the same expression multiplies it by 1e4. Measured on GSM8K at K=8, N=8 with
38% degenerate groups: `adv_var` 1.07e5, `grad_norm` 1.7e3.

`scale="group"` therefore leaves zero-spread groups unscaled. With the guard:
`adv_var` 0.57, `grad_norm` 1.2.

## Tests

```bash
python3 -m pytest tests/ -q
```

115 tests: shape/finiteness contracts, K<4 and S=0 and N=1 edge cases, the
closed-form leave-one-out algebra against a naive O(K²) loop, the verl adapter
under row shuffling, GSM8K answer extraction, spec conformance, and the claims
above stated as assertions.

## Status

- [x] Phase 1 — estimators, tests, synthetic validation (CPU only)
- [x] Phase 0 — GPU env: torch 2.13.0+cu129, sm_120 gate passed, 14.5 GiB VRAM free
- [x] Phase 2 — GRPO loop, Qwen2.5-0.5B-Instruct + GSM8K rule-based reward
- [x] Phase 3 — gradient-variance measurement: JS two-level LOO lands at −21.4% [−51.5, −7.8]
- [ ] Phase 4 — verl PR
