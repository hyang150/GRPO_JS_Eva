# Shrinkage baselines for GRPO on GSM8K

GRPO estimates each prompt's baseline from the N sampled completions of that
prompt alone. With N small that mean is noisy. This replaces it with a
James–Stein / empirical-Bayes estimator that pulls each group mean toward the
batch mean, and measures whether that helps — on Qwen2.5-0.5B-Instruct,
GSM8K, one RTX 5080 for development and two RTX 5090s for the three-seed
sweeps.

Method spec: [`GRPO.md`](GRPO.md) (given).
Prior work: [arXiv:2511.03710](https://arxiv.org/abs/2511.03710) (shrinkage
baselines for RLVR) and [arXiv:2602.05165](https://arxiv.org/abs/2602.05165)
(EBPO). Neither has released code, so everything here is a from-scratch
implementation. Write-up in ICLR format: [`report/report.pdf`](report/report.pdf)
(English) and [`report/report_zh.pdf`](report/report_zh.pdf) (中文).

## Findings

1. **The spec's formula applied to raw 0/1 rewards does not work.** GRPO.md
   states its precondition — rewards rescaled so the per-sample variance is
   ≈ 1, which is what makes `V = 1/N` — but 0/1 correctness rewards have
   `p(1−p) ≤ 0.25` and the spec's own pseudocode takes the raw matrix. Run
   that way (`js_fixed`) the shrinkage factor collapses to 0: the estimator
   becomes the global mean, the group structure is gone, and for N ≥ 4 it is
   *worse* than plain GRPO in simulation. The collapse is monotone in K, and
   at N = 2 it is an algebraic identity for every K > 6 — in training,
   `js_fixed` logged shrink = 0 on all 150 steps of all three seeds at
   K = 32, N = 2 (see [The algebra](#the-algebra)).
2. **Honouring the precondition restores Stein's behaviour.** `js_pooled` is
   exactly GRPO.md's ten steps applied after dividing the rewards by the
   pooled within-group std (`tests/test_spec_conformance.py` pins the
   identity). Its shrinkage factor lands within 0.01 of the closed-form
   `Nτ²/(1+Nτ²)` that the theory predicts, and its baseline MSE is 16–54%
   below vanilla GRPO in simulation, the gain growing as N shrinks. Read it
   as "the spec with its stated assumption honoured", not as a different
   method. None of that shows up as accuracy: see 3.
3. **End-to-end accuracy cannot resolve the effect at this scale, and the
   sweep measures why.** Three seeds at each of two group shapes leave every
   arm inside its own seed-to-seed spread. At K = 32, N = 2 `js_fixed` and
   `global` are the same algorithm, so their three same-seed runs are a
   replicate pair: they finish 0.5, 6.5 and 1.5 points apart. The effect
   reported for this setting is 0.65–1.5 points (arXiv:2511.03710 Table 2,
   K = 64, 500 steps, 5 seeds). The per-step gradient noise/signal ratio is
   58–182 for every baseline alike. The "more stable training" claim finds
   no support: no arm clipped a gradient step, entropy fell identically.
4. **The spec's optional "divide by the within-group std" is unsafe here.**
   Degenerate groups have std 0 *and*, under shrinkage, a non-zero advantage
   by design — so the usual `std + 1e-4` multiplies it by 1e4.

The paired gradient-noise measurement that was the headline of an earlier
draft (`js_loo` −21.4% [−51.5, −7.8]) is **withdrawn**: it was produced with
a projection bug and an off-policy sampler (below). It was **not rerun**:
the corrected instrument is in the code and tested on simulated gradients,
but the ~2 h per shape it needs on the model did not fit the time budget.

### Corrections log

Every number in this README is labelled by which sampler produced it.

* **2026-09-03, gradient projection.** The Hutchinson probes reused random
  entries across equal-shaped parameter tensors, invalidating the covariance
  estimate. Fixed (`probe_scheme="gaussian_stream_v2"`); a probe-free pair
  estimator was added alongside. Affects only the standalone `gradvar`
  measurement.
* **2026-09-04, off-policy rollouts.** `generate()` passed temperature, top-p
  and top-k but not `repetition_penalty`, so it fell through to the
  checkpoint's `generation_config.json`, where Qwen2.5-Instruct ships
  **1.1**. Completions were sampled from a repetition-penalised distribution
  while the loss used the raw policy's logprobs: off-policy without an
  importance correction, for every arm alike, in `train`/`sweep`, `gradvar`,
  `eval` and `select_prompt`. The loop now passes 1.0 explicitly
  (`--repetition-penalty`). The seed-0 five-arm sweep and both historical
  `gradvar` files carry the 1.1 sampler; the Phase 4b sweeps do not.
* **2026-09-04, completion mask.** It stopped only on `<|im_end|>`; Qwen2.5
  also terminates on `<|endoftext|>`, which is the pad token, so such a
  completion was scored as 512 valid tokens including padding. The mask now
  stops on every terminator `generate()` knows plus the pad id.
* **2026-09-04, shrink diagnostic.** Groups with `x_k = x̄` were counted as
  1.0 (unshrunk); they say nothing about the shrinkage (0/0) and are now NaN
  and excluded. `global`'s 0.025 in the seed-0 sweep table is that artefact.
  The synthetic tables below were regenerated with the corrected diagnostic,
  which is why `global` and `js_fixed` now read 0.000 where an earlier
  version of this README said 0.004; the MSE columns moved by < 1.5 points.

### Where to look

| | |
|---|---|
| the estimators | [`src/grpo_eva/baselines.py`](src/grpo_eva/baselines.py) |
| does it match the given spec? | [`tests/test_spec_conformance.py`](tests/test_spec_conformance.py) |
| the GRPO loop | [`src/grpo_eva/grpo.py`](src/grpo_eva/grpo.py) |
| the three-seed sweep, raw | `results/train_*_k{8n8,32n2}_s{0,1,2}.jsonl`, `results/autodl_20260904_compare_*.txt` |
| what was measured, and what was not | [Phase 3](#phase-3--paired-gradient-variance), [Status](#status) |
| the write-up, paper format | [`report/report.pdf`](report/report.pdf) (English), [`report/report_zh.pdf`](report/report_zh.pdf) (中文) |

```bash
uv sync && python main.py smoke        # torch cu129; the RTX 5080 is sm_120
python -m pytest tests/ -q             # 162 tests
```

## CLI

```bash
python main.py smoke                                    # GPU arch gate
python main.py synthetic --k 32 --n 2 4 8 16            # phase 1 + figure 1
python main.py train --baseline js_pooled --steps 200   # phase 2
python main.py sweep  --baselines vanilla js_fixed js_pooled global
python main.py gradvar --batches 200                    # phase 3
python main.py eval --n-problems 200
python main.py prompt                                   # compare system prompts
python main.py compare --glob 'train_*_k8n8_s*.jsonl'   # paired stats over a sweep
python main.py figures                                  # rebuild every figure
python main.py report                                   # compile both reports (pdflatex + xelatex)
```

`train`/`sweep` expose the full GRPO knob set: `--beta` (KL to a frozen
reference, k3 estimator), `--num-iterations` (mu, inner updates per rollout),
`--scale` (`none`/`group`/`batch`), `--epsilon` (PPO clip), `--loss-norm`,
`--lr`, `--k`, `--n`, `--temperature`, `--repetition-penalty` (1.0; see the
corrections log), `--track-grad-var` (per-step gradient noise/signal,
arXiv:2511.03710 eq. 17–18 across micro-batches, ~1.5 s/step). With
`--beta 0.04` the reference model brings peak memory to 11.9 GiB of 15.9.

`sweep` runs one arm per baseline from the same seed, so every arm sees the
same prompt stream — a paired A/B rather than four independent runs.
`--seeds 0 1 2` repeats the whole set per seed; `compare` then pairs arms
within a seed and aggregates across seeds. `compare`, `figures` and
`plot_training.py` key runs by (baseline, seed), so they refuse a `results/`
glob that mixes group shapes — pass `--glob 'train_*_k8n8_s*.jsonl'` and
`--glob 'train_*_k32n2_s*.jsonl'` separately; `figures` does this itself and
writes `fig3_training.png` for 8×8 and `fig3_training_k32n2.png` for 32×2.

The K sweep of Figure 1 is not wired into `main.py synthetic`; regenerate it
with

```bash
python experiments/synthetic_mse.py --k 4 8 16 32 64 --n 8 --trials 3000 --out results/synthetic_k_sweep.json
```

## Baselines

Registered in `src/grpo_eva/baselines.py`, interface mirroring verl's
`register_adv_est` so any of them can be dropped into verl via `to_verl_style`.

| name | baseline `b[k,i]` | role |
|---|---|---|
| `vanilla` | group mean | GRPO, the incumbent |
| `rloo` | leave-one-out group mean | unbiased variance-reduction control |
| `global` | batch mean | **ablation**: what over-shrinkage degenerates to |
| `js_fixed` | JS toward grand mean, `V = 1/N` | GRPO.md as written |
| `js_pooled` | JS toward grand mean, `V` estimated | **the spec with its σ²≈1 precondition honoured** (= spec on `r / pooled_std`) |
| `js_loo` | two-level leave-one-out shrinkage | arXiv:2511.03710 "JS2" |

Without `global` the comparison is uninterpretable: "James–Stein helps"
cannot be told apart from "any baseline other than the group mean helps".

## The algebra

Three facts about GRPO.md's estimator that the experiments below keep
running into. Notation: group means `X_k`, grand mean `X̄`,
`S = Σ_k (X_k − X̄)²`, `c = V(K−3)`, shrink factor `max(0, 1 − c/S)`,
`τ²` the variance of the true per-prompt pass rates, `σ²` the per-sample
reward variance.

**What K is.** K is the number of prompts whose rewards are on hand when the
advantages are formed — the rollout batch, not the backward micro-batch. In
verl it is the per-step prompt count (`data.train_batch_size`); in TRL the
effective batch divided by `num_generations`. Stein's phenomenon needs K ≥ 3
for a fixed shrinkage centre and K ≥ 4 when the centre is the estimated grand
mean, hence the `K − 3`. This project runs K = 8 and 32; the paper runs 64;
production runs use hundreds to thousands.

**Where the shrink factor goes as K grows.** `E[S] ≈ (K−1) Var(X_k)` and
`c = V(K−3)`, so

```
c/S  →  V_assumed / Var(X_k)  =  V_assumed / (τ² + σ²/N)        as K → ∞
```

a constant. K does not make the shrinkage stronger; it makes `S` less noisy,
so the estimator converges to whatever this ratio says.

* If the σ² ≈ 1 precondition holds, `V_assumed = 1/N` is right and the ratio
  is `1/(1 + Nτ²) < 1` always. The shrink factor tends to `Nτ²/(1 + Nτ²)`,
  the empirical-Bayes posterior weight: mild shrinkage that weakens as N
  grows or as prompts spread further apart. `js_pooled` establishes the
  precondition by dividing by the pooled std, and its measured shrink at
  K = 32 matches this formula (`Beta(1.2, 2.2)` has `τ²/σ² = 0.294`):

  | N | predicted `Nτ²/(1+Nτ²)` | `js_pooled` measured |
  |---|---|---|
  | 2 | 0.370 | 0.386 |
  | 4 | 0.541 | 0.547 |
  | 8 | 0.702 | 0.705 |
  | 16 | 0.825 | 0.827 |

* If it does not hold — 0/1 rewards, `V_assumed = 1/N` against a true
  `σ²/N ≤ 1/(4N)` — the ratio is at least `(1/N)/(τ² + 1/(4N))`, and for the
  reward distribution used here it is 1.69 at N = 8 and 3.57 at N = 2. Above
  1 the positive-part clamp fires and the estimator *is* the global mean.
  Larger K makes this certain rather than occasional, which is why the
  specified method gets worse as the batch holds more prompts.

**The N = 2 identity.** Group means of 0/1 rewards lie in [0, 1], so
`S ≤ K/4` for any K. With `c = (K−3)/N` the clamp is guaranteed whenever
`(K−3)/N > K/4`, i.e. `K > 3/(1 − N/4)`: **K > 6 at N = 2**, K > 12 at N = 3,
and never guaranteed for N ≥ 4 (there it needs `S` to be small, which is
what the K → ∞ argument supplies). At the K = 32, N = 2 sweep `c = 14.5`
against `S ≤ 8`: `js_fixed` computes exactly `global` on every batch, which
is what makes those two arms a same-algorithm replicate pair.

**What still does not transfer even with σ² ≈ 1.** Bernoulli rewards are
heteroscedastic (`σ_k² = p_k(1−p_k)`, zero for 30–70% of groups), so a
common known V is an approximation established by pooling. The shrunk
baseline depends on `r_i` itself through `X̄` and `S`, so the policy
gradient is biased — not by the constant `(1 − 1/N)` factor of plain GRPO
but nonlinearly; only the leave-one-out arms (`rloo`, `js_loo`) keep it
unbiased. And a degenerate group receives a non-zero advantage by design,
which pushes all-correct prompts up and all-wrong prompts down: that is
noise traded for bias, not signal created.

## Phase 1 result — synthetic, ground truth known

```bash
python main.py synthetic --k 32 --n 2 4 8 16 --trials 2000
```

`p_k ~ Beta(1.2, 2.2)` (mean 0.35, matching a small model's GSM8K pass-rate
spread), rewards `~ Bernoulli(p_k)`, K=32 prompts, 2000 trials, corrected
shrink diagnostic. MSE of the baseline against the true `p_k`, relative to
`vanilla`:

| N | degenerate groups | `global` | `js_fixed` | `js_pooled` | `js_loo` |
|---|---|---|---|---|---|
| 2  | 64.5% | −39.6% | **−39.6%** | −54.4% | −51.3% |
| 4  | 36.7% | +17.2% | **+17.2%** | −40.8% | −40.0% |
| 8  | 18.7% | +131.1% | **+131.0%** | −26.7% | −27.0% |
| 16 | 8.9%  | +355.5% | **+258.8%** | −15.8% | −16.0% |

Implied shrinkage factor (1 = GRPO, 0 = global mean): `js_fixed` 0.000 /
0.000 / 0.000 / 0.103, `js_pooled` 0.386 / 0.547 / 0.705 / 0.827,
`js_loo` 0.604 / 0.673 / 0.758 / 0.841 at N = 2 / 4 / 8 / 16.

Two findings:

1. **`js_fixed` is `global` for N ≤ 8.** Identical at N ≤ 4, within 0.1% at
   N = 8: it discards the group structure entirely and stops being GRPO.
   For N ≥ 4 it is *worse* than plain GRPO.
2. **Estimating `V` fixes it**, and the gain grows as N shrinks
   (−15.8% at N=16 → −54.4% at N=2), which is the regime the method claims.

`rloo` has exactly the same MSE as `vanilla` — leave-one-out changes the
*independence* of the baseline, not the point estimate. Its effect shows up in
`E[(r−b)²]`, not here.

Sweeping K at N = 8 (3000 trials) shows the collapse is monotone in K:

| K | `js_fixed` shrink | `js_fixed` MSE | `global` MSE | `js_pooled` MSE |
|---|---|---|---|---|
| 4  | 0.333 | +1.0%   | +99.1%  | −9.3%  |
| 8  | 0.058 | +84.3%  | +116.4% | −18.7% |
| 16 | 0.006 | +121.7% | +125.5% | −24.4% |
| 32 | 0.000 | +130.2% | +130.5% | −26.8% |
| 64 | 0.000 | +133.6% | +133.6% | −28.1% |

At K = 4 the specified method is roughly neutral, so a small-K experiment
would show nothing wrong; by K = 64 it is `global` to every digit.
`js_pooled` moves the other way, improving as K grows.

![Figure 1](results/fig1_synthetic_mse.png)

## Phase 2 — GRPO on GSM8K

```bash
python main.py train --baseline js_pooled --steps 200
```

Qwen2.5-0.5B-Instruct, full fine-tune (no LoRA), K=8 prompts x N=8 generations,
512 new tokens, bf16 + gradient checkpointing on one RTX 5080.
**11.0 GiB peak / 14.5 GiB, ~25 s/step.**

Everything that could confound a baseline A/B is off by default and changed
for every arm at once: β = 0 (no reference model), `scale=none` (no division
by the within-group std, the Dr.GRPO difficulty-bias term), constant loss
normalisation (completion length does not reweight the update).

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

Two more that matter for anyone reading the numbers:

* **Pure bf16 rounds most of the update away.** Parameters, gradients and
  AdamW moments are all bf16 (`--precision bf16`, the only setting that fits
  16 GB at K=8×N=8). A weight of 0.0145 has a ULP of 1.1e-4; an AdamW step at
  lr=1e-6 moves it by ~1e-6. Measured: 2.3% of sampled weights change per
  step at lr=1e-6, 90% at lr=1e-4 (`--track-update-precision`).
  `--precision mixed` holds fp32 master weights and does not fit here.
* **Measurement scale.** Over 60 test problems at temperature 1.0 the same
  prompt scored 28.3% and 18.3% on two runs (SE ~5pp). Every eval below is
  200 problems, greedy, same order, so checkpoints are compared with McNemar.

## Phase 3 — paired gradient variance

**Withdrawn, not rerun.** Every number in this section was produced with
the buggy projection and the 1.1 sampler, and the measurement was not
repeated after the fixes (time budget). The instrument is described because
it is what a rerun would use; the result is not evidence.

Every baseline scores the **same rollouts**, so rollout randomness cancels and
the baseline is the only thing that differs. The policy is frozen: this
measures the estimator, not a training outcome. Reports
`tr Cov(g) / ||E[g]||²` — scale-invariant, because the baselines produce
gradients whose magnitude differs ~5x and raw `tr Cov` would just rank them
by scale. Intervals are a paired bootstrap over batches (same resample for
every arm, so the shared rollout noise cancels).

Two estimators are written side by side:

* **Hutchinson probes.** `tr Cov` from the across-batch variance of `<g, u>`
  over 24 fixed Gaussian probes; `||E[g]||²` as the exact `E||g||²` minus it.
  Weak point: `||E[g]||²` was 8% of `E||g||²` in the historical run, so a
  few percent of probe error — which more batches do not reduce — dominates
  the ratio. `trace_cov_probe_se` is now recorded next to `signal_sq` so the
  next run can quantify it.
* **Disjoint pairs, probe-free.** For independent batches
  `E<g_b, g_b'> = ||E[g]||²` exactly. Batches are paired (0,1), (2,3), …,
  each pair gives one unbiased sample of `||E[g]||²` and one of `E||g||²`,
  and the bootstrap resamples pairs — the same idea as the paper's own eq.
  17–18 across micro-batches. Fields are prefixed `pair_`; CPU tests recover
  known `tr Cov` and `||E[g]||²` from simulated gradients. **A rerun should read the
  `pair_` fields; the probe columns exist for comparison with history.**

Historical result, unvalidated (200 batches, K=8 × N=8, mean accuracy 29.1%,
1.1 sampler, buggy projection):

| baseline | noise/signal | vs GRPO | 95% CI | |
|---|---|---|---|---|
| `vanilla` | 11.5 | — | — | |
| `rloo` | 11.5 | −0.3% | [−2.5, +6.9] | no difference — sanity check |
| `js_loo` | 9.1 | −21.4% | [−51.5, −7.8] | withdrawn |
| `js_pooled` | 9.9 | −14.4% | [−38.2, +0.8] | withdrawn |
| `js_fixed` | 6.0 | −48.1% | [−81.3, −15.8] | biased — see below |
| `global` | 5.8 | −49.7% | [−83.5, −15.6] | biased — see below |

**`js_fixed` and `global` look best on this metric and that reading is wrong.**
Only leave-one-out baselines are independent of the reward they are subtracted
from, so only `rloo` and `js_loo` are unbiased for the same gradient. The other
two target a *different* `E[g]`: giving an all-correct group a positive
advantage inflates `||E[g]||²` with a component that merely pushes up easy
prompts. A larger denominator is not more signal.

Agreement between `rloo` and `vanilla` did not validate the measurement:
their proportional gradients agree even under an incorrect projection. CPU
regression tests now compare projected covariance with exact covariance and
check that opposite gradients in separate parameters do not cancel.
48 batches was not enough — every interval crossed zero; do not read effect
sizes off short runs here.

![Figure 2, historical](results/fig2_grad_variance.png)

## Phase 4b — the spec's own regime, 1.0 sampler, three seeds

```bash
python main.py sweep --seeds 0 1 2 --k 32 --n 2 --baselines vanilla js_fixed js_pooled global --track-grad-var --steps 150
python main.py sweep --seeds 0 1 2 --k 8  --n 8 --baselines vanilla js_fixed js_pooled global --track-grad-var --steps 150
python main.py compare --glob 'train_*_k8n8_s*.jsonl'
python main.py compare --glob 'train_*_k32n2_s*.jsonl'
```

Run 2026-09-04 on two RTX 5090s (`SERVER_RUNBOOK.md`; raw logs in
`results/autodl_20260904_logs/`, `compare` output in
`results/autodl_20260904_compare_*.txt`, the 24 per-step logs in
`results/train_*_k{8n8,32n2}_s{0,1,2}.jsonl`). Same model, lr, 150 steps and
200-problem greedy eval as before; the corrected sampler, mask and shrink
diagnostic; K·N = 64 in both shapes so every arm sees the same number of
completions per step. Every run starts from the same checkpoint, so `eval@0`
is 45.0% for all 24 (it was 42.5% under the 1.1 sampler, which also penalised
the greedy eval). `rloo` and `js_loo` were not run.

Final test accuracy, mean ± s.e.m. over seeds 0–2, and the paired difference
against `vanilla` on the same seed:

| K×N | arm | final (per seed) | mean ± sem | vs `vanilla` | p (paired t, df=2) | shrink | collapsed steps |
|---|---|---|---|---|---|---|---|
| 8×8 | `vanilla` | 44.5 / 42.0 / 47.5 | 44.7 ± 1.6 | | | 1.000 | 0 |
| 8×8 | `js_pooled` | 44.0 / 45.5 / 47.0 | 45.5 ± 0.9 | +0.8 ± 1.3 | 0.60 | 0.85 | 0–1 |
| 8×8 | `js_fixed` | 44.0 / 47.5 / 46.0 | 45.8 ± 1.0 | +1.2 ± 2.2 | 0.65 | 0.21 | 38–44 |
| 8×8 | `global` | 48.5 / 44.0 / 45.0 | 45.8 ± 1.4 | +1.2 ± 1.9 | 0.61 | 0.000 | 150 |
| 32×2 | `vanilla` | 46.0 / 45.5 / 44.5 | 45.3 ± 0.4 | | | 1.000 | 0 |
| 32×2 | `js_pooled` | 41.0 / 46.0 / 47.5 | 44.8 ± 2.0 | −0.5 ± 2.4 | 0.85 | 0.56 | 0–1 |
| 32×2 | `js_fixed` | 43.5 / 42.0 / 41.5 | 42.3 ± 0.6 | −3.0 ± 0.3 | 0.009 | 0.000 | 150 |
| 32×2 | `global` | 44.0 / 48.5 / 43.0 | 45.2 ± 1.7 | −0.2 ± 1.6 | 0.93 | 0.000 | 150 |

`shrink` is the run mean of the implied shrinkage factor (1 = plain GRPO,
0 = global mean); "collapsed" counts steps at exactly 0, out of 150.
Degenerate groups: 25–28% of groups per step at N = 8, 68–70% at N = 2.

**At K = 8, N = 8 nothing is resolved.** Every arm is within 1.2 points of
`vanilla`, every s.e.m. is larger than the difference, every p is about 0.6.
The per-seed columns show why: the same arm moves by 3–5 points between
seeds. Seed 0 alone would have said `global` beats `vanilla` by 4.0 points
(McNemar p = 0.039, 10 wins / 2 losses); seeds 1 and 2 put it at +2.0 and
−2.5. Anyone reporting the seed-0 number is reporting noise.

**At K = 32, N = 2 the one "significant" row is the noise floor, measured.**
`js_fixed` at −3.0 ± 0.3 (p = 0.009) looks like a finding. But at this shape
`js_fixed` is algebraically `global` ([The algebra](#the-algebra)):
identical advantages on every batch, verified bit-for-bit at step 1, the two
processes diverging from step 2 because sampled generation is not
reproducible on GPU. Final accuracy, `global` minus `js_fixed`:

| seed | `global` | `js_fixed` | difference |
|---|---|---|---|
| 0 | 44.0% | 43.5% | +0.5 |
| 1 | 48.5% | 42.0% | +6.5 |
| 2 | 43.0% | 41.5% | +1.5 |

Same algorithm, three seeds, same prompts and test set: 0.5 to 6.5 points
apart. Pooling the two arms as what they are — six runs of one algorithm —
gives −1.6 ± 1.0 against `vanilla`, and the p-value goes away. A df = 2
paired t-test with three differences that happen to land within 0.5 of each
other produces p < 0.01 from nothing; that is the trap this row is kept in
the table to illustrate. An earlier two-run estimate of the same floor under
the 1.1 sampler was 3.0 points (48.5% vs 45.5%, same arm, same seed).

**The effect under test is 0.65–1.5 points** (Table 2 of arXiv:2511.03710,
Qwen2.5-0.5B-Instruct on GSM8K, JS minus GRPO at N = 8 / 4 / 2, K = 64
prompts per step, 500 steps, 5 seeds). The noise floor is 2–5x the effect.
Resolving a 1-point effect at this spread needs on the order of 20–40 seeds
per arm.

Two more things the panel settles. First, `js_fixed` is the specification
verbatim, and at N = 2 — the regime the specification is motivated by — it
does not merely over-shrink, it discards the group structure on 100% of steps
(at N = 8 on 25–30%). `js_pooled`, the specification with its σ² ≈ 1
precondition honoured, keeps 56% of the structure at N = 2 and 85% at N = 8,
which is the mild shrinkage Stein's result actually asks for. Second, the
"more stable training" claim finds no support at this scale: no arm clipped
a single gradient step under the 1.0 sampler (the 1.1-sampler clip counts
below were an off-policy artefact), sampled-token entropy falls from 0.52 to
0.26–0.33 for every arm alike, and the per-step gradient noise/signal ratio
(paper eq. 17–18, `--track-grad-var`) is 58–182 with 32–45% of steps
returning a signal estimate ≤ 0 — i.e. at K = 8 or 32 prompts per step the
gradient is essentially all noise for *every* baseline, and the ratio cannot
separate them. `compare` reports those NaN steps as right-censored (+∞) with
their share in brackets rather than dropping them, which would bias the
median down.

![Figure 3, K=8 N=8, seed 0](results/fig3_training.png)
![Figure 3b, K=32 N=2, seed 0](results/fig3_training_k32n2.png)

## Phase 4 — seed 0, five arms, 1.1 sampler (superseded)

The first sweep, kept for the record. All five arms sampled with the 1.1
repetition penalty, so the stability differences below (clip counts, entropy)
are off-policy artefacts that vanish under the corrected sampler above.

| arm | eval@0 | eval@final | Δ | p | shrink | collapsed |
|---|---|---|---|---|---|---|
| `vanilla` | 42.5% | 45.5% | +3.0 | 0.24 | 1.000 | 0/150 |
| `js_pooled` | 42.5% | 45.0% | +2.5 | 0.27 | 0.850 | 1/150 |
| `js_loo` | 42.5% | 45.5% | +3.0 | 0.21 | 0.818 | 0/150 |
| `js_fixed` | 42.5% | 47.0% | +4.5 | 0.11 | 0.202 | 64/150 |
| `global` | 42.5% | 47.5% | +5.0 | 0.06 | 0.025 | 133/150 |

`shrink` for `global` reads 0.025 rather than its definitional 0 because the
diagnostic then counted groups with `x_k = x̄` as unshrunk; the other arms
carry the same few-hundredths inflation.

Nothing here is significant: no arm beats its own starting point at p<0.05,
none differs from `vanilla` (all p ≥ 0.54), and the 2.5-point spread across
five arms is below the floor a single arm shows against itself. `global` has
the largest nominal gain, which read without the p column says "discard the
group structure and GRPO improves". What the sweep does establish is
structural: the shrinkage column separates into tiers that accuracy cannot
resolve —

```
vanilla 1.000  ≫  js_pooled 0.850, js_loo 0.818  ≫  js_fixed 0.202  ≫  global 0.025
```

— and `js_pooled` and `js_loo`, derived independently, land within 0.03 of
each other. Under this sampler `global` also hit the gradient clip on 36 of
150 steps against 9 for `vanilla` and its entropy fell from 2.56 to 0.95
against 1.27; both differences disappear under the 1.0 sampler.

## Conformance with GRPO.md

`tests/test_spec_conformance.py` transcribes section 3.2's ten steps literally
and asserts `js_fixed` agrees to 1e-12 on both continuous and 0/1 rewards, and
that `shrunk_group_means` honours the `(K,) -> mu_shrunk` contract of 3.1.

Two departures, both asserted, plus the pseudocode's own crashes, also
asserted (the 3.3 listing is transcribed verbatim in the test file and run):

* **Step 5 contradicts the pseudocode.** The prose says K<4 returns "the
  original means" (the per-group means X); the 3.3 pseudocode returns the
  *global* mean instead — a different estimator (it is the `global` arm, no
  longer GRPO), and one that raises `IndexError`, since `X_bar` is 0-d and
  `unsqueeze(1)` is out of range for it. We follow the prose.
* **Step 7 is unreachable in the pseudocode.** "S = 0 -> shrink_factor = 1"
  is stated, but `max(S, eps)` is Python's `max`, which hands back the float
  `eps` whenever `S < eps`; `torch.clamp` then receives a float and raises
  `TypeError`. So the listing crashes on exactly the case step 7 claims to
  handle (every group mean equal — it happens at K=8 with 0/1 rewards). We
  clamp to 0 instead, which is observationally identical: S = 0 implies
  diff = 0, so mu = X_bar either way.
* **Section 2's precondition is not in the listing.** The theory assumes
  rewards rescaled to σ² ≈ 1; the code takes the raw matrix. That is the
  over-shrinkage measured above, and `js_pooled` is the listing with the
  rescaling put back. A per-group std is unusable because 30–70% of groups
  have std 0, hence the pooled one.

### The std-normalisation variant is a footgun

Line 110 offers "divide by the within-group std afterwards" as an option. Under
plain GRPO that is safe on a degenerate group only because the advantage is
exactly 0 there, so `0/(0+1e-4) = 0`. **Every shrinkage baseline gives such a
group a non-zero advantage by design** — that is the point of the method — so
the same expression multiplies it by 1e4. Measured on GSM8K at K=8, N=8 with
38% degenerate groups: `adv_var` 1.07e5, `grad_norm` 1.7e3.

`scale="group"` therefore leaves zero-spread groups unscaled. With the guard:
`adv_var` 0.57, `grad_norm` 1.2.

## Related work

* **Zeng, Zhou, Arora, Zanette — [arXiv:2511.03710](https://arxiv.org/abs/2511.03710).**
  The same idea with its preconditions handled: both the within-group and
  the between-prompt variance are estimated from the batch, both levels are
  leave-one-out so the gradient stays unbiased, no hyper-parameters. That is
  `js_loo` here, checked against the PDF (Sec. 3.3, eq. 10, 13–16). They
  report 0.65–1.5 points over GRPO on this model and task at K = 64, 500
  steps, 5 seeds, and 11–67% lower gradient variance.
* **Han et al. — EBPO, [arXiv:2602.05165](https://arxiv.org/abs/2602.05165).**
  Empirical Bayes rather than batch James–Stein: shrinks toward a running
  global prior (Welford), shrinkage set by the within/between variance
  ratio, sold on AIME / OlympiadBench. Its stated feature — an all-zero
  group still receives a non-zero signal — is the same mechanism this
  project files under bias (a degenerate group is pushed in the direction
  the batch happens to lean). Cited, not implemented.
* **Efron & Morris 1975, batting averages.** The classical James–Stein on
  binomial proportions, which is exactly the 0/1-reward case. Their remedy
  for σ² ≠ 1 is an arcsine-square-root variance-stabilising transform
  before shrinking; this project pools the within-group std instead. The
  transform is the more orthodox route and is untried here.
* **Dr.GRPO, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783).** Why
  `scale=none` is the default: the within-group std term is a difficulty
  bias, and it interacts with shrinkage as the footgun above.

## Tests

```bash
python3 -m pytest tests/ -q
```

162 tests: shape/finiteness contracts, K<4 and S=0 and N=1 edge cases, the
closed-form leave-one-out algebra against a naive O(K²) loop, the verl adapter
under row shuffling, GSM8K answer extraction, spec conformance (the ten steps
and the verbatim 3.3 listing, including its two crashes, and `js_pooled` ≡
spec after pooled-std rescaling), the claims above stated as assertions, CPU
random-projection regression checks, the pair estimator and the per-step
micro-batch estimator on simulated gradients, the completion mask on both
Qwen terminators, and the sampler default.

## Status

- [x] Phase 0 — GPU env: torch 2.13.0+cu129, sm_120 gate passed, 14.5 GiB VRAM free
- [x] Phase 1 — estimators, tests, synthetic validation (CPU only); tables regenerated 2026-09-04 with the corrected shrink diagnostic
- [x] Phase 2 — GRPO loop, Qwen2.5-0.5B-Instruct + GSM8K rule-based reward
- [ ] Phase 3 — projection bug corrected, pair estimator added and tested on simulated gradients; **not rerun on the model** (out of the time budget; `python main.py gradvar --batches 200` is the command, ~2 h per shape)
- [x] Phase 4 — 5-arm training sweep, seed 0, 1.1 sampler (superseded)
- [x] Phase 4b — the spec's own regime: K·N = 64 at N = 2 and N = 8, seeds 0–2, 1.0 sampler, gradient variance logged (2026-09-04, AutoDL, 2× RTX 5090, ~35 min per arm). No resolvable accuracy effect at either shape; `js_fixed` ≡ `global` at N = 2 gives a same-algorithm replicate pair 0.5–6.5 points apart. `rloo` and `js_loo` not included.
- [ ] Phase 5 — verl PR (`to_verl_style` adapter is in place; nothing submitted)
