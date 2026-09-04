"""Measure the policy-gradient variance each baseline produces.

The paper's headline claim is 11.2-67.1% lower gradient variance.  Testing it
by training separate runs and comparing curves is the wrong instrument: run-to-
run noise at this scale is larger than the effect.

Instead this is a *paired* measurement.  One set of rollouts is generated per
batch and every baseline scores the *same* rewards, so the rollout randomness
cancels exactly and the only thing that differs is the baseline.  The policy is
frozen throughout -- we are measuring the estimator, not a training outcome.

The quantity reported is the trace of the gradient covariance,

    tr Cov(g) = E[||g||^2] - ||E[g]||^2

and the noise/signal ratio tr Cov(g) / ||E[g]||^2, from two estimators:

* Hutchinson: tr Cov from the across-batch variance of <g, u_m> over M fixed
  Gaussian probes; ||E[g]||^2 is then E||g||^2 (exact) minus that.  Bootstrap
  intervals resample batches, conditional on the probe set.  The weak point:
  ||E[g]||^2 is under a tenth of E||g||^2 here, so a few percent of probe
  error on tr Cov is a large error on the ratio, and no number of batches
  reduces it.  ``trace_cov_probe_se`` reports that error.
* Disjoint pairs: for independent batches b != b', E<g_b, g_b'> = ||E[g]||^2
  exactly, no probes.  Batches are paired (0,1), (2,3), ...; each pair yields
  one unbiased sample of ||E[g]||^2 and one of E||g||^2, and the bootstrap
  resamples pairs.  Costs one CPU copy of the gradient per arm per pair.
"""

import argparse, json, pathlib, random, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_eva.baselines import BASELINE_REGISTRY, compute_advantages
from grpo_eva.data import load_gsm8k
from grpo_eva.grpo import GRPO, GRPOConfig
from grpo_eva.rewards import correctness_reward

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORDER = ["vanilla", "rloo", "global", "js_fixed", "js_pooled", "js_loo"]


N_PROBES = 24   # Hutchinson rel. error ~ sqrt(2/M); 8 probes is ~50%
PROBE_SEED = 20260903
PROBE_SCHEME = "gaussian_stream_v2"


def _pct_censored(x, q):
    """Percentiles of a 1-D sample that may contain +inf (right-censored
    values).  np.percentile interpolates and returns nan between two infs, so
    take order statistics instead: no interpolation, inf stays inf."""
    import numpy as np
    r = np.sort(np.asarray(x, dtype=float))
    idx = np.clip(np.round(np.asarray(q, dtype=float) / 100 * (len(r) - 1)).astype(int),
                  0, len(r) - 1)
    return r[idx]


def project(grads, m: int) -> float:
    """<g, u_m> for a fixed standard-normal probe u_m.

    Each probe is regenerated from its seed on the fly. All gradients must
    use the same device and parameter ordering across batches and arms.
    With u ~ N(0, I),
    E_u[Var_b(<g_b, u>)] = tr Cov(g) and E_u[(E_b<g_b,u>)^2] = ||E[g]||^2,
    so averaging over probes estimates both -- and because each batch now
    yields a *scalar*, the whole thing bootstraps.
    """
    if not grads:
        return 0.0
    gen = torch.Generator(device=grads[0].device).manual_seed(PROBE_SEED + m * 7919)
    acc = 0.0
    for g in grads:
        # Advance one stream so equal-shaped parameters get distinct entries.
        u = torch.randn(g.shape, generator=gen, device=g.device, dtype=torch.float32)
        acc += float((g.detach().float() * u).sum())
    return acc


class GradAccumulator:
    """Per-batch probe projections, squared norms and pair inner products
    for one baseline."""

    def __init__(self, params):
        self.z = []          # [n_batches][N_PROBES]
        self.sq_norms = []   # [n_batches]
        self.dots = []       # <g_{2j}, g_{2j+1}>, one per completed pair
        self._prev = None    # CPU copy of g_{2j} while waiting for g_{2j+1}

    def add(self, grads):
        self.z.append([project(grads, m) for m in range(N_PROBES)])
        self.sq_norms.append(sum(float((g.detach().float() ** 2).sum())
                                 for g in grads))
        if self._prev is None:
            # bf16 grads stay bf16 on the CPU copy: nothing is lost
            self._prev = [g.detach().to("cpu", copy=True) for g in grads]
        else:
            self.dots.append(sum(
                float(torch.dot(g.detach().float().flatten(),
                                q.to(g.device).float().flatten()))
                for g, q in zip(grads, self._prev)))
            self._prev = None

    def summary(self, n_boot: int = 2000, seed: int = 0):
        import numpy as np
        z = np.asarray(self.z)                       # (n, M)
        n = z.shape[0]
        rng = np.random.default_rng(seed)

        sq = np.asarray(self.sq_norms)

        def stats(idx):
            zz = z[idx]
            tr = zz.var(axis=0, ddof=1).mean()       # tr Cov(g), Hutchinson
            # ||E[g]||^2 = E[||g||^2] - tr Cov(g), and E[||g||^2] is known
            # exactly from the stored per-batch norms -- no probe noise there.
            sig = max(sq[idx].mean() - tr, 1e-30)
            return tr, sig, tr / sig

        point = stats(np.arange(n))
        boot = np.array([stats(rng.integers(0, n, n)) for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        m_probes = z.shape[1]
        out = {
            # raw per-batch probe projections, so any pairwise comparison can
            # be re-bootstrapped later without re-running the measurement
            "z": z.tolist(), "sq_norms": list(self.sq_norms),
            "trace_cov": point[0], "trace_cov_ci": [lo[0], hi[0]],
            "signal_sq": point[1], "signal_sq_ci": [lo[1], hi[1]],
            "noise_ratio": point[2], "noise_ratio_ci": [lo[2], hi[2]],
            "E_sq_norm": float(sum(self.sq_norms) / n),
            "n_batches": n, "boot": boot.tolist(),
            # Each probe column's variance is its own unbiased estimate of
            # tr Cov, so the spread across the M columns is the standard error
            # of their mean -- the part of the error the batch bootstrap
            # cannot see.  Compare it with signal_sq.
            "trace_cov_probe_se": (float(z.var(axis=0, ddof=1).std(ddof=1) / np.sqrt(m_probes))
                                   if m_probes > 1 else None),
        }
        out.update(self._pair_summary(sq, n_boot, seed))
        return out

    def _pair_summary(self, sq, n_boot: int, seed: int):
        """Probe-free estimates from disjoint consecutive batch pairs."""
        import numpy as np
        dots = np.asarray(self.dots)
        n_pairs = len(dots)
        if n_pairs < 2:
            return {"dots": dots.tolist(), "n_pairs": n_pairs}
        sq_pair = 0.5 * (sq[0:2 * n_pairs:2] + sq[1:2 * n_pairs:2])   # E||g||^2 per pair
        # same seed in every arm -> same resample -> paired across arms
        rng = np.random.default_rng(seed)

        def stats(idx):
            sig = dots[idx].mean()          # ||E[g]||^2, unbiased
            tr = sq_pair[idx].mean() - sig  # tr Cov(g)
            return tr, sig, (tr / sig if sig > 0 else np.nan)

        point = stats(np.arange(n_pairs))
        boot = np.array([stats(rng.integers(0, n_pairs, n_pairs)) for _ in range(n_boot)])
        # A resample whose signal estimate is <= 0 has an undefined (arbitrarily
        # large) ratio.  Dropping it would censor the noisiest resamples and pull
        # the upper bound down, so it enters the percentile as +inf: the upper
        # bound becomes inf whenever more than 2.5% of resamples are undefined.
        boot_cens = boot.copy()
        boot_cens[~np.isfinite(boot_cens[:, 2]), 2] = np.inf
        lo, hi = np.percentile(boot_cens[:, :2], [2.5, 97.5], axis=0)
        lo, hi = (np.append(lo, _pct_censored(boot_cens[:, 2], 2.5)),
                  np.append(hi, _pct_censored(boot_cens[:, 2], 97.5)))
        return {
            "dots": dots.tolist(), "n_pairs": n_pairs,
            "pair_trace_cov": point[0], "pair_trace_cov_ci": [lo[0], hi[0]],
            "pair_signal_sq": point[1], "pair_signal_sq_ci": [lo[1], hi[1]],
            "pair_noise_ratio": point[2], "pair_noise_ratio_ci": [lo[2], hi[2]],
            # resamples where the signal estimate came out <= 0
            "pair_undefined_frac": float(np.mean(~np.isfinite(boot[:, 2]))),
            # censored copy: undefined ratios are +inf, so the paired
            # "ratio vs GRPO" below inherits the same right-censoring
            "pair_boot": boot_cens.tolist(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()

    cfg = GRPOConfig(k_prompts=args.k, n_generations=args.n,
                     max_new_tokens=args.max_new_tokens,
                     micro_batch=args.micro_batch,
                     gen_micro_batch=args.k * args.n, seed=args.seed)
    trainer = GRPO(model, tok, cfg)
    data = load_gsm8k(tok, "train")
    rng = random.Random(args.seed)

    params = [p for p in model.parameters() if p.requires_grad]
    accs = {b: GradAccumulator(params) for b in ORDER}
    rows, t0 = [], time.time()

    print(f"paired gradient-variance measurement: {args.batches} batches, "
          f"K={args.k} N={args.n}, policy frozen\n")

    for it in range(args.batches):
        batch = rng.sample(data, args.k)
        model.eval()
        seq, attn, plen, cmask, texts = trainer.generate([b["prompt"] for b in batch])
        golds = [b["gold"] for b in batch for _ in range(args.n)]
        rewards = torch.tensor(correctness_reward(texts, golds),
                               device=trainer.device).view(args.k, args.n)

        model.train()
        denom = cmask.shape[0] * args.max_new_tokens
        stat = {"batch": it, "accuracy": rewards.mean().item()}

        for name in ORDER:                      # same rollouts for every arm
            adv = compute_advantages(rewards, baseline=name, scale="none")
            adv_flat = adv.reshape(-1)
            model.zero_grad(set_to_none=True)
            for i in range(0, seq.shape[0], args.micro_batch):
                sl = slice(i, i + args.micro_batch)
                lp = trainer._logprobs(seq[sl], attn[sl], plen, grad=True)
                loss = -(lp * adv_flat[sl].unsqueeze(1) * cmask[sl]).sum() / denom
                loss.backward()
            accs[name].add([p.grad for p in params])
            stat[f"{name}_advvar"] = adv.var().item()

        model.zero_grad(set_to_none=True)
        rows.append(stat)
        print(f"[{it:3d}] acc={stat['accuracy']:.3f}  "
              f"{(time.time() - t0) / (it + 1):.0f}s/batch", flush=True)

    import numpy as np
    out = {n: accs[n].summary() for n in ORDER}
    base = np.asarray(out["vanilla"]["boot"])        # paired bootstrap vs GRPO

    print(f"\n{'baseline':<11}{'tr Cov(g)':>22}{'noise/signal':>24}"
          f"{'ratio vs GRPO':>22}")
    for name in ORDER:
        s = out[name]
        b = np.asarray(s["boot"])
        rel = (b[:, 2] / base[:, 2] - 1) * 100       # same resample -> paired
        rlo, rhi = np.percentile(rel, [2.5, 97.5])
        s["rel_noise_vs_vanilla"] = [float((s["noise_ratio"] /
                                            out["vanilla"]["noise_ratio"] - 1) * 100),
                                     float(rlo), float(rhi)]
        s.pop("boot")
        print(f"{name:<11}"
              f"{s['trace_cov']:>10.3e} [{s['trace_cov_ci'][0]:.2e},{s['trace_cov_ci'][1]:.2e}]"
              f"{s['noise_ratio']:>10.1f} [{s['noise_ratio_ci'][0]:6.1f},{s['noise_ratio_ci'][1]:6.1f}]"
              f"{s['rel_noise_vs_vanilla'][0]:>+9.1f}% [{rlo:+6.1f},{rhi:+6.1f}]")
    print(f"\nprobe SE of tr Cov(g) vs signal_sq (the ratio is only meaningful "
          f"when the first is well below the second):")
    for name in ORDER:
        s = out[name]
        print(f"  {name:<11} probe_se {s['trace_cov_probe_se']:.3e}   signal_sq {s['signal_sq']:.3e}")

    # ---- pair estimator: exact inner products, no probes -----------------
    if out["vanilla"].get("pair_boot"):
        pbase = np.asarray(out["vanilla"]["pair_boot"])
        n_pairs = out["vanilla"]["n_pairs"]
        print(f"\npair estimator, {n_pairs} disjoint batch pairs, no probes")
        print(f"{'baseline':<11}{'tr Cov(g)':>22}{'||E g||^2':>22}{'noise/signal':>24}"
              f"{'ratio vs GRPO':>22}")
        for name in ORDER:
            s = out[name]
            pb = np.asarray(s.pop("pair_boot"))
            # ratio column is +inf where the signal estimate was <= 0:
            # arm undefined / GRPO finite -> +inf (right-censored, kept);
            # arm finite / GRPO undefined -> -100% (kept);
            # both undefined -> nan, genuinely unordered, dropped and counted
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = (pb[:, 2] / pbase[:, 2] - 1) * 100
            ok = ~np.isnan(rel)
            rlo, rhi = (_pct_censored(rel[ok], [2.5, 97.5]) if ok.any()
                        else (float("nan"), float("nan")))
            pt = (s["pair_noise_ratio"] / out["vanilla"]["pair_noise_ratio"] - 1) * 100
            s["pair_rel_noise_vs_vanilla"] = [float(pt), float(rlo), float(rhi)]
            s["pair_rel_undefined_frac"] = float(1 - ok.mean())
            print(f"{name:<11}"
                  f"{s['pair_trace_cov']:>10.3e} [{s['pair_trace_cov_ci'][0]:.2e},{s['pair_trace_cov_ci'][1]:.2e}]"
                  f"{s['pair_signal_sq']:>10.3e} [{s['pair_signal_sq_ci'][0]:.2e},{s['pair_signal_sq_ci'][1]:.2e}]"
                  f"{s['pair_noise_ratio']:>10.1f} [{s['pair_noise_ratio_ci'][0]:6.1f},{s['pair_noise_ratio_ci'][1]:6.1f}]"
                  f"{pt:>+9.1f}% [{rlo:+6.1f},{rhi:+6.1f}]")
    else:
        for name in ORDER:
            out[name].pop("pair_boot", None)

    p = pathlib.Path(args.out or ROOT / f"results/grad_variance_k{args.k}n{args.n}.json")
    config = {**vars(args), "probe_scheme": PROBE_SCHEME,
              "probe_seed": PROBE_SEED, "n_probes": N_PROBES,
              "ci_scope": "batches_conditional_on_fixed_probes",
              "pair_estimator": "disjoint_consecutive_pairs_v1",
              "pair_ci_scope": "pairs"}
    p.write_text(json.dumps({"config": config, "summary": out, "batches": rows},
                            indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
