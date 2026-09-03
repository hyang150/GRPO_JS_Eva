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

accumulated over batches with a running mean of g held on the CPU.
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


def project(grads, m: int) -> float:
    """<g, u_m> for a fixed standard-normal probe u_m.

    Storing 8 probes over 0.5B parameters would cost 16 GB, so each probe is
    regenerated from its seed on the fly.  With u ~ N(0, I),
    E_u[Var_b(<g_b, u>)] = tr Cov(g) and E_u[(E_b<g_b,u>)^2] = ||E[g]||^2,
    so averaging over probes estimates both -- and because each batch now
    yields a *scalar*, the whole thing bootstraps.
    """
    acc = 0.0
    for g in grads:
        gen = torch.Generator(device=g.device).manual_seed(
            PROBE_SEED + m * 7919 + g.numel())
        u = torch.randn(g.shape, generator=gen, device=g.device, dtype=torch.float32)
        acc += float((g.detach().float() * u).sum())
    return acc


class GradAccumulator:
    """Per-batch probe projections and squared norms for one baseline."""

    def __init__(self, params):
        self.z = []          # [n_batches][N_PROBES]
        self.sq_norms = []   # [n_batches]

    def add(self, grads):
        self.z.append([project(grads, m) for m in range(N_PROBES)])
        self.sq_norms.append(sum(float((g.detach().float() ** 2).sum())
                                 for g in grads))

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
        return {
            # raw per-batch probe projections, so any pairwise comparison can
            # be re-bootstrapped later without re-running the measurement
            "z": z.tolist(), "sq_norms": list(self.sq_norms),
            "trace_cov": point[0], "trace_cov_ci": [lo[0], hi[0]],
            "signal_sq": point[1], "signal_sq_ci": [lo[1], hi[1]],
            "noise_ratio": point[2], "noise_ratio_ci": [lo[2], hi[2]],
            "E_sq_norm": float(sum(self.sq_norms) / n),
            "n_batches": n, "boot": boot.tolist(),
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

    p = pathlib.Path(args.out or ROOT / f"results/grad_variance_k{args.k}n{args.n}.json")
    p.write_text(json.dumps({"config": vars(args), "summary": out, "batches": rows},
                            indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
