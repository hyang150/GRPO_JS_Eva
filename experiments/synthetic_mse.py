"""Does shrinkage actually lower the MSE of the baseline?  Synthetic study.

Ground truth is known here, which is the whole point: on GSM8K we never
observe the true per-prompt pass rate p_k, so we can only measure MSE in
simulation.  Rewards are Bernoulli(p_k) with p_k ~ Beta(a, b) chosen to
mimic the spread of per-prompt difficulty a small model shows on GSM8K.

Reported per baseline:
  mse       mean squared error of the implied estimate of p_k   <- Stein's claim
  bias^2    squared bias of that estimate
  g2        E[(r - b)^2], a proxy for policy-gradient second moment
  shrink    mean implied shrinkage (1.0 = vanilla GRPO, 0.0 = global mean)
"""

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from grpo_eva.baselines import get_baseline_fn, shrink_factor

BASELINES = ["vanilla", "rloo", "global", "js_fixed", "js_pooled", "js_loo"]


def run(k, n, trials, alpha, beta, seed, device="cpu"):
    """Returns {baseline: {metric: value}} for one (K, N) setting."""
    g = torch.Generator(device=device).manual_seed(seed)
    dist = torch.distributions.Beta(alpha, beta)

    # (trials, K) true pass rates; (trials, K, N) binary rewards
    p = dist.sample((trials, k)).to(device)
    r = (torch.rand(trials, k, n, generator=g, device=device) < p.unsqueeze(-1)).double()
    p = p.double()

    out = {}
    for name in BASELINES:
        fn = get_baseline_fn(name)
        est = torch.stack([fn(r[t]).mean(dim=1) for t in range(trials)])   # (trials, K)
        b = torch.stack([fn(r[t]) for t in range(trials)])                 # (trials, K, N)

        err = est - p
        se = (err ** 2).mean(dim=1)                     # per-trial MSE
        shrink = torch.stack([shrink_factor(r[t], name).mean() for t in range(trials)])

        out[name] = {
            "mse": se.mean().item(),
            "mse_stderr": (se.std() / trials ** 0.5).item(),
            "bias2": (err.mean(dim=0) ** 2).mean().item(),
            "g2": ((r - b) ** 2).mean().item(),
            "shrink": shrink.mean().item(),
        }

    # how often does a group carry no GRPO signal at all?
    grp = r.sum(dim=2)
    out["_degenerate_frac"] = ((grp == 0) | (grp == n)).double().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[32])
    ap.add_argument("--n", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=1.2)   # Beta(1.2, 2.2):
    ap.add_argument("--beta", type=float, default=2.2)    # mean ~0.35, wide spread
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/synthetic_mse.json")
    args = ap.parse_args()

    rows = []
    for k in args.k:
        for n in args.n:
            res = run(k, n, args.trials, args.alpha, args.beta, args.seed)
            rows.append({"K": k, "N": n, **res})

            deg = res.pop("_degenerate_frac")
            base = res["vanilla"]["mse"]
            print(f"\n=== K={k}  N={n}   (退化组占比 {deg:6.1%}) ===")
            print(f"{'baseline':<11}{'MSE':>10}{'±stderr':>9}{'vs vanilla':>12}"
                  f"{'bias^2':>10}{'E[(r-b)^2]':>12}{'shrink':>8}")
            for name in BASELINES:
                m = res[name]
                rel = (m["mse"] / base - 1) * 100
                print(f"{name:<11}{m['mse']:>10.5f}{m['mse_stderr']:>9.5f}"
                      f"{rel:>+11.1f}%{m['bias2']:>10.5f}{m['g2']:>12.5f}"
                      f"{m['shrink']:>8.3f}")

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
