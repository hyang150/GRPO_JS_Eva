"""Compare sweep arms on the same 200 test problems.

Every arm evaluates the identical problem set in the identical order, so the
comparison is paired and McNemar applies.  That matters here: at n=200 an
unpaired difference of 6 points is ~1.2 sigma and says nothing, while the
paired test only spends power on the problems the two arms disagree about.
"""

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from grpo_eva.evaluate import mcnemar

ROOT = pathlib.Path(__file__).resolve().parents[1]


def nanmean(xs):
    xs = [x for x in xs if x == x]          # drop NaN (shrink undefined on a step)
    return sum(xs) / len(xs) if xs else float("nan")


def load(path):
    cfg, steps, evals = {}, [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("record") == "config":
            cfg = r
        elif r.get("record") == "step":
            steps.append(r)
        elif r.get("record") == "eval":
            evals.append(r)
    return cfg, steps, evals


def multi_seed(by_seed, arms, seeds, reference):
    """Across-seed aggregate, paired by seed.

    Runs sharing a seed walk the same prompt stream, so the per-seed
    difference against the reference arm removes the run-to-run term that
    dominates a single comparison.  With n seeds the 3.0-point noise floor a
    single arm shows against itself falls as 3.0/sqrt(n) -- which is the whole
    reason to spend money on more seeds rather than more steps.
    """
    import statistics as st

    print(f"across {len(seeds)} seeds: {seeds}\n")
    print(f"{'arm':<12}{'final acc (mean+-sem)':>24}{'vs ' + reference:>22}{'p (paired t)':>14}")

    ref = {s: by_seed[(reference, s)][2][-1]["accuracy"]
           for s in seeds if (reference, s) in by_seed}

    for arm in arms:
        finals = {s: by_seed[(arm, s)][2][-1]["accuracy"]
                  for s in seeds if (arm, s) in by_seed}
        if len(finals) < 2:
            print(f"{arm:<12}{'(needs >= 2 seeds)':>24}")
            continue
        vals = list(finals.values())
        mean = st.fmean(vals)
        sem = st.stdev(vals) / len(vals) ** 0.5
        line = f"{arm:<12}{mean:>17.1%} +-{sem:>5.1%}"

        common = [s for s in seeds if s in finals and s in ref]
        if arm == reference or len(common) < 2:
            print(line)
            continue
        d = [finals[s] - ref[s] for s in common]
        dm, ds = st.fmean(d), st.stdev(d)
        if ds == 0:
            print(line + f"{dm:>+20.1%}{'--':>14}")
            continue
        t = dm / (ds / len(d) ** 0.5)
        p = _t_sf(abs(t), len(d) - 1) * 2
        print(line + f"{dm:>+16.1%} +-{ds / len(d) ** 0.5:>4.1%}{p:>14.4f}")

    print(f"\nnote: {len(seeds)} seeds put the run-to-run floor near "
          f"{3.0 / len(seeds) ** 0.5:.1f} points; the effect under test is 0.65-1.5 "
          f"(arXiv:2511.03710 Table 2, K=64, 500 steps, 5 seeds).")


def _t_sf(t, df):
    """Upper tail of Student's t, via the regularised incomplete beta."""
    from math import lgamma, exp, log

    if t <= 0:                 # x would be 1 and log(1-x) undefined
        return 0.5
    x = df / (df + t * t)

    def betacf(a, b, x, it=200):
        tiny = 1e-30
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, it):
            m2 = 2 * m
            for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                        -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
                d = 1.0 + num * d
                d = 1.0 / (d if abs(d) > tiny else tiny)
                c = 1.0 + num / (c if abs(c) > tiny else tiny)
                h *= d * c
            if abs(d * c - 1.0) < 3e-12:
                break
        return h

    lbeta = lgamma(df / 2) + lgamma(0.5) - lgamma(df / 2 + 0.5)
    front = exp(df / 2 * log(x) + 0.5 * log(1 - x) - lbeta)
    ib = front * betacf(df / 2, 0.5, x) / (df / 2) if x > 0 else 0.0
    return 0.5 * min(1.0, ib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="vanilla")
    ap.add_argument("--glob", default="train_*_k*n*_s*.jsonl")
    args = ap.parse_args()

    by_seed = {}          # {(baseline, seed): (cfg, steps, evals)}
    shapes = set()
    for p in sorted((ROOT / "results").glob(args.glob)):
        cfg, steps, evals = load(p)
        if not steps:
            continue
        shapes.add((cfg.get("k"), cfg.get("n")))
        want = cfg.get("steps")
        # a run still in progress has an eval@0 but no final eval; reading
        # evals[-1] as its result would score the untrained policy
        if want is not None and not (evals and evals[-1].get("step") == want):
            print(f"skip {p.name}: {len(steps)}/{want} steps, no final eval yet")
            continue
        by_seed[(cfg.get("baseline", p.stem), cfg.get("seed", 0))] = (cfg, steps, evals)
    if not by_seed:
        sys.exit(f"no runs matched results/{args.glob}")

    if len(shapes) > 1:
        # runs are keyed on (baseline, seed) only, so two group shapes would
        # silently overwrite each other above
        sys.exit(f"results/{args.glob} mixes group shapes (K,N)={sorted(shapes)}; "
                 f"narrow it, e.g. --glob 'train_*_k8n8_s*.jsonl'")

    seeds = sorted({s for _, s in by_seed})
    arms = sorted({a for a, _ in by_seed})
    if args.reference not in arms:
        sys.exit(f"reference arm {args.reference!r} not among {arms}")

    if len(seeds) > 1:
        multi_seed(by_seed, arms, seeds, args.reference)
        print()

    # single-seed view uses the lowest seed, so the per-step detail below is
    # always one concrete run rather than a blend of several
    runs = {a: by_seed[(a, seeds[0])] for a in arms if (a, seeds[0]) in by_seed}

    import statistics as st

    print(f"{'arm':<11}{'eval@0':>9}{'eval@final':>12}{'delta':>8}"
          f"{'reward(last 30)':>17}{'degen':>8}{'shrink':>9}")
    for arm, (_, steps, evals) in runs.items():
        tail = steps[-30:]
        e0 = evals[0]["accuracy"] if evals else float("nan")
        ef = evals[-1]["accuracy"] if evals else float("nan")
        print(f"{arm:<11}{e0:>8.1%}{ef:>12.1%}{ef - e0:>+8.1%}"
              f"{st.fmean(s['accuracy'] for s in tail):>17.3f}"
              f"{st.fmean(s['degenerate_frac'] for s in tail):>8.2f}"
              f"{nanmean(s['shrink'] for s in tail):>9.3f}")

    # ---- within an arm: did it actually improve? -------------------------
    print(f"\npaired McNemar, each arm's final eval vs its own eval@0")
    print(f"{'arm':<11}{'fixed':>7}{'broken':>8}{'delta':>8}{'p':>9}")
    for arm, (_, _, evals) in runs.items():
        if len(evals) < 2 or "per_problem" not in evals[0]:
            print(f"{arm:<11}{'per-problem outcomes not recorded':>32}")
            continue
        m = mcnemar(evals[0]["per_problem"], evals[-1]["per_problem"])
        print(f"{arm:<11}{m['fixed']:>7}{m['broken']:>8}{m['delta']:>+8.1%}"
              f"{m['p_value']:>9.4f}")

    # ---- between arms: is any arm better than the reference? -------------
    ref = runs[args.reference][2]
    if len(ref) >= 1 and "per_problem" in ref[-1]:
        print(f"\npaired McNemar, final eval vs {args.reference} final")
        print(f"{'arm':<11}{'wins':>7}{'losses':>8}{'delta':>8}{'p':>9}")
        for arm, (_, _, evals) in runs.items():
            if arm == args.reference or "per_problem" not in evals[-1]:
                continue
            m = mcnemar(ref[-1]["per_problem"], evals[-1]["per_problem"])
            # against a reference arm the two counts read as wins / losses
            print(f"{arm:<11}{m['fixed']:>7}{m['broken']:>8}"
                  f"{m['delta']:>+8.1%}{m['p_value']:>9.4f}")

    print("\nnote: 200 problems resolves a paired difference of roughly 5-6 points.\n"
          "A p above 0.05 here means 'not resolved at this scale', not 'no effect'.")

    stability(runs, args.reference, seeds[0])


def stability(runs, reference, seed):
    """What GRPO.md's "训练更稳、收敛更快" would have to show up as.

    Accuracy cannot resolve the effect (see the noise floor), but the
    training log carries direct stability signals that every arm records on
    the same prompt stream:

      clipped    fraction of steps with grad_norm above the clip threshold
      entropy    mean -logprob of sampled tokens, start -> last 10 steps
                 (fast collapse = the policy sharpening, not learning)
      adv_var    mean advantage variance -- the quantity shrinkage acts on
      reward sd  sd of the per-step training reward over the last 30 steps
      d reward   per-step reward minus the reference arm's, last 30 steps;
                 same seed -> same prompts, so prompt difficulty cancels
      noise/sig  median per-step gradient noise/signal, arXiv:2511.03710
                 eq. 17-18 across micro-batches (needs --track-grad-var).
                 The signal estimate ||g_bar||^2 - Var is unbiased but can
                 come out <= 0 when noise dominates; such steps are logged
                 as NaN.  Dropping them would censor exactly the noisiest
                 steps and bias the median down, so they enter as +inf
                 (right-censored) and their share is printed alongside.
    """
    import statistics as st

    print(f"\nstability, seed {seed}")
    print(f"{'arm':<11}{'clipped':>9}{'entropy':>13}{'adv_var':>9}{'reward sd':>11}"
          f"{'d reward vs ' + reference:>22}{'noise/sig':>17}")
    ref_steps = runs[reference][1]
    for arm, (_, steps, _) in runs.items():
        gn = [s["grad_norm"] for s in steps]
        clipped = sum(g > 1.0 for g in gn) / len(gn)          # GRPOConfig.grad_clip
        ent = (f"{steps[0]['entropy_proxy']:.2f}->"
               f"{nanmean(s['entropy_proxy'] for s in steps[-10:]):.2f}")
        advv = nanmean(s["adv_var"] for s in steps)
        tail = steps[-30:]
        rsd = st.pstdev([s["accuracy"] for s in tail]) if len(tail) > 1 else float("nan")
        n = min(len(steps), len(ref_steps))
        d = [steps[i]["accuracy"] - ref_steps[i]["accuracy"] for i in range(n)][-30:]
        dm = st.fmean(d) if d else float("nan")
        dse = st.pstdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
        nsr = [s["grad_noise_ratio"] for s in steps if "grad_noise_ratio" in s]
        if nsr:
            censored = sum(v != v for v in nsr) / len(nsr)         # NaN: signal <= 0
            med = st.median(float("inf") if v != v else v for v in nsr)
            nsr_txt = (f"{med:.1f}" if med != float("inf") else "inf") + f" ({censored:.0%})"
        else:
            nsr_txt = "n/a"
        print(f"{arm:<11}{clipped:>9.1%}{ent:>13}{advv:>9.3f}{rsd:>11.3f}"
              f"{dm:>+14.3f} +-{dse:.3f}{nsr_txt:>17}")
    print("note: 'clipped' counts grad_norm > 1.0; 'noise/sig' is logged only with "
          "--track-grad-var, (x%) = share of steps whose signal estimate was <= 0, "
          "counted as +inf.\nNone of these is a test of accuracy; they are the "
          "stability claims the spec makes, measured.")


if __name__ == "__main__":
    main()
