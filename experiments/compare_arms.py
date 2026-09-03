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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="vanilla")
    ap.add_argument("--glob", default="train_*_k*n*_s*.jsonl")
    args = ap.parse_args()

    runs = {}
    for p in sorted((ROOT / "results").glob(args.glob)):
        cfg, steps, evals = load(p)
        if steps:
            runs[cfg.get("baseline", p.stem)] = (cfg, steps, evals)
    if args.reference not in runs:
        sys.exit(f"reference arm {args.reference!r} not among {sorted(runs)}")

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
              f"{st.fmean(s['shrink'] for s in tail):>9.3f}")

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


if __name__ == "__main__":
    main()
