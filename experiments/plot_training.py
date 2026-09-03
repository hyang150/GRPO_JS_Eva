"""Figure 3: training curves, one line per baseline arm.

Reads every results/train_<arm>_*.jsonl the sweep produced.  Works with a
single arm too.
"""

import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd6"

COLOR = {"vanilla": "#2a78d6", "rloo": "#eb6834", "global": "#1baf7a",
         "js_fixed": "#eda100", "js_pooled": "#e87ba4", "js_loo": "#008300"}
LABEL = {"vanilla": "GRPO (group mean)", "rloo": "RLOO", "global": "global mean",
         "js_fixed": "JS, $V{=}1/N$ (GRPO.md)", "js_pooled": "JS, $V$ estimated",
         "js_loo": "JS two-level LOO"}


def smooth(xs, w, kind="mean"):
    """Centred rolling window.

    ``kind="median"`` for spiky series: gradient norm sits at ~0.5 but spikes
    to 7.5 a handful of times, and a rolling *mean* turns each spike into a
    flat plateau the width of the window -- an artefact that reads as if
    training stalled.
    """
    import statistics
    f = statistics.median if kind == "median" else statistics.fmean
    h, n = w // 2, len(xs)

    def window(i):
        vals = [v for v in xs[max(0, i - h):min(n, i + h + 1)] if v == v]  # drop NaN
        return f(vals) if vals else float("nan")

    return [window(i) for i in range(n)]


def load(path):
    steps, evals, cfg = [], [], {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        {"config": lambda: cfg.update(r), "step": lambda: steps.append(r),
         "eval": lambda: evals.append(r)}.get(r.get("record"), lambda: None)()
    return cfg, steps, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="train_*_k*n*_s*.jsonl")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default="results/fig3_training.png")
    ap.add_argument("--seed", type=int, default=None,
                    help="which seed's arms to draw (default: the lowest present)")
    args = ap.parse_args()

    # one figure = one seed.  Keyed by baseline alone, a second seed's file
    # would silently replace the first's for that arm and the panel would mix
    # prompt streams.
    by_seed = {}
    for p in sorted((ROOT / "results").glob(args.glob)):
        cfg, steps, evals = load(p)
        if steps:
            by_seed.setdefault(cfg.get("seed", 0), {})[cfg.get("baseline", p.stem)] = (cfg, steps, evals)
    if not by_seed:
        sys.exit(f"no runs matched results/{args.glob}")
    seed = min(by_seed) if args.seed is None else args.seed
    if seed not in by_seed:
        sys.exit(f"no runs for seed {seed}; have {sorted(by_seed)}")
    runs = {arm: (steps, evals) for arm, (_, steps, evals) in by_seed[seed].items()}
    cfg0 = next(iter(by_seed[seed].values()))[0]
    if len(by_seed) > 1:
        print(f"seeds present: {sorted(by_seed)}; drawing seed {seed}")
    print("arms:", ", ".join(runs))

    panels = [
        ("accuracy", "reward (train, $T{=}1$)", True, "mean", False),
        ("degenerate_frac", "degenerate groups\n(no GRPO signal)", True, "mean", False),
        ("shrink", "implied shrinkage factor", False, "mean", False),
        ("grad_norm", "gradient norm", False, "median", True),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0),
                             facecolor=SURFACE)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.22, wspace=0.30)

    for ax, (key, ylab, pct, kind, logy) in zip(axes, panels):
        for arm, (steps, _) in runs.items():
            xs = [s["step"] for s in steps]
            raw = [s[key] for s in steps]
            c = COLOR.get(arm, INK2)
            if len(set(raw)) > 1:            # show what was actually measured
                ax.plot(xs, raw, color=c, lw=0.7, alpha=0.28, zorder=2)
            ax.plot(xs, smooth(raw, args.window, kind), color=c, lw=2, zorder=3)
        if logy:
            ax.set_yscale("log")
            ax.axhline(1.0, color=INK2, lw=0.9, ls=":", zorder=2)
            ax.annotate("clip threshold", xy=(0.97, 1.0), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="right",
                        color=INK2, fontsize=8, style="italic")
        if pct:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
        ax.set_xlabel("step", color=INK2, fontsize=9.5)

    # eval points ride on the reward panel, marked as a different measurement
    ax = axes[0]
    for arm, (_, evals) in runs.items():
        if not evals:
            continue
        c = COLOR.get(arm, INK2)
        ax.errorbar([e["step"] for e in evals], [e["accuracy"] for e in evals],
                    yerr=[e["stderr"] for e in evals], fmt="o--", color=c,
                    ms=6, lw=1.2, mfc=SURFACE, mew=1.8, capsize=3, zorder=4)
    ax.annotate("hollow: greedy eval on the\ntest split (200 problems)",
                xy=(0.03, 0.04), xycoords="axes fraction", color=INK2,
                fontsize=8, style="italic")

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=3)

    handles = [plt.Line2D([], [], color=COLOR.get(a, INK2), lw=2,
                          label=LABEL.get(a, a)) for a in runs]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(runs), 6),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01),
               labelcolor=INK2)

    fig.suptitle(f"GRPO on GSM8K — Qwen2.5-0.5B-Instruct, K={cfg0.get('k', 8)} x "
                 f"N={cfg0.get('n', 8)}, lr={cfg0.get('lr', 1e-6):g}, seed {seed}, "
                 f"faint = per step, bold = rolling window of {args.window}",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.055, ha="left", y=0.955)

    out = ROOT / args.out
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
