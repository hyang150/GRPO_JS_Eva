"""Figure 1: does shrinkage lower the baseline's MSE, and which variant?

Reads results/synthetic_mse.json (written by synthetic_mse.py).
"""

import json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]

# validated categorical palette (dataviz skill, light surface) -- fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd6"
SERIES = {
    "vanilla":   ("#2a78d6", "-",  "GRPO (group mean)"),
    "rloo":      ("#eb6834", "--", "RLOO (leave-one-out)"),
    "global":    ("#1baf7a", "-",  "global mean"),
    "js_fixed":  ("#eda100", "--", "JS, V=1/N  (GRPO.md as written)"),
    "js_pooled": ("#e87ba4", "-",  "JS, V estimated"),
    "js_loo":    ("#008300", "-",  "JS two-level LOO (arXiv:2511.03710)"),
}


def main():
    rows = json.loads((ROOT / "results/synthetic_mse.json").read_text())
    rows = sorted(rows, key=lambda r: r["N"])
    ns = [r["N"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.20, wspace=0.26)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels(ns)
        ax.set_xlabel("generations per prompt  $N$", color=INK2, fontsize=10)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=3)

    # -- A: absolute MSE -----------------------------------------------------
    ax = axes[0]
    for key, (c, ls, _) in SERIES.items():
        ax.plot(ns, [r[key]["mse"] for r in rows], ls, color=c, lw=2,
                marker="o", ms=5.5, mec=SURFACE, mew=1.2, zorder=3)
    ax.set_ylabel(r"MSE of the baseline vs true $p_k$", color=INK2, fontsize=10)
    ax.set_title("A.  Estimation error", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=8)
    ax.annotate("lower is better", xy=(0.03, 0.06), xycoords="axes fraction",
                color=INK2, fontsize=8.5, style="italic")
    ax.annotate("JS with $V{=}1/N$ collapses\nonto the global mean",
                xy=(4, rows[1]["js_fixed"]["mse"]), xytext=(12, 34),
                textcoords="offset points", ha="left", color="#a87400",
                fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#a87400", lw=1.0))

    # -- B: relative to vanilla ---------------------------------------------
    # Clipped: global/js_fixed run to +356%, which would squash the -16..-54%
    # range that carries the finding.  Off-scale points get an arrow + value.
    ax = axes[1]
    LO, HI = -62, 62
    ax.axhspan(0, HI, color="#e34948", alpha=0.05, zorder=1)
    ax.axhline(0, color=INK2, lw=1.1, zorder=2)
    ax.annotate("worse than GRPO", xy=(0.03, 0.93), xycoords="axes fraction",
                ha="left", color="#b03a39", fontsize=8.5, style="italic")

    OFFSCALE_DY = {"global": -14, "js_fixed": -27}  # inside the axes, clear of the title
    for key, (c, ls, _) in SERIES.items():
        rel = [(r[key]["mse"] / r["vanilla"]["mse"] - 1) * 100 for r in rows]
        vis = [v if v <= HI else float("nan") for v in rel]
        ax.plot(ns, vis, ls, color=c, lw=2, marker="o", ms=5.5,
                mec=SURFACE, mew=1.2, zorder=3)
        for n, v in zip(ns, rel):
            if v > HI:                      # off-scale: mark and label it
                ax.plot([n], [HI - 3], marker="^", ms=7, color=c,
                        mec=SURFACE, mew=1.0, clip_on=False, zorder=4)
                ax.annotate(f"+{v:.0f}%", xy=(n, HI - 3),
                            xytext=(0, OFFSCALE_DY.get(key, 8)),
                            textcoords="offset points", ha="center",
                            color=c, fontsize=8, fontweight="bold", zorder=5)
    ax.set_ylim(LO, HI)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_ylabel("MSE relative to GRPO", color=INK2, fontsize=10)
    ax.set_title("B.  Change vs plain GRPO", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=8)

    # one joint direct label: the two correct variants sit on top of each other
    rel_p = [(r["js_pooled"]["mse"] / r["vanilla"]["mse"] - 1) * 100 for r in rows]
    ax.annotate("JS with $V$ estimated\n(both variants)", xy=(ns[1], rel_p[1]),
                xytext=(6, -30), textcoords="offset points", ha="left",
                color="#a8446a", fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#a8446a", lw=0.9))

    # -- C: implied shrinkage ------------------------------------------------
    ax = axes[2]
    for y, lab in ((1.0, "no shrinkage  =  GRPO"), (0.0, "full collapse  =  global mean")):
        ax.axhline(y, color=INK2, lw=0.9, ls=":", zorder=2)
        ax.annotate(lab, xy=(2.05, y), xytext=(0, 4 if y == 0 else -12),
                    textcoords="offset points", color=INK2, fontsize=8, style="italic")
    for key in ("js_fixed", "js_pooled", "js_loo"):
        c, ls, _ = SERIES[key]
        ax.plot(ns, [r[key]["shrink"] for r in rows], ls, color=c, lw=2,
                marker="o", ms=5.5, mec=SURFACE, mew=1.2, zorder=3)
    ax.set_ylim(-0.12, 1.12)
    ax.set_ylabel("implied shrinkage factor", color=INK2, fontsize=10)
    ax.set_title("C.  Why  —  how hard each variant shrinks", color=INK,
                 fontsize=11.5, fontweight="bold", loc="left", pad=8)

    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, marker="o", ms=5.5,
                          mec=SURFACE, mew=1.2, label=lab)
               for c, ls, lab in SERIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.5, 0.005), columnspacing=1.6,
               handlelength=2.2, labelcolor=INK2)

    k = rows[0]["K"]
    fig.suptitle(f"Shrinkage baselines for GRPO — Bernoulli rewards, K={k} prompts, "
                 "2000 trials, $p_k\\sim$Beta(1.2, 2.2)",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.055, ha="left", y=0.955)

    out = ROOT / "results/fig1_synthetic_mse.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
