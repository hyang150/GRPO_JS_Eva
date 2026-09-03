"""Figure 2: gradient noise per baseline, with bootstrap intervals.

A forest plot, because the honest content here is an effect size and its
uncertainty -- a bar chart of point estimates would hide the thing that
matters most.
"""

import json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd6"

# same hue per entity as figure 1
COLOR = {"vanilla": "#2a78d6", "rloo": "#eb6834", "global": "#1baf7a",
         "js_fixed": "#eda100", "js_pooled": "#e87ba4", "js_loo": "#008300"}
LABEL = {"vanilla": "GRPO (group mean)", "rloo": "RLOO",
         "global": "global mean", "js_fixed": "JS, $V{=}1/N$ (GRPO.md)",
         "js_pooled": "JS, $V$ estimated", "js_loo": "JS two-level LOO"}
# only leave-one-out baselines are unbiased for the same gradient, so only
# those may be compared on variance at face value
UNBIASED = {"rloo", "js_loo"}
ROWS = ["vanilla", "rloo", "js_loo", "js_pooled", "js_fixed", "global"]


def main():
    d = json.loads((ROOT / "results/grad_variance_k8n8_long.json").read_text())
    s, nb = d["summary"], d["summary"]["vanilla"]["n_batches"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=0.20, right=0.975, top=0.80, bottom=0.16, wspace=0.42)

    # ---- A: effect size vs GRPO, with CI ----------------------------------
    ax = axes[0]
    ax.axvspan(-67.1, -11.2, color="#008300", alpha=0.07, zorder=1)
    ax.annotate("range reported by\narXiv:2511.03710", xy=(-39, 5.75),
                ha="center", va="bottom", color="#0a6b3d", fontsize=8, style="italic")
    ax.axvline(0, color=INK2, lw=1.1, zorder=2)

    for y, key in enumerate(ROWS):
        pt, lo, hi = s[key]["rel_noise_vs_vanilla"]
        c = COLOR[key]
        solid = key in UNBIASED or key == "vanilla"
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot([lo, hi], [y, y], "|", color=c, ms=9, mew=2.2, zorder=3)
        ax.plot([pt], [y], "o", ms=9, color=c if solid else SURFACE,
                mec=c, mew=2.2, zorder=4)
        sig = "" if (lo < 0 < hi or key == "vanilla") else "  ✓"
        ax.annotate(f"{pt:+.1f}%{sig}", xy=(hi, y), xytext=(8, 0),
                    textcoords="offset points", va="center", color=c,
                    fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([LABEL[k] for k in ROWS], fontsize=9.5, color=INK)
    ax.set_ylim(-0.7, len(ROWS) - 0.15)
    ax.invert_yaxis()
    ax.set_xlim(-100, 45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("change in gradient noise/signal vs GRPO   (95% bootstrap CI)",
                  color=INK2, fontsize=10)
    ax.set_title("A.  Effect size", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=8)
    ax.annotate("← less noise", xy=(0.02, 0.04), xycoords="axes fraction",
                color=INK2, fontsize=8.5, style="italic")

    # ---- B: absolute noise/signal, split by whether the comparison is valid
    ax = axes[1]
    for y, key in enumerate(ROWS):
        v, (lo, hi) = s[key]["noise_ratio"], s[key]["noise_ratio_ci"]
        c = COLOR[key]
        solid = key in UNBIASED or key == "vanilla"
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=3)
        ax.plot([v], [y], "o", ms=9, color=c if solid else SURFACE,
                mec=c, mew=2.2, zorder=4)
        ax.annotate(f"{v:.1f}", xy=(v, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=c, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([])
    ax.set_ylim(-0.7, len(ROWS) - 0.15)
    ax.invert_yaxis()
    ax.set_xscale("log")
    hi_max = max(s[k]["noise_ratio_ci"][1] for k in ROWS)
    lo_min = min(s[k]["noise_ratio_ci"][0] for k in ROWS)
    ax.set_xlim(lo_min * 0.8, hi_max * 1.25)     # don't clip the intervals
    ax.set_xlabel(r"$\mathrm{tr\,Cov}(g)\;/\;\|\mathbb{E}[g]\|^2$",
                  color=INK2, fontsize=10)
    ax.set_title("B.  Absolute noise/signal", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=8)

    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(True, axis="x", color=GRID, lw=0.6, zorder=0)
        a.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9, length=3)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, mfc=INK2, mec=INK2,
                          label="unbiased for the same gradient — comparable"),
               plt.Line2D([], [], marker="o", ls="", ms=9, mfc=SURFACE, mec=INK2,
                          mew=2.2, label="biased baseline — targets a different $\\mathbb{E}[g]$")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.55, 0.002), labelcolor=INK2)

    fig.suptitle(f"Paired gradient-noise measurement — Qwen2.5-0.5B-Instruct on GSM8K, "
                 f"K=8 x N=8, {nb} batches, frozen policy",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.955)

    out = ROOT / "results/fig2_grad_variance.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
