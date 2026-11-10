"""Two figures the robustness work makes necessary.

fig3b is REPLACED rather than added to: it currently draws only the behaviour-bootstrap
interval, which after Task B is known to cover roughly 60% of the uncertainty. A figure that
shows one source and omits a comparable one asserts a precision the Limitations section now
disclaims, so the grader interval is drawn behind it.

fig5 is new: phi for every pair under the three label/threshold regimes, which is the visual
form of the robustness claim - every pair stays positive everywhere, while individual values
move enough that none should be read to two decimals.

House style matches scripts/make_figures.py: serif, >=8pt body / >=7pt ticks, Okabe-Ito,
series distinguished by shape as well as colour, no titles, vector PDF with the values beside
it as CSV.

Usage:
  python scripts/23_robustness_figures.py --out paper/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})
BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#999999"
FULL_W = 7.0

PRETTY = {"ppl_filter": "perplexity", "token_anomaly": "token anomaly", "probe": "probe L16",
          "probe_b": "probe L8", "refusal_prime": "refusal prime", "smoothllm": "SmoothLLM"}
CANON = {"perplexity": "ppl_filter", "token-anomaly": "token_anomaly",
         "probe16": "probe", "probe8": "probe_b", "refusal-prime": "refusal_prime",
         "smoothllm": "smoothllm"}


def key(a, b):
    return frozenset((a, b))


def label(k):
    a, b = sorted(k)
    return f"{PRETTY[a]} × {PRETTY[b]}"


def load():
    single, grader, majority, external = {}, {}, {}, {}
    for r in csv.DictReader(open("results/hpc_vicuna_autodan/table6.csv")):
        if "llamaguard" in (r["d1"], r["d2"]):
            continue
        single[key(r["d1"], r["d2"])] = (float(r["phi"]), float(r["phi_lo"]),
                                         float(r["phi_hi"]))
    for r in csv.DictReader(open("fusion/taskB/B4_grader_intervals.csv")):
        a, b = [CANON[x.strip()] for x in r["pair"].split(" x ")]
        grader[key(a, b)] = (float(r["phi_grader_lo"]), float(r["phi_grader_hi"]))
    for r in csv.DictReader(open("fusion/taskB/B3_table10_majority.csv")):
        a, b = [CANON[x.strip()] for x in r["pair"].split(" x ")]
        majority[key(a, b)] = float(r["phi"])
    for r in csv.DictReader(open("fusion/taskD/D3_table10_external.csv")):
        a, b = [CANON[x.strip()] for x in r["pair"].split(" x ")]
        external[key(a, b)] = float(r["phi"])
    return single, grader, majority, external


def fig_forest(single, grader, out: Path):
    """fig3b, with the grader interval drawn behind the behaviour bootstrap."""
    order = sorted(single, key=lambda k: single[k][0])
    h = 0.30 * len(order) + 1.1
    fig, ax = plt.subplots(figsize=(FULL_W, h))
    for i, k in enumerate(order):
        phi, lo, hi = single[k]
        if k in grader:
            g_lo, g_hi = grader[k]
            ax.plot([g_lo, g_hi], [i, i], color=ORANGE, lw=3.2, alpha=0.55,
                    solid_capstyle="butt", zorder=1)
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.1, solid_capstyle="butt", zorder=2)
        ax.plot([phi], [i], marker="o", ms=4.5, color=BLUE, markerfacecolor="white",
                markeredgecolor=BLUE, markeredgewidth=0.9, zorder=3)
    ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(k) for k in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(r"failure correlation $\phi$")
    ax.yaxis.grid(True, color="0.90", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=1.1, marker="o", ms=4.5,
               markerfacecolor="white", label="95% behaviour bootstrap"),
        Line2D([0], [0], color=ORANGE, lw=3.2, alpha=0.55, label="95% grader-only interval"),
    ], loc="lower right", frameon=False, handletextpad=0.5, borderpad=0.3)
    fig.savefig(out / "fig3b_forest_phi.pdf")
    plt.close(fig)
    with open(out / "fig3b_forest_phi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "phi", "boot_lo", "boot_hi", "grader_lo", "grader_hi"])
        for k in order:
            g = grader.get(k, ("", ""))
            w.writerow([label(k), f"{single[k][0]:.3f}", f"{single[k][1]:.3f}",
                        f"{single[k][2]:.3f}",
                        f"{g[0]:.3f}" if g[0] != "" else "",
                        f"{g[1]:.3f}" if g[1] != "" else ""])


def fig_regimes(single, majority, external, out: Path):
    """fig5: phi under the three regimes, one row per pair."""
    order = sorted(single, key=lambda k: single[k][0])
    h = 0.30 * len(order) + 1.1
    fig, ax = plt.subplots(figsize=(FULL_W, h))
    for i, k in enumerate(order):
        xs = [single[k][0], majority.get(k), external.get(k)]
        live = [x for x in xs if x is not None]
        if len(live) > 1:
            ax.plot([min(live), max(live)], [i, i], color="0.80", lw=0.8, zorder=1)
        ax.plot([single[k][0]], [i], marker="o", ms=4.6, color=BLUE,
                markerfacecolor="white", markeredgewidth=0.9, zorder=3)
        if k in majority:
            ax.plot([majority[k]], [i], marker="s", ms=4.2, color=VERM,
                    markerfacecolor=VERM, zorder=3)
        if k in external:
            ax.plot([external[k]], [i], marker="^", ms=4.8, color=GREEN,
                    markerfacecolor="white", markeredgewidth=0.9, zorder=3)
    ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(k) for k in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"failure correlation $\phi$")
    ax.yaxis.grid(True, color="0.90", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Placed OUTSIDE the axes: with three regimes the points spread across the full width,
    # so every in-axes corner collides with data on some row.
    fig.legend(handles=[
        Line2D([0], [0], ls="", marker="o", ms=4.6, color=BLUE, markerfacecolor="white",
               label="single judgment, in-sample (as reported)"),
        Line2D([0], [0], ls="", marker="s", ms=4.2, color=VERM, label="majority of 5 judgments"),
        Line2D([0], [0], ls="", marker="^", ms=4.8, color=GREEN, markerfacecolor="white",
               label="external thresholds"),
    ], loc="outside lower center", ncol=3, frameon=False, handletextpad=0.5,
        columnspacing=2.0)
    fig.savefig(out / "fig5_regimes_phi.pdf")
    plt.close(fig)
    with open(out / "fig5_regimes_phi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "phi_single", "phi_majority", "phi_external"])
        for k in order:
            w.writerow([label(k), f"{single[k][0]:.3f}",
                        f"{majority[k]:.3f}" if k in majority else "",
                        f"{external[k]:.3f}" if k in external else ""])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    single, grader, majority, external = load()
    print(f"pairs: single {len(single)}, grader {len(grader)}, "
          f"majority {len(majority)}, external {len(external)}")
    fig_forest(single, grader, a.out)
    fig_regimes(single, majority, external, a.out)
    lo = min(min(single[k][0] for k in single),
             min(majority.values()), min(external.values()))
    print(f"minimum phi across all three regimes: {lo:.3f} "
          f"({'all positive' if lo > 0 else 'SOME NON-POSITIVE'})")
    print(f"wrote fig3b_forest_phi.pdf (now with grader intervals) and "
          f"fig5_regimes_phi.pdf to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
