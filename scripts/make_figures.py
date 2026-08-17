"""Build the three results figures as separate vector PDFs, each with its values as CSV.

Deliberately NOT a re-plot of Table 6. Each figure answers one question the table cannot:
  fig3a  does the independence prediction hold?          (predicted vs observed joint)
  fig3b  which pairs are distinguishable from zero?      (forest of phi with CIs)
  fig4   does the attack class change the verdict?       (per-defense residual, GCG vs fluent)

House style: serif, >=8 pt body / >=7 pt ticks, Okabe-Ito colourblind-safe palette, series
distinguished by marker SHAPE as well as colour, no titles (LaTeX supplies captions), no
internal run names in any artwork, no gridlines except a light y-grid on the forest.

Usage:
  python scripts/make_figures.py --fluent results/hpc_vicuna_autodan \
      --suffix results/hpc_vicuna --out paper/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- house style -----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,          # embed TrueType, keep text selectable/vector
    "ps.fonttype": 42,
    "savefig.bbox": None,          # keep the declared figure width exactly
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
})

# Okabe-Ito
BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#999999"

SINGLE_COL, FULL_W = 3.4, 7.0     # inches

PRETTY = {
    "ppl_filter": "perplexity",
    "token_anomaly": "token anomaly",
    "llamaguard": "Llama Guard",
    "refusal_prime": "refusal prime",
    "smoothllm": "SmoothLLM",
    "probe": "probe L16",
    "probe_b": "probe L8",
    "undefended": "undefended",
    "stack": "stack",
}


def pretty_pair(a: str, b: str) -> str:
    return f"{PRETTY.get(a, a)} × {PRETTY.get(b, b)}"


def read_pairs(run: Path, drop: set[str]) -> list[dict]:
    """Pairs as plotted. `drop` removes defenses whose marginal is ~0, whose phi is
    uninformative; this keeps the figures showing exactly the pairs Table 6 reports."""
    rows = []
    with open(run / "table6.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["d1"] in drop or r["d2"] in drop:
                continue
            rows.append({
                "pair": pretty_pair(r["d1"], r["d2"]),
                "same_row": int(r["same_row"]) == 1,
                "p1": float(r["p1"]), "p2": float(r["p2"]),
                "indep": float(r["indep"]), "joint": float(r["joint"]),
                "excess": float(r["excess"]),
                "phi": float(r["phi"]), "lo": float(r["phi_lo"]), "hi": float(r["phi_hi"]),
            })
    return rows


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


# --- fig 3a: predicted vs observed ------------------------------------------------------
def fig_predicted_vs_observed(pairs: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    lim = 0.65
    ax.plot([0, lim], [0, lim], color=GREY, lw=0.8, zorder=1)
    ax.annotate("independence", xy=(0.46, 0.46), xytext=(0.50, 0.40), rotation=45,
                color=GREY, fontsize=7, ha="center", va="center", rotation_mode="anchor")

    cross = [r for r in pairs if not r["same_row"]]
    same = [r for r in pairs if r["same_row"]]
    ax.scatter([r["indep"] for r in cross], [r["joint"] for r in cross],
               marker="o", s=22, facecolor="none", edgecolor=BLUE, linewidth=0.9,
               label="cross row", zorder=3)
    ax.scatter([r["indep"] for r in same], [r["joint"] for r in same],
               marker="s", s=26, facecolor=VERM, edgecolor=VERM, linewidth=0.9,
               label="same row", zorder=4)

    # Annotate ONLY the same-row pairs, at fixed anchors chosen in empty regions of the
    # panel (upper-left and lower-centre). Fixed anchors rather than point-relative offsets
    # keep the labels from colliding with the legend or the point cloud.
    anchors = [(0.045, 0.620, "left"), (0.315, 0.245, "left")]
    for r, (tx, ty, ha) in zip(sorted(same, key=lambda x: -x["joint"]), anchors):
        ax.annotate(r["pair"], xy=(r["indep"], r["joint"]), xytext=(tx, ty),
                    fontsize=7, ha=ha, va="center", color=VERM,
                    arrowprops=dict(arrowstyle="-", color=VERM, lw=0.5, shrinkA=1, shrinkB=3))

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"independence prediction $\hat p_1 \hat p_2$")
    ax.set_ylabel("observed joint breach rate")
    ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderpad=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / "fig3a_predicted_vs_observed.pdf")
    plt.close(fig)
    write_csv(out / "fig3a_predicted_vs_observed.csv", pairs,
              ["pair", "same_row", "p1", "p2", "indep", "joint", "excess"])


# --- fig 3b: forest of phi --------------------------------------------------------------
def fig_forest(pairs: list[dict], out: Path) -> None:
    same = sorted([r for r in pairs if r["same_row"]], key=lambda r: r["phi"])
    cross = sorted([r for r in pairs if not r["same_row"]], key=lambda r: r["phi"])
    ordered = cross + same                      # bottom-up: cross first, same-row at top
    h = 0.30 * len(ordered) + 0.9
    fig, ax = plt.subplots(figsize=(FULL_W, h))

    for i, r in enumerate(ordered):
        c, m = (VERM, "s") if r["same_row"] else (BLUE, "o")
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=1.1, solid_capstyle="butt", zorder=2)
        ax.plot([r["phi"]], [i], marker=m, ms=4.5, color=c,
                markerfacecolor=c if r["same_row"] else "white",
                markeredgecolor=c, markeredgewidth=0.9, zorder=3)

    ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=1)
    ax.annotate("independence", xy=(0, len(ordered) - 0.35), xytext=(0.012, len(ordered) - 0.35),
                fontsize=7, color="0.35", va="center", ha="left")

    if same:
        ax.axhline(len(cross) - 0.5, color="0.75", lw=0.6)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([r["pair"] for r in ordered])
    ax.set_ylim(-0.6, len(ordered) - 0.05)
    ax.set_xlabel("failure correlation $\phi$  (95% bootstrap CI)")
    ax.yaxis.grid(True, color="0.90", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color=VERM, ls="-", ms=4.5, lw=1.1, label="same row"),
        Line2D([0], [0], marker="o", color=BLUE, ls="-", ms=4.5, lw=1.1,
               markerfacecolor="white", label="cross row"),
    ], loc="lower right", frameon=False, handletextpad=0.5, borderpad=0.2)

    fig.savefig(out / "fig3b_forest_phi.pdf")
    plt.close(fig)
    write_csv(out / "fig3b_forest_phi.csv", ordered[::-1],
              ["pair", "same_row", "phi", "lo", "hi"])


# --- fig 4: attack class dumbbell -------------------------------------------------------
def fig_attack_class(fluent: dict, suffix: dict, out: Path) -> None:
    defs = [d for d in fluent if d not in ("undefended", "stack") and d in suffix]
    rows = [{"defense": PRETTY.get(d, d), "asr_suffix_attack": suffix[d],
             "asr_fluent_attack": fluent[d]} for d in defs]
    rows.sort(key=lambda r: r["asr_fluent_attack"])

    h = 0.34 * len(rows) + 1.0
    fig, ax = plt.subplots(figsize=(SINGLE_COL, h))
    for i, r in enumerate(rows):
        ax.plot([r["asr_suffix_attack"], r["asr_fluent_attack"]], [i, i],
                color="0.70", lw=1.0, zorder=1, solid_capstyle="round")
        ax.plot([r["asr_suffix_attack"]], [i], marker="o", ms=4.5, color=BLUE,
                markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.0, zorder=3)
        ax.plot([r["asr_fluent_attack"]], [i], marker="^", ms=5, color=VERM,
                markerfacecolor=VERM, markeredgecolor=VERM, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["defense"] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.02, 0.75)
    ax.set_xlabel("residual attack success rate")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=BLUE, ls="", ms=4.5, markerfacecolor="white",
               markeredgewidth=1.0, label="suffix attack"),
        Line2D([0], [0], marker="^", color=VERM, ls="", ms=5, label="fluent attack"),
    ], loc="lower right", frameon=False, handletextpad=0.5, borderpad=0.2)

    fig.savefig(out / "fig4_attack_class.pdf")
    plt.close(fig)
    write_csv(out / "fig4_attack_class.csv", rows,
              ["defense", "asr_suffix_attack", "asr_fluent_attack"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fluent", required=True, help="run dir for the fluent-attack results")
    ap.add_argument("--suffix", required=True, help="run dir for the suffix-attack results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--drop-pairs-with", nargs="*", default=["llamaguard"],
                    help="defenses excluded from the PAIR figures (near-zero marginal -> "
                         "uninformative phi); they remain in the attack-class figure")
    a = ap.parse_args()

    fl, sf, out = Path(a.fluent), Path(a.suffix), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(fl, set(a.drop_pairs_with))
    fig_predicted_vs_observed(pairs, out)
    fig_forest(pairs, out)

    m_fl = json.load(open(fl / "analysis.json"))["marginals_adaptive"]
    m_sf = json.load(open(sf / "analysis.json"))["marginals_adaptive"]
    fig_attack_class(m_fl, m_sf, out)

    print(f"wrote 3 PDFs + 3 CSVs to {out}  (pairs plotted: {len(pairs)})")


if __name__ == "__main__":
    main()
