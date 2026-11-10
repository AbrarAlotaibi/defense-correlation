"""Stage 34: the figures for the v2 paper, drawn from results/<run>/analysis_v2.json.

Print figures for a journal, so there is no interaction layer and no dark mode; everything
else follows the project's visualisation rules. Colours are the validated categorical slots
(blue #2a78d6, orange #eb6834, aqua #1baf7a: all six checks pass on a light surface), used in
fixed order and never cycled. Every series also carries a distinct marker or hatch, so the
figures survive greyscale printing and the aqua's sub-3:1 contrast against the surface is
relieved by direct labels. One axis per plot, recessive grid and spines, thin marks.

  fig1_attack_class   residual attack success per safeguard under each attack family
  fig2_null_forest    phi per pair with intervals, against the undefended seed-pair null
  fig3_difficulty     observed phi against the phi that behaviour difficulty alone predicts
  fig4_estimands      prompt-level phi (E2) against behaviour-level phi (E1), per pair
  fig5_refusal        false refusal per safeguard on both benign sets

Usage:
  python scripts/34_figures_v2.py --run v2_vicuna
  python scripts/34_figures_v2.py --run v2_vicuna --demo    # synthetic data, to check layout
"""
from __future__ import annotations

import os
import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.9,
    "axes.axisbelow": True, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4,
})

SHORT = {"ppl_filter": "likelihood", "token_anomaly": "token-anomaly", "llamaguard": "guard model",
         "probe": "probe (mid)", "probe_b": "probe (early)", "refusal_prime": "refusal prompt",
         "smoothllm": "perturbation", "undefended": "undefended"}


def nm(d: str) -> str:
    return SHORT.get(d, d)


def pair_label(k: str) -> str:
    a, b = k.split("|")
    return f"{nm(a)} x {nm(b)}"


# Where the manuscript includes figures from. Copying on write, rather than by hand later,
# because a regenerated figure that never reaches the paper leaves the build succeeding with
# a stale PDF under the right filename, and nothing downstream can catch that.
MANUSCRIPT_FIGURES = Path(os.environ.get(
    "DCORR_MANUSCRIPT_FIGURES",
    Path(__file__).resolve().parents[1].parent / "defense-correlation-paper" / "paper_A_mlj",
))
# The journal build numbers its figure files in order of appearance. A figure that is not
# listed here is written to the results folder only.
MANUSCRIPT_NAMES = {"fig2_null_forest": "Fig1", "fig3_difficulty": "Fig2", "fig5_refusal": "Fig3"}


def _finish(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    target = MANUSCRIPT_NAMES.get(name)
    if target is None:
        print(f"  wrote {name}.pdf / .png  (not used in the manuscript)")
    elif MANUSCRIPT_FIGURES.is_dir():
        shutil.copy2(out / f"{name}.pdf", MANUSCRIPT_FIGURES / f"{target}.pdf")
        print(f"  wrote {name}.pdf / .png, copied the pdf to the manuscript as {target}.pdf")
    else:
        print(f"  wrote {name}.pdf / .png  (manuscript dir absent, copy by hand as {target}.pdf)")


# --------------------------------------------------------------------------- figures
def fig_attack_class(res: dict, out: Path):
    """Grouped horizontal bars: the same safeguard under two attack families."""
    per = res.get("attack_class")
    if not per:
        return
    defs = [d for d in per if d != "undefended"]
    defs.sort(key=lambda d: -max(per[d].values()))
    y = np.arange(len(defs))
    fams = list(next(iter(per.values())).keys())[:2]
    fig, ax = plt.subplots(figsize=(5.0, 0.42 * len(defs) + 1.1))
    for i, (fam, colour, hatch) in enumerate(zip(fams, (BLUE, ORANGE), ("", "///"))):
        v = [per[d][fam] for d in defs]
        ax.barh(y + (0.21 if i == 0 else -0.21), v, height=0.38, color=colour,
                edgecolor="white", linewidth=0.8, hatch=hatch, label=fam, zorder=3)
        for yy, vv in zip(y + (0.21 if i == 0 else -0.21), v):
            ax.text(vv + 0.012, yy, f"{vv:.2f}", va="center", ha="left",
                    fontsize=7, color=MUTED)
    ax.set_yticks(y, [nm(d) for d in defs])
    ax.set_xlabel("residual attack success rate")
    ax.set_xlim(0, 1.06)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.1), ncol=len(fams),
              mode="expand", borderaxespad=0)
    _finish(fig, out, "fig1_attack_class")


def fig_null_forest(res: dict, out: Path):
    """phi per pair with its interval, against the undefended seed-pair null."""
    e1 = res.get("e1") or {}
    if not e1:
        return
    items = sorted(e1.items(), key=lambda kv: kv[1]["mean_phi"] or 0)
    labels = [pair_label(k) for k, _ in items]
    mid = [v["mean_phi"] for _, v in items]
    lo, hi = [], []
    for _, v in items:
        cis = [s["ci"] for s in v["per_seed"].values() if s.get("ci")]
        lo.append(min(c[0] for c in cis) if cis else v["mean_phi"])
        hi.append(max(c[1] for c in cis) if cis else v["mean_phi"])
    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(items) + 1.3))
    null = (res.get("null") or {}).get("mean")
    if null is not None:
        ax.axvspan(0, null, color=MUTED, alpha=0.10, zorder=1)
        ax.axvline(null, color=MUTED, lw=1.1, ls="--", zorder=2)
        ax.text(null, -0.9, "  attack reproducibility (null)",
                fontsize=7, color=MUTED, va="center")
    ax.hlines(y, lo, hi, color=BLUE, lw=1.4, zorder=3)
    ax.plot(mid, y, "o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_ylim(-1.4, len(items) - 0.4)
    ax.set_xlabel(r"$\phi$ between safeguard failures (behavior level)")
    ax.yaxis.grid(False)
    _finish(fig, out, "fig2_null_forest")


def fig_difficulty(res: dict, out: Path):
    """Observed phi against the phi behaviour difficulty alone predicts."""
    d = res.get("difficulty") or {}
    pairs = d.get("pairs") or {}
    if not pairs:
        return
    obs = [v["observed"] for v in pairs.values()]
    pred = [v["predicted_difficulty_only"] for v in pairs.values()]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    lim = max(max(obs, default=0), max(pred, default=0), 0.1) * 1.15
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.text(lim * 0.97, lim * 0.97, "difficulty explains it all", fontsize=7, color=MUTED,
            ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    ax.plot(pred, obs, "o", ms=5, color=BLUE, mec="white", mew=0.8, zorder=3)
    ax.set_xlabel(r"$\phi$ predicted by behavior difficulty alone")
    ax.set_ylabel(r"observed $\phi$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    icc = (d.get("fit") or {}).get("icc")
    _finish(fig, out, "fig3_difficulty")


def fig_estimands(res: dict, out: Path):
    """Prompt-level phi (E2) against behaviour-level phi (E1), the paper's central distinction."""
    e1 = res.get("e1") or {}
    e2all = res.get("e2") or {}
    if not e1 or not e2all:
        return
    e2 = next(iter(e2all.values()))
    keys = [k for k in e1 if k in e2]
    if not keys:
        return
    x = [e1[k]["mean_phi"] for k in keys]
    y = [e2[k]["mean_phi"] for k in keys]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    lim = max(max(x, default=0), max(y, default=0), 0.1) * 1.15
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.plot(x, y, "s", ms=5, color=ORANGE, mec="white", mew=0.8, zorder=3)
    ax.set_xlabel(r"E1: behavior level, separate searches")
    ax.set_ylabel(r"E2: prompt level, shared input")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    _finish(fig, out, "fig4_estimands")


def fig_refusal(res: dict, out: Path):
    """False refusal per member on each benign set, with Wilson intervals.

    Only the seven members are drawn. The undefended target and the stacks are in the
    stored results too, and are reported in the manuscript's table, not here.
    """
    frr = res.get("frr") or {}
    if not frr:
        return
    first = next(iter(frr.values()))
    sets = [s for s in ("jbb", "orbench") if s in first]
    if not sets:
        return
    members = [d for d in first[sets[0]] if d != "undefended" and not d.startswith("stack")]
    defs = sorted(members, key=lambda d: -first[sets[0]][d]["frr"])
    y = np.arange(len(defs))
    fig, ax = plt.subplots(figsize=(5.0, 0.42 * len(defs) + 1.1))
    for i, (s, colour, marker) in enumerate(zip(sets, (BLUE, AQUA), ("o", "^"))):
        v = [first[s].get(d, {}).get("frr", math.nan) for d in defs]
        ci = [first[s].get(d, {}).get("ci", (math.nan, math.nan)) for d in defs]
        off = -0.16 if i == 0 else 0.16
        ax.hlines(y + off, [c[0] for c in ci], [c[1] for c in ci], color=colour, lw=1.3, zorder=3)
        ax.plot(v, y + off, marker, ms=4.5, color=colour, mec="white", mew=0.8, zorder=4,
                label={"jbb": "matched benign (n = 100)", "orbench": "OR-Bench (n = 400)"}[s])
    ax.set_yticks(y, [nm(d) for d in defs])
    ax.set_xlabel("false refusal rate")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()  # highest rate at the top, the order of the manuscript's table
    ax.yaxis.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.1), ncol=len(sets),
              mode="expand", borderaxespad=0)
    _finish(fig, out, "fig5_refusal")


# --------------------------------------------------------------------------- demo data
def demo() -> dict:
    rng = np.random.default_rng(0)
    defs = ["ppl_filter", "token_anomaly", "probe", "probe_b", "refusal_prime", "smoothllm"]
    pairs = [f"{a}|{b}" for i, a in enumerate(defs) for b in defs[i + 1:]]
    e1 = {}
    for k in pairs:
        m = float(rng.uniform(0.15, 0.70))
        e1[k] = {"mean_phi": m, "disattenuated": m * 1.2,
                 "per_seed": {s: {"phi": m, "ci": [m - 0.14, m + 0.14]} for s in (0, 1, 2)}}
    return {
        "run": "demo", "seeds": [0, 1, 2], "n_behaviours": 100,
        "null": {"available": True, "mean": 0.34},
        "e1": e1,
        "e2": {"0": {k: {"mean_phi": max(0.0, e1[k]["mean_phi"] - rng.uniform(0.0, 0.3))}
                     for k in pairs}},
        "difficulty": {"fit": {"icc": 0.41},
                       "pairs": {k: {"observed": e1[k]["mean_phi"],
                                     "predicted_difficulty_only": e1[k]["mean_phi"] * rng.uniform(0.5, 1.0),
                                     "excess": 0.0} for k in pairs}},
        "attack_class": {d: {"fluent search": float(rng.uniform(0.2, 0.9)),
                             "gradient suffix": float(rng.uniform(0.0, 0.6))} for d in defs},
        "frr": {"0": {s: {d: {"frr": float(rng.uniform(0.02, 0.5)), "n": 100,
                              "ci": [0.0, 0.0]} for d in defs} for s in ("jbb", "orbench")}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--demo", action="store_true", help="synthetic data, to check layout")
    a = ap.parse_args()

    run_dir = REPO / "results" / a.run
    out = run_dir / "figures"
    if a.demo:
        res = demo()
        out = REPO / "results" / "figure_demo"
    else:
        # Prefer the pinned resolved analysis, which is what the manuscript's tables and
        # scripts/39_verify_claims.py both read. analysis_v2.json is whatever stage 26 last
        # wrote, and stage 26 takes a --gold-name: reading it here would let a figure and a
        # table in the same paper sit on different label sets without any error.
        p = run_dir / "analysis_v2_resolved.json"
        if not p.exists():
            p = run_dir / "analysis_v2.json"
        if not p.exists():
            raise SystemExit(f"no analysis json in {run_dir}: run scripts/26_v2_analysis.py first")
        print(f"[34] reading {p.name}")
        res = json.loads(p.read_text(encoding="utf-8"))

    for f in (fig_attack_class, fig_null_forest, fig_difficulty, fig_estimands, fig_refusal):
        f(res, out)
    print(f"[34] figures in {out}")


if __name__ == "__main__":
    main()
