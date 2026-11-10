"""Figure: per-member attack success on both targets, with the undefended level marked.

The replication's message is not that the two targets agree, which they largely do on the
central question, but that one member reverses sign between them. A grouped bar chart with the
undefended rate drawn as a reference line on each panel shows that directly: every member sits
near its target's undefended line except the perturbation aggregator, which sits far above it
on the aligned target and slightly below on the weakly aligned one.

Follows the project's print conventions: validated categorical colours in fixed order, one
axis, recessive grid, no interaction layer, and a hatch on the second series so the figure
survives greyscale printing.

Usage: python scripts/41_fig_replication.py
"""
from __future__ import annotations

import os
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.9,
    "axes.axisbelow": True, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False,
})

SHORT = {"ppl_filter": "likelihood", "token_anomaly": "token-anomaly",
         "llamaguard": "guard model", "probe": "probe (mid)", "probe_b": "probe (early)",
         "refusal_prime": "refusal prompt", "smoothllm": "perturbation"}
ORDER = ["smoothllm", "probe", "probe_b", "ppl_filter", "token_anomaly",
         "refusal_prime", "llamaguard"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def asr(run: str, seeds: list[int]) -> dict:
    """adaptive-attack breach rate per defense, pooled over whichever seeds are graded."""
    # Prefer the resolved labels, where grader abstentions have been adjudicated by the second
    # grader, because that is what the manuscript's tables report. Falling back to gold.jsonl
    # silently would put a figure and a table in the same paper on different label sets.
    rows, used = [], []
    for s in seeds:
        d = REPO / "results" / run / f"seed{s}"
        p = d / "gold_resolved.jsonl"
        if not p.exists():
            p = d / "gold.jsonl"
        if not p.exists():
            continue
        used.append(f"seed{s}:{p.name}")
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  {run}: {', '.join(used)}")
    out = {}
    for d in set(r["defense"] for r in rows):
        g = [r for r in rows if r["defense"] == d and r["attack"] == "adaptive"]
        if g:
            k = sum(1 for r in g if r["breach"])
            out[d] = (k / len(g), *wilson(k, len(g)), len(g))
    return out


vic = asr("v2_vicuna", [0, 1, 2])
lla = asr("v2_llama2", [0, 1, 2])
if not vic or not lla:
    raise SystemExit("need graded gold.jsonl for both runs")

members = [d for d in ORDER if d in vic and d in lla]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, (data, title, colour, hatch) in zip(
        axes,
        [(vic, "Vicuna-7B-v1.5 (weakly aligned)", BLUE, None),
         (lla, "Llama-2-7b-chat (aligned)", ORANGE, "//")]):
    y = np.arange(len(members))
    vals = [data[d][0] for d in members]
    lo = [data[d][0] - data[d][1] for d in members]
    hi = [data[d][2] - data[d][0] for d in members]
    ax.barh(y, vals, color=colour, height=0.62, hatch=hatch, edgecolor="white", linewidth=0.8)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=2)
    und = data.get("undefended", (None,))[0]
    if und is not None:
        ax.axvline(und, color=INK, linestyle="--", linewidth=1.1)
        ax.text(und, len(members) - 0.35, "  undefended", fontsize=7, color=INK, va="center")
    ax.set_yticks(y, [SHORT.get(d, d) for d in members])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("attack success under the fluent adaptive adversary")
    ax.set_title(title, color=INK)

out = REPO / "results" / "v2_vicuna" / "figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig6_replication.{ext}")
plt.close(fig)

# Copy into the manuscript tree in the same breath as writing it. Regenerating here and
# forgetting the copy leaves the paper including an older PDF under the right filename, which
# nothing downstream can detect: the build succeeds and the figure is simply wrong.
MS = Path(os.environ.get(
    "DCORR_MANUSCRIPT_FIGURES",
    Path(__file__).resolve().parents[1].parent / "defense-correlation-paper" / "paper_A" / "figures",
))
if MS.is_dir():
    shutil.copy2(out / "fig6_replication.pdf", MS / "fig6_replication.pdf")
    print(f"copied fig6_replication.pdf -> {MS}")
else:
    print(f"NOT copied: {MS} does not exist; copy it by hand before building")
n_v = next(iter(vic.values()))[3]
n_l = next(iter(lla.values()))[3]
print(f"wrote fig6_replication.pdf / .png   (Vicuna n={n_v}, Llama-2 n={n_l} per member)")
for d in members:
    print(f"   {SHORT.get(d,d):<16}vicuna {vic[d][0]:.2f}   llama2 {lla[d][0]:.2f}")
