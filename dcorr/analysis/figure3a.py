"""Figure 3(a): measured joint-breach points plotted over the analytic curves.

The analytic family is joint(rho) = p1 p2 + rho sqrt(p1(1-p1) p2(1-p2)) for a few
reference (p1, p2). Each measured pair is a point at (phi_hat, joint_hat) with a
horizontal bootstrap CI bar on phi. Same-row pairs are drawn distinctly from cross-row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .stats import PairStat, phi_from_rates


def render(pairs: list[PairStat], dest: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    rhos = np.linspace(-0.3, 1.0, 100)

    ref_rates = sorted({(round(p.p1, 1), round(p.p2, 1)) for p in pairs}) or [(0.3, 0.3)]
    for (p1, p2) in ref_rates[:5]:
        ax.plot(rhos, [phi_from_rates(p1, p2, r) for r in rhos], lw=1.0, alpha=0.5,
                label=f"analytic p=({p1},{p2})")

    for p in pairs:
        marker = "s" if p.same_row else "o"
        color = "#c0392b" if p.same_row else "#2c3e50"
        ax.errorbar(p.phi, p.joint,
                    xerr=[[p.phi - p.phi_ci[0]], [p.phi_ci[1] - p.phi]],
                    fmt=marker, color=color, ms=6, capsize=3, lw=1.0, zorder=5)
        ax.annotate(f"{p.d1[:4]}x{p.d2[:4]}", (p.phi, p.joint),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"failure correlation $\phi$ (= $\rho$)")
    ax.set_ylabel(r"joint breach ASR$_{d_1 d_2}$")
    ax.set_title(title or "Figure 3(a): measured joint breach vs failure correlation")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="#c0392b", ls="", label="same row (H1)"),
        Line2D([0], [0], marker="o", color="#2c3e50", ls="", label="cross row (H2)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200)
    fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)
    return dest
