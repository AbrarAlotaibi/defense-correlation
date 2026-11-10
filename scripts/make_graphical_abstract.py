"""Build the graphical abstract as a vector PDF plus a raster PNG for the README.

The figure states the paper's claim in one picture: layering assumes each defense is blind
somewhere different (so an attack is caught by whichever layer covers it), while the measured
result is that the layers are blind in the same place (so one attack walks through the whole
stack). Nothing here is data-driven -- it is a schematic, drawn from constants below, and it
carries no measured number. Every quantitative figure lives in `make_figures.py`.

Deliberately sans-serif rather than the serif house style of the results figures: a graphical
abstract is read standalone at thumbnail size on a journal page, not inside the typeset paper.

Canvas is 1328 x 531 px at 100 dpi, which is Elsevier's minimum graphical-abstract size; the
PNG is written at 200 dpi (2656 x 1062 px), inside their preferred range.

Usage:
  python scripts/make_graphical_abstract.py --out paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# --- canvas ----------------------------------------------------------------------------
W, H = 1328.0, 531.0          # px at 100 dpi; also the axes coordinate system
DPI_PNG = 200

# --- palette ---------------------------------------------------------------------------
INK = "#1a1a1a"               # title
GREY = "#8c8c8c"              # subheads and caption
GREEN = "#1f6b45"             # "what layering assumes"
RED = "#a3281a"               # "what we measured", and the breach arrow
LAYER_FILL = "#c2d4e2"        # a defense layer
LAYER_EDGE = "#a8bfd2"
MODEL_FILL = "#eef2f7"        # the target model
MODEL_EDGE = "#90aec6"
RULE = "#dcdcdc"              # panel divider

# matplotlib walks this list and takes the first family present, so the figure looks the
# same on a machine with Helvetica as on a bare Linux box with only DejaVu.
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT = "sans-serif"

# --- geometry (top-down pixel coordinates; y_() flips them for matplotlib) --------------
BAND_TOP, BAND_BOT = 190.0, 375.0     # vertical extent of a defense layer
ATTACK_Y = 283.0                      # the height the attack travels at
LAYER_W = 26.0

# Left panel: every layer is blind somewhere different, and layer 1 covers ATTACK_Y.
LEFT_LAYERS = [
    (180.0, [(190.0, 222.0), (258.0, 375.0)]),
    (258.0, [(190.0, 305.0), (332.0, 375.0)]),
    (340.0, [(190.0, 250.0), (292.0, 375.0)]),
    (422.0, [(190.0, 312.0), (330.0, 375.0)]),
]
# Right panel: every layer is blind in the same place, and that place is ATTACK_Y.
RIGHT_LAYERS = [(x, [(190.0, 265.0), (300.0, 375.0)]) for x in (840.0, 918.0, 996.0, 1074.0)]

LEFT_MODEL = (490.0, 188.0, 90.0, 187.0)      # x, y_top, w, h
RIGHT_MODEL = (1148.0, 188.0, 100.0, 187.0)


def y_(y_top: float) -> float:
    """Top-down px -> matplotlib's bottom-up axes coordinate."""
    return H - y_top


def layer_column(ax, x: float, segments: list[tuple[float, float]]) -> None:
    """One defense layer, drawn as solid segments with a gap where it is blind."""
    for top, bot in segments:
        ax.add_patch(Rectangle((x, y_(bot)), LAYER_W, bot - top,
                               facecolor=LAYER_FILL, edgecolor=LAYER_EDGE, linewidth=0.8))


def model_box(ax, x: float, y_top: float, w: float, h: float) -> None:
    ax.add_patch(FancyBboxPatch((x, y_(y_top + h)), w, h,
                                boxstyle="round,pad=0,rounding_size=14",
                                facecolor=MODEL_FILL, edgecolor=MODEL_EDGE, linewidth=1.4))
    ax.text(x + w / 2, y_(y_top + h / 2), "model", rotation=90, rotation_mode="anchor",
            ha="center", va="center", fontsize=11.5, color=GREY, family=FONT)


def arrow(ax, x0: float, x1: float, y_top: float, color: str, lw: float) -> None:
    ax.annotate("", xy=(x1, y_(y_top)), xytext=(x0, y_(y_top)),
                arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.62",
                                color=color, linewidth=lw, shrinkA=0, shrinkB=0))


def panel_heading(ax, x: float, heading: str, colour: str, subhead: str) -> None:
    ax.text(x, y_(122), heading, fontsize=13.5, fontweight="bold", color=colour,
            ha="left", va="baseline", family=FONT)
    ax.text(x, y_(152), subhead, fontsize=11.5, color=GREY,
            ha="left", va="baseline", family=FONT)


def build(out: Path) -> None:
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(W / 2, y_(40), "Stacked LLM defenses share a blind spot",
            fontsize=19.5, fontweight="bold", color=INK, ha="center", va="baseline",
            family=FONT)

    ax.plot([668, 668], [y_(400), y_(100)], color=RULE, linewidth=1.0, solid_capstyle="butt")

    # --- left panel: the assumption ------------------------------------------------------
    panel_heading(ax, 72, "what layering assumes", GREEN,
                  "each layer is blind somewhere different")
    for x, segments in LEFT_LAYERS:
        layer_column(ax, x, segments)
    model_box(ax, *LEFT_MODEL)
    ax.text(82, y_(258), "attack", fontsize=12, color=INK, ha="left", va="baseline",
            family=FONT)
    arrow(ax, 82, 172, ATTACK_Y, INK, 2.0)
    # blocked at the first layer, because that layer is solid at the attack's height
    ax.add_patch(Circle((193, y_(ATTACK_Y)), 17, facecolor="white", edgecolor=GREEN,
                        linewidth=2.6, zorder=3))
    for dx, dy in ((1, 1), (1, -1)):
        ax.plot([193 - 8 * dx, 193 + 8 * dx],
                [y_(ATTACK_Y) - 8 * dy, y_(ATTACK_Y) + 8 * dy],
                color=GREEN, linewidth=2.4, solid_capstyle="round", zorder=4)

    # --- right panel: the measurement ----------------------------------------------------
    panel_heading(ax, 736, "what we measured", RED,
                  "the layers are blind in the same place")
    for x, segments in RIGHT_LAYERS:
        layer_column(ax, x, segments)
    model_box(ax, *RIGHT_MODEL)
    ax.text(746, y_(258), "attack", fontsize=12, color=INK, ha="left", va="baseline",
            family=FONT)
    arrow(ax, 746, 822, ATTACK_Y, INK, 2.0)
    arrow(ax, 828, 1150, ATTACK_Y, RED, 3.0)      # straight through every aligned gap

    # --- caption -------------------------------------------------------------------------
    ax.text(W / 2, y_(462), "Every layer reads the same model, so no amount of diversity "
                            "between layers", fontsize=12.5, color=GREY, ha="center",
            va="baseline", family=FONT)
    ax.text(W / 2, y_(492), "removes the failure they have in common.",
            fontsize=12.5, color=GREY, ha="center", va="baseline", family=FONT)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(out / "graphical_abstract.png", dpi=DPI_PNG, facecolor="white")
    plt.close(fig)
    print(f"wrote {out/'graphical_abstract.pdf'} and {out/'graphical_abstract.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("paper/figures"))
    build(p.parse_args().out)


if __name__ == "__main__":
    main()
