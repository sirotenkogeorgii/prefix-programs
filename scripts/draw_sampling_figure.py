"""Schematic of the two units (slots, bits) and of where posterior draws happen; writes figures/sampling.png.

Numbers in the labels come from the code and the synced results: prefixlab/prefix.py (draw rule), prefixlab/train.py
(batch 16, eval batch 32, eval every 100 steps), prefixlab/sweep.py (8 test passes), results_sync/flat2.csv (set sizes),
and the flat2 translation-de k=2 beta=3 checkpoint (prior sigma 0.0248, posterior sigma 0.0234).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

DRAW, DRAW_SOFT = "#1d6d8a", "#d9ebf2"
RATE, RATE_SOFT = "#b0540c", "#f6e4d3"
INK, MUTED, SOFT = "#1c221f", "#5f6a64", "#c7cec8"


def box(ax, x, y, w, h, lines, *, fc="white", ec=INK, ls="-", sizes=None, colors=None, weights=None, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=lw, ls=ls))
    n = len(lines)
    sizes = sizes or [6.5] * n
    colors = colors or [INK] * n
    weights = weights or ["normal"] * n
    step = h / (n + 1)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - step * (i + 1), t, ha="center", va="center", fontsize=sizes[i], color=colors[i],
                fontweight=weights[i])


def arrow(ax, p, q, color=INK, ls: str | tuple = "-", lw=0.9, head=True):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=lw,
                                                       shrinkA=0, shrinkB=0, mutation_scale=7, linestyle=ls))


def bracket(ax, x0, x1, y, color=INK, depth=0.8):
    ax.plot([x0, x0, x1, x1], [y + depth, y, y, y + depth], color=color, lw=0.8, solid_capstyle="butt")


def panel_units(ax):
    ax.set_xlim(0, 100); ax.set_ylim(-2.8, 50); ax.axis("off")
    # token-embedding matrix and prior
    ax.text(1.5, 46.8, "token-embedding matrix", ha="left", va="center", fontsize=6.5, fontweight="bold")
    ax.text(1.5, 44.6, "151,936 rows × 1,536", ha="left", va="center", fontsize=6, color=MUTED)
    ax.add_patch(Rectangle((2, 31), 10, 10, fc="white", ec=INK, lw=0.8))
    for yy in (33, 35, 37, 39):
        ax.plot([2.6, 11.4], [yy, yy], color=SOFT, lw=0.7)
    arrow(ax, (7, 31), (7, 26.2))
    ax.text(7.8, 28.6, "fit mean, std\nper coordinate", ha="left", va="center", fontsize=5.5, color=MUTED, linespacing=1.1)
    box(ax, 1.5, 20.5, 12, 5.5, ["prior $\\pi_0$", "$\\mathcal{N}(m_0,\\,\\mathrm{diag}\\,s_0^2)$"], weights=["bold", "normal"])
    ax.text(7.5, 19, "$s_0 \\approx 0.025$ per coordinate", ha="center", va="center", fontsize=5.5, color=MUTED)
    ax.plot([13.5, 16], [23.25, 23.25], color=RATE, lw=1, ls=(0, (3, 2)))
    # rate bus into the bits box
    ax.plot([16, 16], [13.5, 44.5], color=RATE, lw=1, ls=(0, (3, 2)))
    arrow(ax, (16, 44.5), (19, 44.5), color=RATE, ls=(0, (3, 2)), lw=1)
    for yy in (36, 30, 24, 13.5):
        ax.plot([16, 19], [yy, yy], color=RATE, lw=1, ls=(0, (3, 2)))
    ax.text(14.8, 16, "$\\mathrm{KL}(q\\,\\|\\,\\pi_0)$ per slot", rotation=90, ha="center", va="center", fontsize=5.5, color=RATE)
    box(ax, 19, 42, 35, 5, ["unit 2 · bits $= \\sum_{\\mathrm{slots}} \\mathrm{KL}(q\\,\\|\\,\\pi_0)\\,/\\ln 2$"],
        fc=RATE_SOFT, ec=RATE, sizes=[6.5], colors=[RATE], weights=["bold"])
    # slot posteriors
    for yy, name in ((36, "slot 1"), (30, "slot 2"), (24, "slot 3"), (13.5, "slot $k$")):
        box(ax, 19, yy - 2.4, 18, 4.8, [name, "$q = \\mathcal{N}(\\mu,\\,\\mathrm{diag}\\,\\sigma^2)$"],
            sizes=[5.5, 6.5], colors=[MUTED, INK])
    ax.text(28, 18.6, "$\\vdots$", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(28, 9.6, "each $\\mu, \\sigma \\in \\mathbb{R}^{1536}$, trained with its own noise", ha="center", va="center",
            fontsize=5.5, color=MUTED)
    # fan-in to the draw
    for yy in (36, 30, 24, 13.5):
        ax.plot([37, 43], [yy, 25], color=DRAW, lw=0.9)
    ax.text(50.5, 32.2, "$\\varepsilon \\sim \\mathcal{N}(0, I)$, fresh at every forward call", ha="center", va="center",
            fontsize=5.8, color=INK)
    arrow(ax, (50.5, 30.8), (50.5, 27.7), color=DRAW)
    box(ax, 43, 22.3, 15, 5.4, ["one draw per slot", "$z = \\mu + \\sigma \\odot \\varepsilon$"], fc=DRAW_SOFT, ec=DRAW,
        sizes=[5.5, 7], colors=[MUTED, INK])
    arrow(ax, (58, 25), (62, 25), color=DRAW)
    # the input sequence
    cells = [("$z_1$", "z"), ("$z_2$", "z"), ("$z_3$", "z"), ("$z_k$", "z"),
             ("The", "t"), ("cat", "t"), ("…", "t"), ("↵", "t"), ("Кот", "y"), ("…", "y")]
    x = 62.0; w, g = 3.4, 0.3
    xs = []
    for label, kind in cells:
        fc = DRAW_SOFT if kind == "z" else "white"
        ec = DRAW if kind == "z" else INK
        ls = (0, (2, 1.5)) if kind == "y" else "-"
        ax.add_patch(Rectangle((x, 22.3), w, 5.4, fc=fc, ec=ec, lw=0.8, ls=ls))
        ax.text(x + w / 2, 25, label, ha="center", va="center", fontsize=6)
        xs.append(x); x += w + g
    bracket(ax, xs[0], xs[3] + w, 21.4, color=DRAW)
    ax.text((xs[0] + xs[3] + w) / 2, 19.6, "unit 1 · $k$ slots", ha="center", va="center", fontsize=6.2, color=DRAW,
            fontweight="bold")
    bracket(ax, xs[4], xs[7] + w, 21.4)
    ax.text((xs[4] + xs[7] + w) / 2, 19.6, "input tokens", ha="center", va="center", fontsize=5.5, color=MUTED)
    bracket(ax, xs[8], xs[9] + w, 21.4)
    ax.text((xs[8] + xs[9] + w) / 2, 19.6, "target, scored", ha="center", va="center", fontsize=5.5, color=MUTED)
    # model and distortion
    xm = (xs[0] + xs[9] + w) / 2
    arrow(ax, (xm, 18.4), (xm, 15.2))
    box(ax, xm - 16, 10, 32, 5.2, ["frozen Qwen2.5-1.5B, bfloat16", "attends to the drawn vectors as to tokens"],
        sizes=[6.5, 5.5], colors=[INK, MUTED], weights=["bold", "normal"])
    arrow(ax, (xm, 10), (xm, 7.2), color=DRAW)
    box(ax, xm - 16, 2, 32, 5.2, ["distortion", "$L$ = bits per target token, teacher-forced"], fc=DRAW_SOFT, ec=DRAW,
        sizes=[6.5, 6], colors=[DRAW, INK], weights=["bold", "normal"])
    ax.text(xm, 0.5, "$\\hat D = (L - L_{\\mathrm{floor}})\\,/\\,(L_{\\mathrm{none}} - L_{\\mathrm{floor}})$", ha="center", va="center",
            fontsize=5.8, color=INK)
    ax.text(xm, -1.5, "frontier: smallest bits among runs with $\\hat D \\leq$ op", ha="center", va="center", fontsize=5.8,
            color=MUTED)


def die(ax, x, y, s=2.2, label="ε", fs=5.5):
    ax.add_patch(Rectangle((x, y), s, s, fc=DRAW_SOFT, ec=DRAW, lw=0.7))
    if label:
        ax.text(x + s / 2, y + s / 2, label, ha="center", va="center", fontsize=fs)


def panel_schedule(ax):
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 24); ax.axis("off")
    # lane 1: training
    ax.text(1, 21.2, "training", fontsize=6.8, fontweight="bold", va="center")
    ax.text(1, 19.4, "1,500 steps · batch 16", fontsize=5.5, color=MUTED, va="center")
    for j in range(12):
        die(ax, 20 + 2.8 * j, 19.4)
    ax.text(54.6, 20.5, "…", ha="center", va="center", fontsize=7, color=MUTED)
    die(ax, 56.5, 19.4)
    for j, t in ((0, "step 1"), (1, "2"), (2, "3")):
        ax.text(21.1 + 2.8 * j, 18.5, t, ha="center", va="center", fontsize=4.8, color=MUTED)
    ax.text(57.6, 18.5, "1500", ha="center", va="center", fontsize=4.8, color=MUTED)
    ax.text(20, 16.9, "one draw per step, shared by the 16 examples of that batch;  loss $= L(\\mathrm{batch}) + "
            "\\beta\\,\\mathrm{KL}/N_{\\mathrm{train\\ tokens}}$", fontsize=5.6, va="center")
    ax.plot([1, 99], [15.6, 15.6], color=SOFT, lw=0.6)
    # lane 2: dev selection
    ax.text(1, 13.6, "dev, every 100 steps", fontsize=6.8, fontweight="bold", va="center")
    ax.text(1, 11.8, "corrected protocol", fontsize=5.5, color=MUTED, va="center")
    for j in range(7):
        die(ax, 20 + 2.8 * j, 11.8)
    bracket(ax, 20, 20 + 2.8 * 6 + 2.2, 11.2, depth=0.5)
    ax.text(29.2, 10.4, "one pass over the dev set, a batch of 32 per die", ha="center", va="center", fontsize=4.8, color=MUTED)
    ax.text(41, 12.9, "× 15 evaluations", fontsize=5.5, color=MUTED, va="center")
    ax.text(20, 8.9, "7 draws per pass on translation (199 dev examples), 72 to 104 on recall;  checkpoint = lowest "
            "$L(\\mathrm{dev}) + \\beta \\cdot$ rate", fontsize=5.6, va="center")
    ax.plot([1, 99], [7.7, 7.7], color=SOFT, lw=0.6)
    # lane 3: test
    ax.text(1, 5.9, "test, reported", fontsize=6.8, fontweight="bold", va="center")
    ax.text(1, 4.1, "8 passes · batch 32", fontsize=5.5, color=MUTED, va="center")
    rows = ((6.0, "pass 1"), (4.5, "pass 2"), (3.0, "pass 3"), (0.3, "pass 8"))
    for yy, name in rows:
        ax.text(19.3, yy + 0.55, name, ha="right", va="center", fontsize=4.8, color=MUTED)
        for j in range(12):
            ax.add_patch(Rectangle((20 + 1.7 * j, yy), 1.4, 1.1, fc=DRAW_SOFT, ec=DRAW, lw=0.6))
        ax.text(41.3, yy + 0.55, "…", va="center", fontsize=6, color=MUTED)
    ax.text(30, 1.9, "$\\vdots$", ha="center", va="center", fontsize=6, color=MUTED)
    ax.text(44.5, 6.9, "B batches per pass: 32 on translation,", fontsize=5, color=MUTED, va="center")
    ax.text(44.5, 5.7, "140 to 203 on recall", fontsize=5, color=MUTED, va="center")
    arrow(ax, (44.5, 2.4), (50.5, 2.4))
    box(ax, 51.5, 0.3, 47.5, 4.2, ["mean of the 8 pass values $= L_{\\mathrm{sampled}} \\rightarrow \\hat D$",
                                   "256 to 1,624 independent draws behind one reported number"],
        fc=DRAW_SOFT, ec=DRAW, sizes=[6.3, 5.3], colors=[INK, MUTED], weights=["bold", "normal"])


def main(out: Path):
    fig = plt.figure(figsize=(7.6, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[50, 24.5], hspace=0.04, left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1])
    panel_units(ax_a); panel_schedule(ax_b)
    ax_a.text(0.5, 49.3, "(a)", fontsize=7, fontweight="bold", va="center")
    ax_b.text(0.5, 23.2, "(b)", fontsize=7, fontweight="bold", va="center")
    ax_b.plot([1, 99], [23.6, 23.6], color=INK, lw=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "figures" / "sampling.png")
