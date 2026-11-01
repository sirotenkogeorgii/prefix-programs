"""One slot of a variational prefix drawn like a VAE latent; writes figures/sampling.png.

Labels follow the code: the draw z = mu + sigma * eps is prefixlab/prefix.py (VariationalPrefix.forward), the rate is
kl_bits() there, the objective is the loss in prefixlab/train.py, and the prior is gaussian_prior_from_embeddings.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

DRAW, DRAW_SOFT = "#1d6d8a", "#d9ebf2"
RATE, RATE_SOFT = "#b0540c", "#f6e4d3"
INK, MUTED, SOFT = "#1c221f", "#5f6a64", "#c7cec8"


def box(ax, x, y, w, h, lines, *, fc="white", ec=INK, ls="-", sizes=None, colors=None, weights=None, lw=0.8, pad=0.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}", fc=fc, ec=ec, lw=lw, ls=ls))
    n = len(lines)
    sizes = sizes or [7] * n
    colors = colors or [INK] * n
    weights = weights or ["normal"] * n
    step = h / (n + 1)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - step * (i + 1), t, ha="center", va="center", fontsize=sizes[i], color=colors[i],
                fontweight=weights[i])


def arrow(ax, p, q, color=INK, ls: str | tuple = "-", lw=0.9, style="-|>"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                                       mutation_scale=7, linestyle=ls))


def node(ax, x, y, sym, r=1.8, color=DRAW):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=0.9))
    ax.text(x, y, sym, ha="center", va="center", fontsize=8, color=color)


def bell(ax, x0, y0, w, h, color, shift=0.0, width=1.0):
    t = np.linspace(-2.6, 2.6, 80)
    ax.plot(x0 + (t + shift) / 5.2 * w + w / 2, y0 + h * np.exp(-0.5 * (t / width) ** 2), color=color, lw=0.9)
    ax.plot([x0, x0 + w], [y0, y0], color=SOFT, lw=0.6)


def main(out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 100); ax.set_ylim(-6, 50); ax.axis("off")

    # prior
    box(ax, 2, 40, 40, 8, [""], fc="white", ec=INK)
    bell(ax, 4, 41.5, 7, 4.5, INK)
    ax.text(25, 46, "prior $\\pi_0 = \\mathcal{N}(m_0,\\,\\mathrm{diag}\\,s_0^2)$", ha="center", va="center", fontsize=7)
    ax.text(25, 42.3, "fitted to the token-embedding matrix", ha="center", va="center", fontsize=5.8, color=MUTED)

    # rate: KL between prior and posterior
    arrow(ax, (17, 40), (17, 34.4), color=RATE, ls=(0, (3, 2)), style="<|-|>")
    ax.text(19, 37.2, "rate $= \\mathrm{KL}(q\\,\\|\\,\\pi_0)\\,/\\ln 2$ bits", ha="left", va="center", fontsize=7, color=RATE,
            fontweight="bold")

    # posterior group (the latent, as in a VAE)
    ax.add_patch(FancyBboxPatch((2, 2), 44, 32, boxstyle="round,pad=0,rounding_size=0.8", fc=DRAW_SOFT, ec=DRAW, lw=0.9,
                                ls=(0, (4, 2.5)), alpha=0.55))
    ax.text(4.5, 31, "one slot: posterior $q = \\mathcal{N}(\\mu,\\,\\mathrm{diag}\\,\\sigma^2)$ over $\\mathbb{R}^{1536}$",
            ha="left", va="center", fontsize=7, fontweight="bold")
    bell(ax, 38.5, 27.5, 7, 4, DRAW, shift=0.6, width=0.8)
    box(ax, 6, 20, 9, 6, ["$\\mu$"], sizes=[9])
    box(ax, 6, 8, 9, 6, ["$\\sigma$"], sizes=[9])
    ax.text(10.5, 17.6, "trained", ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.text(10.5, 5.6, "trained", ha="center", va="center", fontsize=5.8, color=MUTED)
    node(ax, 24, 11, "×")
    node(ax, 24, 23, "+")
    arrow(ax, (15, 11), (22.2, 11), color=DRAW)
    arrow(ax, (15, 23), (22.2, 23), color=DRAW)
    node(ax, 34, 11, "$\\varepsilon$", color=INK)
    ax.text(34, 5.6, "$\\varepsilon \\sim \\mathcal{N}(0, I)$, fresh each call", ha="center", va="center", fontsize=5.8,
            color=MUTED)
    arrow(ax, (32.2, 11), (25.8, 11), color=INK)
    arrow(ax, (24, 12.8), (24, 21.2), color=DRAW)
    ax.text(25.3, 17, "$\\sigma \\odot \\varepsilon$", ha="left", va="center", fontsize=6.5, color=DRAW)
    arrow(ax, (25.8, 23), (30, 23), color=DRAW)
    box(ax, 30, 19.5, 8, 7, ["$z$"], fc="white", ec=DRAW, sizes=[9])
    ax.text(34, 16.6, "$z = \\mu + \\sigma \\odot \\varepsilon$", ha="center", va="center", fontsize=6.5, color=DRAW)
    ax.text(24, -0.5, "the latent of a VAE, without an encoder: $\\mu$ and $\\sigma$ are the free parameters",
            ha="center", va="center", fontsize=5.8, color=MUTED)

    # decoder: the frozen model
    arrow(ax, (38, 23), (52, 23), color=DRAW)
    ax.text(45, 25, "prefix", ha="center", va="center", fontsize=5.8, color=MUTED)
    box(ax, 52, 15, 26, 16, ["frozen Qwen2.5-1.5B", "reads $z$ as one extra token", "before the input"],
        sizes=[7.5, 6, 6], colors=[INK, MUTED, MUTED], weights=["bold", "normal", "normal"])
    box(ax, 55, 2, 20, 6, ["input tokens"], sizes=[6.5])
    arrow(ax, (65, 8), (65, 15))
    ax.text(65, 33.5, "the decoder, not trained", ha="center", va="center", fontsize=5.8, color=MUTED)

    # distortion
    arrow(ax, (78, 23), (84, 23), color=DRAW)
    box(ax, 84, 15, 14, 16, ["distortion", "$L$ = bits per", "target token"], fc=DRAW_SOFT, ec=DRAW,
        sizes=[7, 6.2, 6.2], colors=[DRAW, INK, INK], weights=["bold", "normal", "normal"])
    ax.text(91, 33.5, "reconstruction term", ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.text(91, 11.5, "scored on draws $z$,", ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.text(91, 9, "never on $\\mu$", ha="center", va="center", fontsize=5.8, color=MUTED)

    # objective
    ax.text(50, -4.2, "objective $= L + \\beta \\cdot \\mathrm{KL}\\,/\\,N_{\\mathrm{train\\ tokens}}$, one draw per step;"
            "  a $k$-slot prefix repeats the slot $k$ times and sums the rates", ha="center", va="center", fontsize=6.3)
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "figures" / "sampling.png")
