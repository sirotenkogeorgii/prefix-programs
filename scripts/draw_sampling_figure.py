"""The sampling step for one slot, drawn like a VAE figure, in the paper's notation; writes figures/sampling.png.

The draw rule z = mu + sigma * eps is prefixlab/prefix.py (VariationalPrefix.forward); mu and sigma are the trained
parameters of the slot posterior q (no encoder); the frozen model M scores the target y given the sample z and input x.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True})


def rect(ax, x, y, w, h, label=None, fs=11):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=1.0))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, x1, y, above, below, fs=10):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=12))
    ax.text((x0 + x1) / 2, y + 2.6, above, ha="center", va="bottom", fontsize=fs)
    ax.text((x0 + x1) / 2, y - 2.6, below, ha="center", va="top", fontsize=fs)


def header(ax, x, lines, fs=10):
    for i, t in enumerate(lines):
        ax.text(x, 44.5 - 3.7 * i, t, ha="center", va="top", fontsize=fs)


def main(out: Path):
    fig, ax = plt.subplots(figsize=(6.6, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    # column 1: the slot's distribution, trained parameters mu and sigma (no encoder)
    header(ax, 14, ["posterior", "of the slot", r"$q = \mathcal{N}(\mu,\ \mathrm{diag}\,\sigma^2)$"])
    rect(ax, 11, 19, 6, 10, r"$\mu$")
    rect(ax, 11, 6, 6, 10, r"$\sigma$")
    ax.text(14, 3.2, r"trained $\mu$, $\sigma$", ha="center", va="top", fontsize=9)

    # arrow 1: the draw
    arrow(ax, 18.5, 44, 17.5, r"sample $z = \mu + \sigma \odot \varepsilon$", r"$\varepsilon \sim \mathcal{N}(0, I)$")

    # column 2: the slot vector
    header(ax, 50, ["posterior sample", r"$z \in \mathbb{R}^{1536}$"])
    rect(ax, 46.5, 12.5, 7, 10, r"$z$")

    # arrow 2: the frozen model as the decoder
    arrow(ax, 55, 83.5, 17.5, r"frozen model $M$", r"$p_M(y \mid z, x)$")

    # column 3: the target tokens
    header(ax, 90, ["target", "tokens $y$"])
    rect(ax, 86, 3, 8, 32)
    ax.text(90, 19, r"$y$", ha="center", va="center", fontsize=11)

    fig.savefig(out, dpi=220)
    print("wrote", out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "figures" / "sampling.png")
