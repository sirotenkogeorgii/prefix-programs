"""The sampling step of a variational prefix, drawn like a VAE figure in the paper's notation; writes figures/sampling.png.

Each slot i has a posterior q_i with trained mu_i, sigma_i; a sample z_i = mu_i + sigma_i * eps_i (prefixlab/prefix.py,
VariationalPrefix.forward) is drawn per slot at every forward call; the k samples form the prefix, which precedes the
input tokens x; the frozen model M scores the target tokens y. d is the model's embedding dimension (1,536).
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
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    # column 1: the posterior of one slot, trained mu and sigma (no encoder); one such pair per slot
    header(ax, 9, ["posterior of slot $i$", r"$q_i = \mathcal{N}(\mu_i,\ \mathrm{diag}\,\sigma_i^2)$"])
    rect(ax, 6, 19, 6, 10, r"$\mu_i$")
    rect(ax, 6, 6, 6, 10, r"$\sigma_i$")
    ax.text(3, 3.4, r"trained, one pair per slot, $i = 1, \ldots, k$", ha="left", va="top", fontsize=8.5)

    # arrow 1: the draw, repeated for every slot
    arrow(ax, 13.5, 35.5, 17.5, r"sample $z_i = \mu_i + \sigma_i \odot \varepsilon_i$", r"$\varepsilon_i \sim \mathcal{N}(0, I_d)$", fs=9)

    # column 2: the prefix of k samples followed by the input tokens, one sequence
    header(ax, 46.5, ["prefix: $k$ samples", r"$z_i \in \mathbb{R}^d$"])
    header(ax, 68, ["input", "tokens $x$"])
    xs = [36, 41.5, 47, 52.5]
    for x0, lab in zip(xs, [r"$z_1$", r"$z_2$", None, r"$z_k$"]):
        if lab is None:
            ax.text(x0 + 2.25, 17.5, r"$\cdots$", ha="center", va="center", fontsize=11)
        else:
            rect(ax, x0, 12.5, 4.5, 10, lab, fs=10)
    rect(ax, 63, 12.5, 10, 10, r"$x$")
    ax.plot([36, 73], [10.2, 10.2], color="black", lw=0.6)
    ax.plot([36, 36], [10.2, 11.2], color="black", lw=0.6); ax.plot([73, 73], [10.2, 11.2], color="black", lw=0.6)
    ax.text(54.5, 9.2, "one input sequence to the model", ha="center", va="top", fontsize=8.5)

    # arrow 2: the frozen model scores the target given the whole sequence
    arrow(ax, 74.5, 86.5, 17.5, r"frozen model $M$", r"$p_M(y \mid z_{1:k}, x)$")

    # column 3: the target tokens
    header(ax, 91.5, ["target", "tokens $y$"])
    rect(ax, 88, 3, 7, 32)
    ax.text(91.5, 19, r"$y$", ha="center", va="center", fontsize=11)

    fig.savefig(out, dpi=220)
    print("wrote", out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "figures" / "sampling.png")
