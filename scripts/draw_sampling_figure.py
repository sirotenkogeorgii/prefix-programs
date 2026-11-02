"""The sampling step of a variational prefix, for one coordinate of one slot; writes figures/sampling.png.

Top: a standard-normal draw eps. Bottom: the same draw mapped to z = mu + sigma * eps under the slot's posterior, with
the embedding prior behind it. mu, sigma, m0, s0 are read from a saved corrected-protocol checkpoint; the coordinate
shown is the one with the largest KL in slot 1 of that run. The draw rule is prefixlab/prefix.py (VariationalPrefix.forward).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch

TEAL, GREY, INK, MUTED = "#1d6d8a", "#8a938d", "#1c221f", "#5f6a64"
CKPT = "results_sync/checkpoints/flat2_translation-de_none_k2_s0_b3.pt"


def load_coordinate(path: Path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    mu, ls = sd["mu"][0].double(), sd["log_sigma"][0].double()
    m0, ls0 = sd["prior_mu"].double(), sd["prior_log_sigma"].double()
    kl = (ls0 - ls) + (torch.exp(2 * ls) + (mu - m0) ** 2) / (2 * torch.exp(2 * ls0)) - 0.5
    j = int(kl.argmax())
    return j, float(mu[j]), float(torch.exp(ls[j])), float(m0[j]), float(torch.exp(ls0[j])), float(kl[j] / math.log(2))


def pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def main(out: Path, ckpt: Path):
    j, mu, sigma, m0, s0, kl_bits = load_coordinate(ckpt)
    rng = np.random.default_rng(0)
    eps = np.sort(rng.standard_normal(6))
    z = mu + sigma * eps

    fig, (top, bot) = plt.subplots(2, 1, figsize=(6.4, 3.4), gridspec_kw=dict(hspace=0.9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.17)
    for ax in (top, bot):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=2.5)
        ax.spines["bottom"].set_color(GREY)

    # top: the noise
    x = np.linspace(-3.6, 3.6, 400)
    top.plot(x, pdf(x, 0, 1), color=INK, lw=1.1)
    top.set_xlim(-3.6, 3.6); top.set_ylim(0, 0.5)
    top.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    top.plot(eps, np.zeros_like(eps), "|", color=TEAL, ms=9, mew=1.3)
    top.text(-3.5, 0.44, "$\\varepsilon \\sim \\mathcal{N}(0, 1)$", fontsize=9, va="top")
    top.text(-3.5, 0.3, "fresh at every forward call", fontsize=7, color=MUTED, va="top")

    # bottom: the posterior, the prior behind it, and the mapped draws
    lo = min(m0, mu) - 3.4 * max(s0, sigma); hi = max(m0, mu) + 3.4 * max(s0, sigma)
    xz = np.linspace(lo, hi, 500)
    bot.plot(xz, pdf(xz, m0, s0), color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=3)
    bot.plot(xz, pdf(xz, mu, sigma), color=TEAL, lw=1.3, zorder=4)
    bot.set_xlim(lo, hi); ymax = pdf(mu, mu, sigma) * 1.22; bot.set_ylim(0, ymax)
    bot.plot(z, np.zeros_like(z), "|", color=TEAL, ms=9, mew=1.3, zorder=5)
    bot.axvline(mu, 0, pdf(mu, mu, sigma) / ymax, color=TEAL, lw=0.7, ls=(0, (1.5, 2)))
    bot.text(mu, pdf(mu, mu, sigma) * 1.04, "$\\mu$", ha="center", va="bottom", fontsize=9, color=TEAL)
    h = pdf(mu + sigma, mu, sigma)
    bot.annotate("", xy=(mu + sigma, h), xytext=(mu, h), arrowprops=dict(arrowstyle="<->", color=TEAL, lw=0.7, shrinkA=0, shrinkB=0))
    bot.text(mu + sigma / 2, h * 1.06, "$\\sigma$", ha="center", va="bottom", fontsize=8, color=TEAL)
    bot.text(m0, pdf(m0, m0, s0) * 1.04, "$\\pi_0$", ha="center", va="bottom", fontsize=9, color=GREY)
    side = "left" if mu > m0 else "right"
    bot.text(lo if side == "left" else hi, ymax * 0.98,
             "$z = \\mu + \\sigma\\,\\varepsilon \;\\sim\; q = \\mathcal{N}(\\mu, \\sigma^2)$", fontsize=9, va="top", ha=side, color=TEAL)
    bot.text(lo if side == "left" else hi, ymax * 0.70, "prior $\\pi_0 = \\mathcal{N}(m_0, s_0^2)$, dashed", fontsize=7, va="top",
             ha=side, color=MUTED)
    bot.set_xlabel(f"value at coordinate {j + 1} of 1,536 in slot 1 (embedding units)", fontsize=7, color=MUTED, labelpad=3)

    # the mapping of each draw
    for e, zz in zip(eps, z):
        bot.add_artist(ConnectionPatch(xyA=(e, 0), coordsA=top.transData, xyB=(zz, 0), coordsB=bot.transData,
                                       color=TEAL, lw=0.5, alpha=0.45, clip_on=False, zorder=1))
    fig.text(0.985, 0.955, f"$\\mu = {mu:.3f}$, $\\sigma = {sigma:.4f}$, $m_0 = {m0:.3f}$, $s_0 = {s0:.4f}$;  "
             f"KL for this coordinate: {kl_bits:.1f} bits", fontsize=6.5, color=MUTED, ha="right", va="center")
    fig.savefig(out, dpi=200)
    print("wrote", out, "coordinate", j, f"mu={mu:.4f} sigma={sigma:.4f} m0={m0:.4f} s0={s0:.4f} kl_bits={kl_bits:.2f}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "figures" / "sampling.png", root / CKPT)
