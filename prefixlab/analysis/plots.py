"""Figures for the report; every function returns a matplotlib Figure and reads only data frames."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import lang_of, task_of

LANG_COLORS = {"ru": "#1f77b4", "de": "#d62728", "en": "#2ca02c"}
TASKS = ["translation", "recall", "continuation"]


def plot_curves(sweepD: pd.DataFrame, conds=("none", "lang", "task"), col: str = "D", ops=(0.25, 0.10)):
    """D̂ vs k per task (panels), one line per language, line style per conditioning."""
    styles = {"none": "-", "lang": "--", "task": ":"}
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.2 * len(TASKS), 3.6), sharey=True)
    a = sweepD[sweepD.design == "A"]
    for ax, task in zip(axes, TASKS):
        for cell in sorted(c for c in a.cell.unique() if task_of(c) == task):
            for cond in conds:
                g = a[(a.cell == cell) & (a.cond == cond)].groupby("k")[col].agg(["mean", "min", "max"]).sort_index()
                if g.empty:
                    continue
                ax.plot(g.index, g["mean"], styles.get(cond, "-"), marker="o", ms=3, color=LANG_COLORS[lang_of(cell)],
                        label=f"{lang_of(cell)} {cond}" if cond != "none" else lang_of(cell))
                if cond == "none":
                    ax.fill_between(g.index, g["min"], g["max"], color=LANG_COLORS[lang_of(cell)], alpha=0.12)
        for op in ops:
            ax.axhline(op, color="grey", lw=0.6, ls="--")
        ax.set_xscale("log", base=2); ax.set_title(task); ax.set_xlabel("slots k"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalized distortion D̂")
    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
    axes[-1].legend(handles.values(), handles.keys(), fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_frontier(tables: dict[str, pd.DataFrame], cells, dcol: str = "D_sampled", ops=(0.25, 0.10)):
    """Rate–distortion scatter per cell: KL bits (log) vs D̂, one marker style per table (flat / flat2)."""
    markers = {"flat": "o", "flat2": "s", "flat2 seed 1": "D"}
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 3.4), sharey=True)
    for ax, cell in zip(np.atleast_1d(axes), cells):
        for name, df in tables.items():
            g = df[df.cell == cell]
            if g.empty:
                continue
            ax.scatter(g[dcol], g.kl_bits, marker=markers.get(name, "x"), s=28, alpha=0.8, label=name)
        for op in ops:
            ax.axvline(op, color="grey", lw=0.6, ls="--")
        ax.set_yscale("log"); ax.set_title(cell); ax.set_xlabel("D̂ (sampled)"); ax.set_xlim(-0.02, 0.6); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("variational code length (bits)"); np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_hier_bars(totals_task: pd.DataFrame, totals_lang: pd.DataFrame, op: float):
    """Stacked bars: centre + deviations per group under both groupings at one operating point."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (name, t) in zip(axes, (("task-centred", totals_task), ("language-centred", totals_lang))):
        t = t[t.op == op].reset_index(drop=True)
        x = np.arange(len(t))
        ax.bar(x, t.centre.fillna(0), color="#444", label="centre")
        bottom = t.centre.fillna(0).to_numpy()
        devcols = [c for c in t.columns if c.startswith("dev_")]
        for c in devcols:
            v = t[c].fillna(0).to_numpy()
            ax.bar(x, v, bottom=bottom, label=c[4:], alpha=0.7)
            bottom = bottom + v
        ax.set_xticks(x); ax.set_xticklabels(t.group, rotation=20, fontsize=8); ax.set_title(f"{name} (D̂ ≤ {op:g})")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("bits")
    fig.tight_layout()
    return fig


def plot_distance(S: pd.DataFrame, Z: pd.DataFrame | None = None):
    """Symmetrised relative distance heatmap (and, if given, the zero-shot D̂ matrix clipped at 2)."""
    n = 2 if Z is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    axes = np.atleast_1d(axes)
    im = axes[0].imshow(S.values, cmap="viridis")
    axes[0].set_xticks(range(len(S))); axes[0].set_xticklabels(S.columns, rotation=60, fontsize=7)
    axes[0].set_yticks(range(len(S))); axes[0].set_yticklabels(S.index, fontsize=7)
    axes[0].set_title("symmetrised warm-start cost / scratch"); fig.colorbar(im, ax=axes[0], shrink=0.8)
    if Z is not None:
        im2 = axes[1].imshow(np.clip(Z.values, -0.5, 2.0), cmap="magma_r", vmin=-0.5, vmax=2.0)
        axes[1].set_xticks(range(len(Z))); axes[1].set_xticklabels(Z.columns, rotation=60, fontsize=7)
        axes[1].set_yticks(range(len(Z))); axes[1].set_yticklabels(Z.index, fontsize=7)
        axes[1].set_title("zero-shot D̂ of src prefix on dst (clipped at 2)"); fig.colorbar(im2, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    return fig


def plot_mechanism(profiles: dict[str, pd.DataFrame]):
    """One panel per probe: mean recovered fraction per block, one line per task."""
    fig, axes = plt.subplots(1, len(profiles), figsize=(4.2 * len(profiles), 3.4), sharey=True)
    for ax, (probe, prof) in zip(np.atleast_1d(axes), profiles.items()):
        for col in prof.columns:
            ax.plot(prof.index, prof[col], marker="o", ms=2.5, label=col)
        ax.axhline(0, color="grey", lw=0.5); ax.set_title(probe); ax.set_xlabel("block"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("fraction of prefix effect recovered"); np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_trajectories(trajs: dict[str, pd.DataFrame]):
    """KL bits (total for flat runs, centre KL for hierarchical ones) and dev bits/token vs step."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, t in trajs.items():
        kl = t.kl_bits if "kl_bits" in t else t.kl_centre_bits
        axes[0].plot(t.step, kl, marker=".", label=name + ("" if "kl_bits" in t else " [centre KL]"))
        axes[1].plot(t.step, t.dev_bits_per_token, marker=".", label=name)
    axes[0].set_yscale("log"); axes[0].set_ylabel("KL (bits; centre KL for hierarchical runs)"); axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("dev bits/token"); axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_compose(ho: pd.DataFrame):
    """Assembled vs trained-on D̂ per held-out cell."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ho = ho.sort_values(["cell", "holdout"]).reset_index(drop=True)
    x = np.arange(len(ho))
    ax.bar(x - 0.2, ho.D_assembled, 0.4, label="assembled (held out)")
    ax.bar(x + 0.2, ho.D_trained, 0.4, label="trained on")
    short = lambda c: c.replace("translation", "tr").replace("recall", "rc").replace("continuation", "ct")
    ax.set_xticks(x); ax.set_xticklabels([f"{short(c)} {int(kt)}+{int(kl)}\nheld out with {short(h).replace(short(c), '').strip('+')}"
                                          for c, h, kt, kl in zip(ho.cell, ho.holdout, ho.k_task, ho.k_lang)], fontsize=6, rotation=30, ha="right")
    ax.axhline(1.0, color="grey", lw=0.6, ls="--"); ax.set_ylabel("D̂"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
