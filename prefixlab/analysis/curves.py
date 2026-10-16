"""Slot complexity k*, conditioning savings, and the natural-language-prompt comparison (D̂ comes from load.attach_D)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def k_star(df: pd.DataFrame, cell: str, cond: str, level: float, design: str = "A", col: str = "D"):
    """Smallest k whose seed-mean D̂ is <= level, or None."""
    sub = df[(df.cell == cell) & (df.cond == cond) & (df.design == design)]
    if sub.empty:
        return None
    means = sub.groupby("k")[col].mean().sort_index()
    ok = means[means <= level + 1e-9]  # tolerate float noise at the boundary
    return int(ok.index.min()) if len(ok) else None


def k_star_table(sweepD: pd.DataFrame, levels, conds=("none", "lang", "task"), cells=None, col: str = "D") -> pd.DataFrame:
    """k* per (cell, cond, level) from the seed-mean D̂ of Design A runs; NaN when no k reaches the level."""
    cells = cells or sorted(sweepD.cell.unique())
    rows = []
    for cell in cells:
        for cond in conds:
            r = dict(cell=cell, cond=cond)
            for lv in levels:
                ks = k_star(sweepD, cell, cond, lv, col=col)
                r[f"k*({lv:g})"] = ks if ks is not None else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def savings_table(kstar: pd.DataFrame, level: float) -> pd.DataFrame:
    """Δk_lang = k*(none) - k*(lang) and Δk_task = k*(none) - k*(task) per cell at one level (H2)."""
    col = f"k*({level:g})"
    w = kstar.pivot(index="cell", columns="cond", values=col)
    out = pd.DataFrame({"k_none": w.get("none"), "k_lang": w.get("lang"), "k_task": w.get("task")})
    out["dk_lang"] = out.k_none - out.k_lang
    out["dk_task"] = out.k_none - out.k_task
    return out.reset_index()


def nl_prompt_table(sweepD: pd.DataFrame, col: str = "D") -> pd.DataFrame:
    """Natural-language prompt (cond=both, no prefix) vs one trained slot (cond=none, k=1), D̂ per cell."""
    nl = sweepD[(sweepD.design == "none") & (sweepD.cond == "both")].groupby("cell")[col].mean()
    k1 = sweepD[(sweepD.design == "A") & (sweepD.cond == "none") & (sweepD.k == 1)].groupby("cell")[col].mean()
    k1_both = sweepD[(sweepD.design == "A") & (sweepD.cond == "both") & (sweepD.k == 1)].groupby("cell")[col].mean()
    df = pd.DataFrame({"D_nl_prompt": nl, "D_one_slot": k1})
    if len(k1_both):
        df["D_nl_prompt_plus_one_slot"] = k1_both
    return df.reset_index()


def seed_gap_table(sweepD: pd.DataFrame, col: str = "D") -> pd.DataFrame:
    """Max |D̂(seed 0) - D̂(seed 1)| per cell over k, Design A cond=none."""
    a = sweepD[(sweepD.design == "A") & (sweepD.cond == "none")]
    w = a.pivot_table(index=["cell", "k"], columns="seed", values=col)
    if w.shape[1] < 2:
        return pd.DataFrame(columns=["cell", "max_seed_gap"])
    gap = (w.iloc[:, 0] - w.iloc[:, 1]).abs().groupby("cell").max()
    return gap.rename("max_seed_gap").reset_index()
