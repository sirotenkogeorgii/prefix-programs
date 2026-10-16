"""Normalized distortion D̂, slot complexity k*, and bootstrap intervals over test examples."""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_normalized_distortion(df: pd.DataFrame, col: str = "test_bits_per_token", out: str = "D_hat") -> pd.DataFrame:
    """D̂ = (L - L_floor) / (L_none - L_floor) per (cell, cond), using that cond's floor or, failing that, the
    cell's cond='none' floor. NaN where a floor or no-prefix row is missing."""
    df = df.copy()
    df[out] = np.nan
    for (cell, cond), g in df.groupby(["cell", "cond"]):
        none = g.loc[g.design == "none", col]
        floor = g.loc[g.design == "floor", col]
        if floor.empty:
            floor = df.loc[(df.cell == cell) & (df.cond == "none") & (df.design == "floor"), col]
        if none.empty or floor.empty:
            continue
        ln, lf = none.mean(), floor.mean()
        df.loc[g.index, out] = (g[col] - lf) / (ln - lf)
    return df


def k_star(df: pd.DataFrame, cell: str, cond: str, level: float, design: str = "A", col: str = "D_hat"):
    """Smallest k whose seed-mean D̂ is <= level, or None."""
    sub = df[(df.cell == cell) & (df.cond == cond) & (df.design == design)]
    if sub.empty:
        return None
    means = sub.groupby("k")[col].mean().sort_index()
    ok = means[means <= level + 1e-9]  # tolerate float noise at the boundary
    return int(ok.index.min()) if len(ok) else None


def bootstrap_mean_ci(x, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(a, b, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """CI for mean(a - b) resampling paired examples together."""
    return bootstrap_mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_boot, seed, alpha)


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
