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
