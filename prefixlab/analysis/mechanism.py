"""Mechanism analysis: fraction of a prefix's effect recovered by patching its positions into one block."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import task_of

EARLY, MID, LATE = (0, 3), (4, 13), (14, 27)


def recovery(m: pd.DataFrame, min_effect: float = 0.1) -> pd.DataFrame:
    """Adds effect = recipient - donor (bits/token) and frac = recovered / effect; frac is NaN when effect < min_effect,
    because a probe whose donor is not clearly better than its recipient has nothing to recover."""
    m = m.copy()
    m["effect"] = m.recipient_bits_per_token - m.donor_bits_per_token
    m["frac"] = np.where(m.effect >= min_effect, m.recovered_bits_per_token / m.effect, np.nan)
    m["task"] = m.cell.map(task_of)
    return m


def sanity(m: pd.DataFrame) -> pd.DataFrame:
    """all-layers patch must reproduce the donor exactly."""
    g = m.groupby(["cell", "probe"]).agg(recipient=("recipient_bits_per_token", "first"), donor=("donor_bits_per_token", "first"),
                                         all_layers=("all_layers_bits_per_token", "first")).reset_index()
    g["all_minus_donor"] = g.all_layers - g.donor
    return g


def layer_profile(m: pd.DataFrame, probe: str = "A", by: str = "task") -> pd.DataFrame:
    """Mean frac per layer, one column per `by` value (cells with NaN frac are dropped)."""
    g = m[(m.probe == probe) & m.frac.notna()]
    return g.pivot_table(index="layer", columns=by, values="frac", aggfunc="mean")


def depth_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Per (probe, task): peak block, frac at peak, and summed frac over early/mid/late blocks."""
    rows = []
    for (probe, task), g in m[m.frac.notna()].groupby(["probe", "task"]):
        prof = g.groupby("layer").frac.mean()
        rows.append(dict(probe=probe, task=task, n_cells=g.cell.nunique(), effect=float(g.effect.mean()),
                         peak_block=int(prof.idxmax()), peak_frac=float(prof.max()), sum_frac=float(prof.sum()),
                         early=float(prof.loc[EARLY[0]:EARLY[1]].sum()), mid=float(prof.loc[MID[0]:MID[1]].sum()),
                         late=float(prof.loc[LATE[0]:LATE[1]].sum())))
    return pd.DataFrame(rows)


def peak_blocks(m: pd.DataFrame) -> pd.DataFrame:
    """H5: per cell, the block of maximal recovery for each probe (NaN when the probe's effect is too small)."""
    rows = []
    for (cell, probe), g in m.groupby(["cell", "probe"]):
        if g.frac.notna().any():
            rows.append(dict(cell=cell, probe=probe, peak_block=int(g.loc[g.frac.idxmax(), "layer"]), effect=float(g.effect.iloc[0])))
        else:
            rows.append(dict(cell=cell, probe=probe, peak_block=np.nan, effect=float(g.effect.iloc[0])))
    return pd.DataFrame(rows).pivot(index="cell", columns="probe", values="peak_block")
