"""Composition analysis: assembled (held-out) vs trained-on distortion of tied task + language parts."""
from __future__ import annotations

import pandas as pd


def holdout_table(c: pd.DataFrame, sweepD: pd.DataFrame, k_ref: int = 4) -> pd.DataFrame:
    """One row per held-out cell: D̂ of the zero-shot assembled prefix, D̂ when trained on (holdout=none, same slots),
    and the Design A reference at k_ref (seed mean)."""
    ref = sweepD[(sweepD.design == "A") & (sweepD.cond == "none") & (sweepD.k == k_ref)].groupby("cell").D.mean()
    rows = []
    for r in c[c.held_out.astype(bool)].itertuples():
        trained = c[(c.holdout == "none") & (c.k_task == r.k_task) & (c.k_lang == r.k_lang) & (c.cell == r.cell)]
        rows.append(dict(cell=r.cell, k_task=r.k_task, k_lang=r.k_lang, holdout=r.holdout, D_assembled=float(r.D),
                         D_trained=float(trained.D.iloc[0]) if len(trained) else float("nan"),
                         D_designA=float(ref.get(r.cell, float("nan")))))
    df = pd.DataFrame(rows)
    if len(df):
        df["gap_closed"] = 1.0 - df.D_assembled  # fraction of the no-prefix -> floor gap closed without training on the cell
    return df


def shared_vs_flat(c: pd.DataFrame, sweepD: pd.DataFrame) -> pd.DataFrame:
    """Trained-on-all composition at k_task + k_lang slots vs Design A at the same total k (seed mean), per cell."""
    rows = []
    for r in c[(c.holdout == "none")].itertuples():
        k_tot = int(r.k_task + r.k_lang)
        a = sweepD[(sweepD.design == "A") & (sweepD.cond == "none") & (sweepD.k == k_tot) & (sweepD.cell == r.cell)].D.mean()
        rows.append(dict(cell=r.cell, k_task=r.k_task, k_lang=r.k_lang, D_composed=float(r.D), D_designA_same_k=float(a)))
    return pd.DataFrame(rows)
