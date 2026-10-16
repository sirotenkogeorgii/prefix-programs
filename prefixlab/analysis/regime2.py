"""Regime 2 (exploratory): how many per-example input vectors replace the real input, with and without the task prefix."""
from __future__ import annotations

import numpy as np
import pandas as pd


def m_star_table(r: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    """Per (cell, k): smallest m whose bits/token after optimisation is <= frac * bits/token with the real input."""
    rows = []
    for (cell, k), g in r.groupby(["cell", "k"]):
        g = g.sort_values("m")
        thr = frac * float(g.bits_per_token_real_input.iloc[0])
        ok = g[g.bits_per_token_after <= thr]
        rows.append(dict(cell=cell, k=int(k), real_input=float(g.bits_per_token_real_input.iloc[0]),
                         no_input=float(g.bits_per_token_no_input.iloc[0]), threshold=thr,
                         m_star=int(ok.m.min()) if len(ok) else np.nan,
                         **{f"after_m{int(x.m)}": float(x.bits_per_token_after) for x in g.itertuples()}))
    return pd.DataFrame(rows)
