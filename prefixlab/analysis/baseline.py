"""Zero-rate baselines on the per-cell distortion scale: untrained prefixes (prior draws, random token rows) against the
floor and no-prefix loss of each cell."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def baseline_distortion(b: pd.DataFrame, fl: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, kind, k): D̂ of the untrained prefix, mean over draws on the cell's scale, with the sd, min and max of
    the per-draw D̂. NaN where the cell has no floor or a zero gap."""
    rows = []
    for r in b.itertuples():
        f = fl[fl.cell == r.cell]
        if f.empty:
            continue
        f = f.iloc[0]
        scale = float(f.L_none - f.L_floor)
        draws = json.loads(r.test_bits_draws) if isinstance(r.test_bits_draws, str) else []
        D = [(x - f.L_floor) / scale for x in draws] if scale else []
        rows.append(dict(cell=r.cell, kind=r.kind, k=int(r.k), n_draws=len(draws),
                         D_mean=(r.test_bits_per_token - f.L_floor) / scale if scale else np.nan,
                         D_sd=float(np.std(D, ddof=1)) if len(D) > 1 else np.nan,
                         D_min=min(D) if D else np.nan, D_max=max(D) if D else np.nan))
    return pd.DataFrame(rows)
