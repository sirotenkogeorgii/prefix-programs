"""Rate–distortion frontiers: minimum code length (variational KL bits) that reaches an operating point."""
from __future__ import annotations

import numpy as np
import pandas as pd


def frontier(df: pd.DataFrame, ops, dcol: str = "D_sampled", cells=None) -> pd.DataFrame:
    """Per (cell, op): min kl_bits among runs with dcol <= op; NaN bits when no run reaches it."""
    rows = []
    cells = cells or sorted(df.cell.unique())
    for cell in cells:
        g = df[df.cell == cell]
        for op in ops:
            q = g[g[dcol] <= op + 1e-9]
            if q.empty:
                rows.append(dict(cell=cell, op=op, bits=np.nan, k=np.nan, beta=np.nan, D=np.nan,
                                 best_D=float(g[dcol].min()) if len(g) else np.nan))
            else:
                r = q.loc[q.kl_bits.idxmin()]
                rows.append(dict(cell=cell, op=op, bits=float(r.kl_bits), k=int(r.k), beta=float(r.beta),
                                 D=float(r[dcol]), best_D=float(g[dcol].min())))
    return pd.DataFrame(rows)


def hier_totals(h: pd.DataFrame, ops, dcol: str = "D_sampled") -> pd.DataFrame:
    """Per (group, op): the cheapest run (k, beta) whose members all reach op; total = centre + sum of deviations."""
    rows = []
    for grp, g in h.groupby("group"):
        for op in ops:
            best = None
            for (k, beta), gg in g.groupby(["k", "beta"]):
                if (gg[dcol] <= op + 1e-9).all():
                    tot = float(gg.kl_centre_bits.iloc[0] + gg.kl_dev_bits.sum())
                    if best is None or tot < best["total"]:
                        best = dict(group=grp, op=op, k=int(k), beta=float(beta), centre=float(gg.kl_centre_bits.iloc[0]),
                                    total=tot, n_members=len(gg),
                                    **{f"dev_{m}": float(v) for m, v in zip(gg.member, gg.kl_dev_bits)})
            rows.append(best or dict(group=grp, op=op, k=np.nan, beta=np.nan, centre=np.nan, total=np.nan, n_members=len(g.member.unique())))
    return pd.DataFrame(rows)


def scheme_comparison(flat_front: pd.DataFrame, hier_task: pd.DataFrame, hier_lang: pd.DataFrame, op: float, cells) -> pd.DataFrame:
    """Total bits over `cells` at `op` under three codes: task-centred hierarchy, independent flat programs,
    language-centred hierarchy. Groups whose members are not all in `cells` are summed over their members anyway
    (the member set is fixed by the run), so pass the full cell list for the headline comparison."""
    f = flat_front[(flat_front.op == op) & flat_front.cell.isin(cells)]
    ht = hier_task[hier_task.op == op]
    hl = hier_lang[hier_lang.op == op]
    return pd.DataFrame([
        dict(scheme="task-centred hierarchy", bits=float(ht.total.sum()), complete=bool(ht.total.notna().all())),
        dict(scheme="independent flat programs", bits=float(f.bits.sum()), complete=bool(f.bits.notna().all())),
        dict(scheme="language-centred hierarchy", bits=float(hl.total.sum()), complete=bool(hl.total.notna().all())),
    ])


def dev_trajectory(history: list[dict]) -> pd.DataFrame:
    """(step, dev_bits_per_token, kl_bits[, dev_objective, kl_centre_bits, kl_dev_bits...]) as a frame."""
    rows = []
    for e in history:
        r = {k: v for k, v in e.items() if not isinstance(v, dict)}
        if isinstance(e.get("kl_dev_bits"), dict):
            for m, v in e["kl_dev_bits"].items():
                r[f"kl_dev_{m}"] = v
        if isinstance(e.get("per_member"), dict):
            for m, v in e["per_member"].items():
                r[f"dev_{m}"] = v
        rows.append(r)
    return pd.DataFrame(rows)
