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


def frontier_by_seed(df: pd.DataFrame, ops, dcol: str = "D_sampled", cells=None) -> pd.DataFrame:
    """Per (cell, op): the frontier of each seed side by side, with the ratio seed 1 / seed 0. Each seed is an
    independent replicate of the whole run grid, so the per-seed frontier is the unit of replication."""
    seeds = sorted(df.seed.unique())
    parts = []
    for sd in seeds:
        fr = frontier(df[df.seed == sd], ops, dcol=dcol, cells=cells)[["cell", "op", "bits", "k", "beta", "D"]]
        parts.append(fr.rename(columns={c: f"{c}_s{sd}" for c in ("bits", "k", "beta", "D")}))
    out = parts[0]
    for p_ in parts[1:]:
        out = out.merge(p_, on=["cell", "op"], how="outer")
    if len(seeds) >= 2:
        out["ratio_s1_s0"] = out[f"bits_s{seeds[1]}"] / out[f"bits_s{seeds[0]}"]
    return out


def hier_totals(h: pd.DataFrame, ops, dcol: str = "D_sampled") -> pd.DataFrame:
    """Per (group, op): the cheapest run (k, beta) whose members all reach op; total = centre + sum of deviations."""
    rows = []
    for grp, g in h.groupby("group"):
        for op in ops:
            best = None
            n_members = g.member.nunique()
            for (k, beta), gg in g.groupby(["k", "beta"]):
                assert len(gg) == n_members, f"{grp} k={k} beta={beta}: {len(gg)} rows for {n_members} members"
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

    def covered(t: pd.DataFrame) -> set[str]:
        """Cells a hierarchy table covers: group 'task:X' + member 'y' -> 'X-y'; group 'lang:y' + member 'X' -> 'X-y'."""
        out = set()
        for r in t.itertuples():
            kind, name = r.group.split(":", 1)
            for col in t.columns:
                if col.startswith("dev_") and pd.notna(getattr(r, col)):
                    out.add(f"{name}-{col[4:]}" if kind == "task" else f"{col[4:]}-{name}")
        return out

    return pd.DataFrame([
        dict(scheme="task-centred hierarchy", bits=float(ht.total.sum()), complete=bool(ht.total.notna().all()) and covered(ht) >= set(cells)),
        dict(scheme="independent flat programs", bits=float(f.bits.sum()), complete=bool(f.bits.notna().all()) and len(f) == len(set(cells))),
        dict(scheme="language-centred hierarchy", bits=float(hl.total.sum()), complete=bool(hl.total.notna().all()) and covered(hl) >= set(cells)),
    ])


def group_vs_independent(h: pd.DataFrame, totals: pd.DataFrame, flat_front: pd.DataFrame, ops) -> pd.DataFrame:
    """Per (group, op): the hierarchy's cheapest total against the sum of the independent frontier over the same cells.
    `h` gives the group -> cell membership, `totals` comes from hier_totals, `flat_front` from frontier (any protocol mix)."""
    cells_of = h.groupby("group").cell.unique()
    rows = []
    for grp, cells in cells_of.items():
        for op in ops:
            t = totals[(totals.group == grp) & (totals.op == op)]
            f = flat_front[(flat_front.op == op) & flat_front.cell.isin(cells)]
            ind = float(f.bits.sum()) if len(f) == len(cells) and f.bits.notna().all() else np.nan
            hb = float(t.total.iloc[0]) if len(t) else np.nan
            rows.append(dict(group=grp, op=op, n_cells=len(cells), hier_bits=hb,
                             hier_k=float(t.k.iloc[0]) if len(t) else np.nan, hier_beta=float(t.beta.iloc[0]) if len(t) else np.nan,
                             independent_bits=ind, ratio=hb / ind if ind and not np.isnan(ind) else np.nan))
    return pd.DataFrame(rows)


def matched_hyperparameters(h: pd.DataFrame, flat: pd.DataFrame, dcol: str = "D_sampled", seed: int = 0) -> pd.DataFrame:
    """Per (group, k, beta): hierarchy total against the sum of the independent runs at the same (k, beta, seed) over the
    group's cells, with the worst member distortion of each. NaN when an independent run is missing."""
    rows = []
    for (grp, k, beta), g in h.groupby(["group", "k", "beta"]):
        f = flat[(flat.k == k) & (flat.beta == beta) & (flat.seed == seed) & flat.cell.isin(g.cell)]
        f = f.drop_duplicates("cell")
        have = len(f) == len(g)
        rows.append(dict(group=grp, k=int(k), beta=float(beta), n_cells=len(g),
                         hier_bits=float(g.kl_centre_bits.iloc[0] + g.kl_dev_bits.sum()), hier_maxD=float(g[dcol].max()),
                         independent_bits=float(f.kl_bits.sum()) if have else np.nan, independent_maxD=float(f[dcol].max()) if have else np.nan))
    out = pd.DataFrame(rows)
    out["ratio"] = out.hier_bits / out.independent_bits
    return out


def nestings_matched(hier_task: pd.DataFrame, hier_lang: pd.DataFrame, dcol: str = "D_sampled") -> pd.DataFrame:
    """Per (k, beta): total bits (sum over groups of centre + deviations) and worst member distortion under the two
    nestings, with the number of cells each covers (a nesting with a missing run covers fewer than all cells)."""
    def total(t: pd.DataFrame) -> float:
        return float(sum(g.kl_centre_bits.iloc[0] + g.kl_dev_bits.sum() for _, g in t.groupby("group")))
    keys = sorted(set(map(tuple, hier_task[["k", "beta"]].drop_duplicates().to_numpy())) | set(map(tuple, hier_lang[["k", "beta"]].drop_duplicates().to_numpy())))
    rows = []
    for k, beta in keys:
        gt = hier_task[(hier_task.k == k) & (hier_task.beta == beta)]
        gl = hier_lang[(hier_lang.k == k) & (hier_lang.beta == beta)]
        rows.append(dict(k=int(k), beta=float(beta),
                         task_bits=total(gt) if len(gt) else np.nan, task_maxD=float(gt[dcol].max()) if len(gt) else np.nan, task_cells=len(gt),
                         lang_bits=total(gl) if len(gl) else np.nan, lang_maxD=float(gl[dcol].max()) if len(gl) else np.nan, lang_cells=len(gl)))
    out = pd.DataFrame(rows)
    out["lang_over_task"] = out.lang_bits / out.task_bits
    return out


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
