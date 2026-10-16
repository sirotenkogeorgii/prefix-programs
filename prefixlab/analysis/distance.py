"""Distance matrix analysis: warm-start learning cost, symmetrised relative distance, task vs language clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .load import lang_of, task_of


def cost_matrix(d: pd.DataFrame, col: str = "learning_cost_bits", cells=None, scratch: str = "scratch"):
    cells = cells or sorted(d.dst.unique())
    M = d[d.src != scratch].pivot(index="src", columns="dst", values=col).reindex(index=cells, columns=cells)
    s = d[d.src == scratch].set_index("dst")[col].reindex(cells)
    return M, s


def relative_cost(M: pd.DataFrame, scratch: pd.Series) -> pd.DataFrame:
    """cost(src -> dst) / cost(scratch -> dst)."""
    return M.div(scratch, axis=1)


def symmetrised_distance(M: pd.DataFrame, scratch: pd.Series) -> pd.DataFrame:
    """d(a, b) = (cost(a->b) + cost(b->a)) / (scratch(a) + scratch(b)); zero diagonal."""
    S = (M.values + M.values.T) / (scratch.values[:, None] + scratch.values[None, :])
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, 0.0)
    return pd.DataFrame(S, index=M.index, columns=M.columns)


def pair_group_means(S: pd.DataFrame) -> pd.DataFrame:
    cells = list(S.index)
    buckets = {"same task": [], "same language": [], "neither": []}
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            key = "same task" if task_of(a) == task_of(b) else "same language" if lang_of(a) == lang_of(b) else "neither"
            buckets[key].append(S.iloc[i, j])
    return pd.DataFrame([dict(pairs=k, mean=float(np.mean(v)), n=len(v), sd=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan)
                         for k, v in buckets.items()])


def clustering_silhouettes(S: pd.DataFrame) -> pd.DataFrame:
    """Silhouette of the task partition vs the language partition on the symmetrised distance (H4)."""
    D = S.values.copy()
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    out = []
    for name, fn in (("task", task_of), ("language", lang_of)):
        labels = [fn(c) for c in S.index]
        out.append(dict(partition=name, silhouette=float(silhouette_score(D, labels, metric="precomputed"))))
    return pd.DataFrame(out)


def zero_shot_matrix(d: pd.DataFrame, fl: pd.DataFrame, cells=None, scratch: str = "scratch") -> pd.DataFrame:
    """D̂ of the src prefix applied to dst with no training."""
    cells = cells or sorted(d.dst.unique())
    key = fl.set_index("cell")
    z = d[d.src != scratch].copy()
    z["Dz"] = [(r.zero_shot_test_bits_per_token - key.loc[r.dst, "L_floor"]) / (key.loc[r.dst, "L_none"] - key.loc[r.dst, "L_floor"])
               for r in z.itertuples()]
    return z.pivot(index="src", columns="dst", values="Dz").reindex(index=cells, columns=cells)
