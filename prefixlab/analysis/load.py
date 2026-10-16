"""Load result tables, floors and normalized distortion for the analysis notebook.

Floors follow the ledger ruling: one scale per cell from its cond="none" rows, L_floor = min(LoRA floor, best
Design A prefix); every conditioning is measured on that scale. D̂ = (L - L_floor) / (L_none - L_floor).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ["sweep", "flat", "flat2", "hier_task", "hier_lang", "hier2_task", "hier2_lang", "compose", "distance", "regime2", "mechanism", "mechanism2", "recall_long"]
CELLS = ["translation-ru", "translation-de", "recall-ru", "recall-de", "recall-en",
         "continuation-ru", "continuation-de", "continuation-en"]
MEASURABLE = ["translation-ru", "translation-de", "recall-ru", "recall-de", "recall-en"]  # continuation: floor ≈ no-prefix


def task_of(cell: str) -> str:
    return cell.split("-")[0]


def lang_of(cell: str) -> str:
    return cell.split("-")[1]


def load_tables(results_dir) -> dict[str, pd.DataFrame]:
    d = Path(results_dir)
    out = {}
    for name in TABLES:
        p = d / f"{name}.csv"
        out[name] = pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()
    return out


def load_prereg(path="prereg.json") -> dict:
    return json.loads(Path(path).read_text())


def floors(sweep: pd.DataFrame) -> pd.DataFrame:
    """One row per cell, on the cell's own scale (cond='none'): L_none, L_lora, L_bestA, L_floor = min(L_lora, L_bestA).
    Conditioning runs are measured on the same scale, so D̂ can be slightly below 0 when a conditioned run beats the floor."""
    rows = []
    for cell, g in sweep[sweep.cond == "none"].groupby("cell"):
        none = g.loc[g.design == "none", "test_bits_per_token"]
        lora = g.loc[g.design == "floor", "test_bits_per_token"]
        best = g.loc[g.design == "A", "test_bits_per_token"]
        if none.empty:
            continue
        l_lora = float(lora.mean()) if len(lora) else np.nan
        l_best = float(best.min()) if len(best) else np.nan
        l_floor = np.nan if (np.isnan(l_lora) and np.isnan(l_best)) else float(np.nanmin([l_lora, l_best]))
        rows.append(dict(cell=cell, L_none=float(none.mean()), L_lora=l_lora, L_bestA=l_best, L_floor=l_floor))
    return pd.DataFrame(rows)


def attach_D(df: pd.DataFrame, fl: pd.DataFrame, col: str = "test_bits_per_token", out: str = "D") -> pd.DataFrame:
    """Add column `out` = D̂ of `col` on the cell's scale; NaN where the cell has no floor or no no-prefix row."""
    df = df.copy()
    df[out] = np.nan
    if df.empty:
        return df
    if col not in df.columns:
        raise KeyError(f"attach_D: column {col!r} not in table (columns: {list(df.columns)[:8]}...)")
    key = fl.set_index("cell")
    vals = []
    for cell, L in zip(df["cell"], df[col]):
        if cell not in key.index:
            vals.append(np.nan)
            continue
        f = key.loc[cell]
        vals.append(np.nan if (np.isnan(f.L_floor) or f.L_none == f.L_floor) else (L - f.L_floor) / (f.L_none - f.L_floor))
    df[out] = vals
    return df


def history_path(results_dir, stored: str) -> Path:
    """Rows store 'results/history/<run_id>.json' from Colab; map onto the local results dir."""
    p = Path(stored)
    return Path(results_dir) / "history" / p.name


def load_history(results_dir, stored: str) -> list[dict]:
    p = history_path(results_dir, stored)
    return json.loads(p.read_text()) if p.exists() else []


def per_example_path(results_dir, stored: str) -> Path:
    return Path(results_dir) / "per_example" / Path(stored).name


def load_per_example(results_dir, stored: str) -> dict:
    p = per_example_path(results_dir, stored)
    return json.loads(p.read_text()) if p.exists() else {}
