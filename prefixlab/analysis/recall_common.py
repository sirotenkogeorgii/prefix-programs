"""Recall across languages on the uuids common to all three test sets (the mLAMA test sets differ in size and
coverage per language, which confounds a cross-language comparison of recall cost)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.mlama import cross_lingual_recall, find_mlama_root, split_by_uuid
from .load import load_per_example

LANG_MLAMA = {"ru": "ru", "de": "de", "en": "en"}


def test_uuids(mlama_root, lang: str, seed: int = 0) -> list[str]:
    rows = cross_lingual_recall(find_mlama_root(mlama_root), LANG_MLAMA[lang])
    _, _, test = split_by_uuid(rows, seed=seed)
    return [r["uuid"] for r in test]


def common_uuid_bits(results_dir, mlama_root, sweep: pd.DataFrame, langs=("ru", "de", "en"), k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Bits/token and D̂ for no-prefix, floor (min of LoRA and best Design A, on the common items) and the Design A k-slot prefix, on the common uuids only."""
    uuids = {l: test_uuids(mlama_root, l) for l in langs}
    common = set.intersection(*(set(u) for u in uuids.values()))
    rows = []
    for l in langs:
        cell = f"recall-{l}"
        idx = [i for i, u in enumerate(uuids[l]) if u in common]
        vals = {}
        def bits_on_common(r):
            pe = load_per_example(results_dir, r.per_example.iloc[0])
            assert len(pe["bits"]) == len(uuids[l]), f"{cell}: per-example file has {len(pe['bits'])} rows, test split {len(uuids[l])}"
            b, t = np.asarray(pe["bits"])[idx], np.asarray(pe["tokens"])[idx]
            return float(b.sum() / t.sum())
        here = sweep[(sweep.cell == cell) & (sweep.cond == "none")]
        vals["none"] = bits_on_common(here[here.design == "none"])
        vals["lora"] = bits_on_common(here[here.design == "floor"])
        a = here[here.design == "A"]
        vals["bestA"] = min(bits_on_common(a[a.run_id == rid]) for rid in a.run_id)
        vals["floor"] = min(vals["lora"], vals["bestA"])  # same rule as load.floors, on the common items
        vals["prefix"] = bits_on_common(a[(a.k == k) & (a.seed == seed)])
        rows.append(dict(cell=cell, n_test=len(uuids[l]), n_common=len(idx), **vals))
    df = pd.DataFrame(rows)
    df["D_prefix"] = (df.prefix - df.floor) / (df.none - df.floor)
    return df
