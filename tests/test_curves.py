"""Slot complexity k* on the cell scale from load.attach_D."""
import numpy as np
import pandas as pd

from prefixlab.analysis import load
from prefixlab.analysis.curves import k_star


def _df():
    rows = [
        dict(cell="translation-ru", cond="none", design="none", k=0, seed=0, test_bits_per_token=4.0),
        dict(cell="translation-ru", cond="none", design="floor", k=0, seed=0, test_bits_per_token=2.0),
        dict(cell="translation-ru", cond="none", design="A", k=1, seed=0, test_bits_per_token=3.5),
        dict(cell="translation-ru", cond="none", design="A", k=2, seed=0, test_bits_per_token=2.5),
        dict(cell="translation-ru", cond="none", design="A", k=2, seed=1, test_bits_per_token=2.7),
        dict(cell="translation-ru", cond="none", design="A", k=4, seed=0, test_bits_per_token=2.1),
        dict(cell="recall-ru", cond="none", design="none", k=0, seed=0, test_bits_per_token=3.0),
        dict(cell="recall-ru", cond="none", design="A", k=1, seed=0, test_bits_per_token=2.0),  # no LoRA: best prefix is the floor
    ]
    return pd.DataFrame(rows)


def test_k_star_uses_seed_means_and_returns_none_when_unreached():
    df = load.attach_D(_df(), load.floors(_df()))
    assert df[(df.cell == "recall-ru") & (df.design == "A")].D.iloc[0] == 0.0
    assert k_star(df, "translation-ru", "none", 0.75) == 1
    assert k_star(df, "translation-ru", "none", 0.30) == 2   # seed mean 0.30 at k=2
    assert k_star(df, "translation-ru", "none", 0.10) == 4
    assert k_star(df, "translation-ru", "none", 0.01) is None
    assert k_star(df, "recall-ru", "none", 0.5) == 1
    assert k_star(df, "recall-de", "none", 0.5) is None
