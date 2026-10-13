"""Normalized distortion, slot complexity k*, and bootstrap intervals."""
import numpy as np
import pandas as pd
import pytest

from prefixlab.analysis.curves import attach_normalized_distortion, bootstrap_mean_ci, k_star, paired_bootstrap_ci


def _df():
    rows = [
        dict(cell="translation-ru", cond="none", design="none", k=0, seed=0, test_bits_per_token=4.0),
        dict(cell="translation-ru", cond="none", design="floor", k=0, seed=0, test_bits_per_token=2.0),
        dict(cell="translation-ru", cond="none", design="A", k=1, seed=0, test_bits_per_token=3.5),
        dict(cell="translation-ru", cond="none", design="A", k=2, seed=0, test_bits_per_token=2.5),
        dict(cell="translation-ru", cond="none", design="A", k=2, seed=1, test_bits_per_token=2.7),
        dict(cell="translation-ru", cond="none", design="A", k=4, seed=0, test_bits_per_token=2.1),
        dict(cell="recall-ru", cond="none", design="none", k=0, seed=0, test_bits_per_token=3.0),
        dict(cell="recall-ru", cond="none", design="A", k=1, seed=0, test_bits_per_token=2.0),  # no floor -> NaN
    ]
    return pd.DataFrame(rows)


def test_normalized_distortion_uses_cell_floor_and_no_prefix():
    df = attach_normalized_distortion(_df())
    a = df[(df.design == "A") & (df.cell == "translation-ru")].set_index(["k", "seed"])["D_hat"]
    assert a[(1, 0)] == pytest.approx(0.75)
    assert a[(2, 0)] == pytest.approx(0.25)
    assert a[(2, 1)] == pytest.approx(0.35)
    assert a[(4, 0)] == pytest.approx(0.05)
    assert np.isnan(df[(df.cell == "recall-ru") & (df.design == "A")]["D_hat"].iloc[0])


def test_k_star_is_smallest_k_whose_seed_mean_reaches_the_level():
    df = attach_normalized_distortion(_df())
    assert k_star(df, "translation-ru", "none", level=0.30) == 2  # mean(0.25, 0.35) = 0.30
    assert k_star(df, "translation-ru", "none", level=0.25) == 4
    assert k_star(df, "translation-ru", "none", level=0.01) is None


def test_bootstrap_intervals():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=400)
    lo, hi = bootstrap_mean_ci(x, n_boot=500, seed=0)
    assert lo < x.mean() < hi and hi - lo < 0.5
    lo2, hi2 = paired_bootstrap_ci(x, x, n_boot=200, seed=0)
    assert lo2 == pytest.approx(0.0) and hi2 == pytest.approx(0.0)
    lo3, hi3 = paired_bootstrap_ci(x + 1.0, x, n_boot=200, seed=0)
    assert lo3 == pytest.approx(1.0) and hi3 == pytest.approx(1.0)
