"""Bootstrap intervals over test examples for bits/token ratios and their differences."""
from __future__ import annotations

import numpy as np


def ratio_ci(bits, tokens, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi) for sum(bits)/sum(tokens) resampling examples."""
    bits, tokens = np.asarray(bits, float), np.asarray(tokens, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bits), size=(n_boot, len(bits)))
    r = bits[idx].sum(1) / tokens[idx].sum(1)
    return float(bits.sum() / tokens.sum()), float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def paired_ratio_diff_ci(bits_a, bits_b, tokens, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi) for bits/token(a) - bits/token(b) on the same examples, resampled together."""
    a, b, t = (np.asarray(x, float) for x in (bits_a, bits_b, tokens))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].sum(1) / t[idx].sum(1) - b[idx].sum(1) / t[idx].sum(1)
    return float(a.sum() / t.sum() - b.sum() / t.sum()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
