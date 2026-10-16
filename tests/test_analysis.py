"""Analysis modules on small synthetic tables, plus a smoke run on the synced results when present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prefixlab.analysis import bootstrap, compose, curves, distance, frontier, load, mechanism, regime2


def _sweep():
    rows = [
        dict(cell="translation-ru", cond="none", design="none", k=0, seed=0, test_bits_per_token=4.0),
        dict(cell="translation-ru", cond="none", design="floor", k=0, seed=0, test_bits_per_token=2.0),
        dict(cell="translation-ru", cond="none", design="A", k=1, seed=0, test_bits_per_token=3.0),
        dict(cell="translation-ru", cond="none", design="A", k=2, seed=0, test_bits_per_token=1.8),  # beats LoRA -> floor
        dict(cell="translation-ru", cond="none", design="A", k=2, seed=1, test_bits_per_token=2.2),
        dict(cell="translation-ru", cond="lang", design="none", k=0, seed=0, test_bits_per_token=3.5),
        dict(cell="translation-ru", cond="lang", design="floor", k=0, seed=0, test_bits_per_token=2.0),
        dict(cell="translation-ru", cond="lang", design="A", k=1, seed=0, test_bits_per_token=2.1),
        dict(cell="translation-ru", cond="both", design="none", k=0, seed=0, test_bits_per_token=3.0),
    ]
    return pd.DataFrame(rows)


def test_floors_take_min_of_lora_and_best_prefix_and_D_uses_them():
    fl = load.floors(_sweep())
    none = fl.iloc[0]
    assert len(fl) == 1 and none.L_lora == 2.0 and none.L_bestA == 1.8 and none.L_floor == 1.8
    sw = load.attach_D(_sweep(), fl)
    a = sw[(sw.design == "A") & (sw.cond == "none")].set_index(["k", "seed"]).D
    assert a[(1, 0)] == pytest.approx((3.0 - 1.8) / (4.0 - 1.8))
    assert a[(2, 0)] == pytest.approx(0.0)
    # every cond is measured on the cell's cond=none scale
    both = sw[(sw.cond == "both")].D.iloc[0]
    assert both == pytest.approx((3.0 - 1.8) / (4.0 - 1.8))


def test_k_star_savings_and_nl_prompt_tables():
    sw = load.attach_D(_sweep(), load.floors(_sweep()))
    ks = curves.k_star_table(sw, [0.25, 0.10], conds=("none", "lang"))
    none = ks[ks.cond == "none"].iloc[0]
    assert none["k*(0.25)"] == 2 and none["k*(0.1)"] == 2
    lang = ks[ks.cond == "lang"].iloc[0]
    assert lang["k*(0.25)"] == 1
    sav = curves.savings_table(ks, 0.25)
    assert sav.dk_lang.iloc[0] == 1
    nl = curves.nl_prompt_table(sw)
    assert nl.D_nl_prompt.iloc[0] == pytest.approx((3.0 - 1.8) / 2.2) and nl.D_one_slot.iloc[0] == pytest.approx(1.2 / 2.2)


def test_frontier_and_hier_totals_and_scheme_comparison():
    f = pd.DataFrame([
        dict(cell="c1", k=2, beta=1.0, kl_bits=1000, D_sampled=0.08),
        dict(cell="c1", k=4, beta=3.0, kl_bits=600, D_sampled=0.12),
        dict(cell="c1", k=8, beta=3.0, kl_bits=400, D_sampled=0.30),
        dict(cell="c2", k=2, beta=1.0, kl_bits=2000, D_sampled=0.20),
    ])
    fr = frontier.frontier(f, [0.25, 0.10], cells=["c1", "c2"]).set_index(["cell", "op"])
    assert fr.loc[("c1", 0.25), "bits"] == 600 and fr.loc[("c1", 0.10), "bits"] == 1000
    assert fr.loc[("c2", 0.25), "bits"] == 2000 and np.isnan(fr.loc[("c2", 0.10), "bits"])
    h = pd.DataFrame([
        dict(group="task:t", k=2, beta=1.0, member="ru", cell="c1", kl_centre_bits=500, kl_dev_bits=100, D_sampled=0.05),
        dict(group="task:t", k=2, beta=1.0, member="de", cell="c2", kl_centre_bits=500, kl_dev_bits=150, D_sampled=0.20),
        dict(group="task:t", k=4, beta=3.0, member="ru", cell="c1", kl_centre_bits=300, kl_dev_bits=50, D_sampled=0.05),
        dict(group="task:t", k=4, beta=3.0, member="de", cell="c2", kl_centre_bits=300, kl_dev_bits=60, D_sampled=0.30),
    ])
    tot = frontier.hier_totals(h, [0.25, 0.10]).set_index("op")
    assert tot.loc[0.25, "total"] == 750 and tot.loc[0.25, "k"] == 2  # k4 fails on de at 0.25
    assert np.isnan(tot.loc[0.10, "total"])
    cmp = frontier.scheme_comparison(fr.reset_index(), tot.reset_index(), tot.reset_index(), 0.25, ["c1", "c2"]).set_index("scheme")
    assert cmp.loc["independent flat programs", "bits"] == 2600 and cmp.loc["task-centred hierarchy", "bits"] == 750


def test_distance_symmetrisation_groups_and_silhouette():
    cells = ["translation-ru", "translation-de", "recall-ru", "recall-de"]
    rows = []
    for s in cells + ["scratch"]:
        for d in cells:
            same_task = s != "scratch" and s.split("-")[0] == d.split("-")[0]
            rows.append(dict(src=s, dst=d, learning_cost_bits=100 if s == "scratch" else (20 if s == d else (50 if same_task else 95)),
                             zero_shot_test_bits_per_token=3.0))
    d = pd.DataFrame(rows)
    M, scratch = distance.cost_matrix(d, cells=cells)
    S = distance.symmetrised_distance(M, scratch)
    assert S.loc["translation-ru", "translation-de"] == pytest.approx(0.5) and S.loc["translation-ru", "recall-ru"] == pytest.approx(0.95)
    assert (np.diag(S.values) == 0).all()
    g = distance.pair_group_means(S).set_index("pairs")
    assert g.loc["same task", "mean"] == pytest.approx(0.5) and g.loc["same language", "mean"] == pytest.approx(0.95)
    sil = distance.clustering_silhouettes(S).set_index("partition").silhouette
    assert sil["task"] > sil["language"]
    fl = pd.DataFrame([dict(cell=c, L_none=4.0, L_floor=2.0) for c in cells])
    Z = distance.zero_shot_matrix(d, fl, cells=cells)
    assert Z.loc["translation-ru", "recall-de"] == pytest.approx(0.5)


def test_mechanism_recovery_fractions_and_depth_summary():
    rows = []
    for cell, eff in (("translation-ru", 1.0), ("recall-ru", 2.0), ("continuation-ru", 0.01)):
        for layer in range(28):
            rec = eff * (0.5 if layer == 0 else 0.1 if layer == 17 else 0.0)
            rows.append(dict(cell=cell, probe="A", layer=layer, recipient_bits_per_token=3.0 + eff, donor_bits_per_token=3.0,
                             all_layers_bits_per_token=3.0, recovered_bits_per_token=rec))
    m = mechanism.recovery(pd.DataFrame(rows))
    assert m[(m.cell == "continuation-ru")].frac.isna().all()  # effect below threshold
    assert m[(m.cell == "recall-ru") & (m.layer == 0)].frac.iloc[0] == pytest.approx(0.5)
    assert (mechanism.sanity(m).all_minus_donor == 0).all()
    ds = mechanism.depth_summary(m).set_index("task")
    assert ds.loc["recall", "peak_block"] == 0 and ds.loc["recall", "late"] == pytest.approx(0.1) and "continuation" not in ds.index
    pk = mechanism.peak_blocks(m)
    assert pk.loc["translation-ru", "A"] == 0 and np.isnan(pk.loc["continuation-ru", "A"])


def test_compose_holdout_table_and_regime2_m_star():
    c = pd.DataFrame([
        dict(cell="translation-ru", k_task=2, k_lang=2, holdout="none", held_out=False, D=0.02),
        dict(cell="translation-ru", k_task=2, k_lang=2, holdout="translation-ru+recall-de", held_out=True, D=0.30),
        dict(cell="recall-de", k_task=2, k_lang=2, holdout="translation-ru+recall-de", held_out=True, D=0.70),
        dict(cell="recall-de", k_task=2, k_lang=2, holdout="none", held_out=False, D=0.15),
    ])
    sw = pd.DataFrame([dict(cell="translation-ru", cond="none", design="A", k=4, seed=0, D=0.01),
                       dict(cell="recall-de", cond="none", design="A", k=4, seed=0, D=0.08)])
    ho = compose.holdout_table(c, sw).set_index("cell")
    assert ho.loc["translation-ru", "D_assembled"] == 0.30 and ho.loc["translation-ru", "D_trained"] == 0.02
    assert ho.loc["recall-de", "D_designA_k4"] == 0.08 and ho.loc["recall-de", "gap_closed"] == pytest.approx(0.30)
    r = pd.DataFrame([dict(cell="translation-ru", k=0, m=m, bits_per_token_after=a, bits_per_token_real_input=2.0, bits_per_token_no_input=3.5)
                      for m, a in ((1, 1.2), (2, 0.15), (4, 0.01))])
    ms = regime2.m_star_table(r, frac=0.1)
    assert ms.m_star.iloc[0] == 2 and ms.threshold.iloc[0] == pytest.approx(0.2)


def test_bootstrap_ratio_cis_cover_point_estimate():
    rng = np.random.default_rng(0)
    bits, tokens = rng.uniform(1, 5, 200), rng.integers(2, 6, 200).astype(float)
    pt, lo, hi = bootstrap.ratio_ci(bits, tokens, n_boot=300)
    assert lo <= pt <= hi
    pt2, lo2, hi2 = bootstrap.paired_ratio_diff_ci(bits, bits * 0.9, tokens, n_boot=300)
    assert lo2 <= pt2 <= hi2 and pt2 > 0 and lo2 > 0


@pytest.mark.skipif(not Path("results_sync/sweep.csv").exists(), reason="synced results not present")
def test_pipeline_runs_on_synced_results():
    T = load.load_tables("results_sync")
    fl = load.floors(T["sweep"])
    sw = load.attach_D(T["sweep"], fl)
    assert sw.D.notna().sum() > 100
    f2 = load.attach_D(T["flat2"], fl, "test_bits_per_token_sampled", "D_sampled")
    fr = frontier.frontier(f2, [0.25, 0.10], cells=load.MEASURABLE)
    assert len(fr) == 2 * len(load.MEASURABLE)
    ht = load.attach_D(T["hier_task"], fl, "test_bits_per_token_sampled", "D_sampled")
    assert frontier.hier_totals(ht, [0.25]).total.notna().any()
    M, s = distance.cost_matrix(T["distance"], cells=load.CELLS)
    assert distance.clustering_silhouettes(distance.symmetrised_distance(M, s)).shape[0] == 2
    assert mechanism.depth_summary(mechanism.recovery(T["mechanism"])).shape[0] > 0
