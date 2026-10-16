"""Design flat (variational, global prior) through run_one, and the hierarchical prefix."""
import json
import math
from pathlib import Path

import pytest
import torch

from prefixlab.data.formats import Cond, build_example
from prefixlab.prefix import HierarchicalPrefix, gaussian_prior_from_embeddings
from prefixlab.results import ResultsTable
from prefixlab.sweep import RunSpec, run_one
from prefixlab.tasks import Cell
from prefixlab.train import TrainCfg

PAIRS = [
    ("The cat sat on the mat.", "Кошка сидела на коврике."),
    ("I like green tea.", "Я люблю зелёный чай."),
    ("Where is the station?", "Где находится вокзал?"),
    ("It is raining today.", "Сегодня идёт дождь."),
]


def _data(tok):
    exs = [build_example(tok, x, y, Cond()) for x, y in PAIRS]
    return {"train": exs, "dev": exs[:2], "test": exs[2:]}


def test_flat_variational_run_records_kl_and_sampled_bits(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "sweep.csv")
    spec = RunSpec(cell=Cell("translation", "ru"), k=2, seed=0, cond="none", design="flat", beta=1.0)
    cfg = TrainCfg(steps=4, batch_size=4, lr=0.05, warmup=1, eval_every=2, seed=0)
    row = run_one(tiny_model, tok, spec, _data(tok), cfg, results, tmp_path / "ckpt")
    assert row["design"] == "flat" and row["beta"] == 1.0
    assert row["kl_bits"] > 0 and math.isfinite(row["kl_bits"])
    assert row["test_bits_per_token_sampled"] > 0
    assert results.is_done(cell="translation-ru", k=2, seed=0, cond="none", design="flat", beta=1.0)
    # a different beta is a different run
    assert not results.is_done(cell="translation-ru", k=2, seed=0, cond="none", design="flat", beta=0.1)


def test_flat2_selects_by_dev_objective_with_sampled_dev(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "flat2.csv")
    spec = RunSpec(Cell("translation", "ru"), k=2, seed=0, cond="none", design="flat2", beta=3.0)
    cfg = TrainCfg(steps=6, batch_size=4, lr=0.01, warmup=1, eval_every=2, seed=0)
    row = run_one(tiny_model, tok, spec, _data(tok), cfg, results, tmp_path / "ckpt")
    assert row["design"] == "flat2" and row["select"] == "objective" and row["dev_sample"] and row["lr"] == 0.01
    assert row["run_id"] == "flat2_translation-ru_none_k2_s0_b3"
    assert "test_bits_per_token_sampled" in row and math.isfinite(row["kl_bits"])
    hist = json.loads(Path(row["history"]).read_text())
    assert [h["step"] for h in hist] == [2, 4, 6] and all("dev_objective" in h for h in hist)
    best = min(hist, key=lambda h: h["dev_objective"])
    assert row["best_step"] == best["step"] and row["dev_bits_per_token"] == pytest.approx(best["dev_bits_per_token"])
    assert results.is_done(cell="translation-ru", k=2, seed=0, cond="none", design="flat2", beta=3.0)
    assert not results.is_done(cell="translation-ru", k=2, seed=0, cond="none", design="flat", beta=3.0)


def test_hierarchical_prefix_shapes_and_kl_decomposition():
    d, k = 8, 3
    prior = (torch.zeros(d), torch.zeros(d))
    hp = HierarchicalPrefix(k, d, members=["ru", "de"], prior=prior, tau=0.5)
    out = hp("ru", batch_size=2, sample=False)
    assert out.shape == (2, k, d)
    # mean prefix of a member = centre + its deviation
    assert torch.allclose(out[0], hp.centre_mu + hp.dev_mu["ru"])
    assert hp.kl_centre_nats().item() >= 0
    assert set(hp.kl_dev_nats().keys()) == {"ru", "de"}
    with torch.no_grad():
        hp.dev_mu["ru"].zero_()
        hp.dev_log_sigma["ru"].fill_(math.log(0.5))  # posterior == prior N(0, tau^2) (prior sigma is 1 here)
        hp.dev_mu["de"].fill_(0.3)
    assert hp.kl_dev_nats()["ru"].item() == pytest.approx(0.0, abs=1e-6)
    assert hp.kl_dev_nats()["de"].item() > 0


def test_tau_is_in_prior_sigma_units_and_deviations_start_at_zero_kl():
    d, k = 6, 2
    prior = (torch.zeros(d), torch.full((d,), math.log(0.025)))  # embedding-scale prior
    hp = HierarchicalPrefix(k, d, members=["ru"], prior=prior, tau=0.5)
    assert hp.tau_units == "prior"
    assert torch.allclose(hp.dev_log_sigma["ru"], torch.full((k, d), math.log(0.5 * 0.025)))
    assert torch.allclose(hp.centre_log_sigma, torch.full((k, d), math.log(0.025)))
    assert hp.kl_dev_nats()["ru"].item() == pytest.approx(0.0, abs=1e-6)
    # moving a deviation by one prior sigma costs the same under tau = 1 as moving the centre by one prior sigma
    hp1 = HierarchicalPrefix(k, d, members=["ru"], prior=prior, tau=1.0)
    with torch.no_grad():
        hp1.dev_mu["ru"].fill_(0.025)
        hp1.centre_mu.fill_(0.025)
    assert hp1.kl_dev_nats()["ru"].item() == pytest.approx(hp1.kl_centre_nats().item(), rel=1e-5)
    # absolute units: tau = 1 is a 40x wider prior, so the same displacement is far cheaper
    hpa = HierarchicalPrefix(k, d, members=["ru"], prior=prior, tau=1.0, tau_units="abs")
    assert torch.allclose(hpa.dev_log_sigma["ru"], torch.zeros(k, d))
    with torch.no_grad():
        hpa.dev_mu["ru"].fill_(0.025)
    assert hpa.kl_dev_nats()["ru"].item() < 0.01 * hp1.kl_dev_nats()["ru"].item()
    with pytest.raises(ValueError):
        HierarchicalPrefix(k, d, members=["ru"], prior=prior, tau=1.0, tau_units="relative")


def test_variational_prefix_sigma_starts_at_prior_sigma():
    from prefixlab.prefix import VariationalPrefix
    d = 5
    prior = (torch.zeros(d), torch.full((d,), math.log(0.03)))
    vp = VariationalPrefix(2, d, prior[0], prior[1])
    assert torch.allclose(vp.log_sigma, torch.full((2, d), math.log(0.03)))
    assert vp.kl_nats().item() == pytest.approx(0.0, abs=1e-6)  # mu == prior mu, sigma == prior sigma
    vp2 = VariationalPrefix(2, d, prior[0], prior[1], init_log_sigma=-2.0)
    assert torch.allclose(vp2.log_sigma, torch.full((2, d), -2.0))


def test_hierarchical_sample_adds_noise_from_both_levels():
    hp = HierarchicalPrefix(2, 4, members=["ru"], prior=(torch.zeros(4), torch.zeros(4)), tau=1.0)
    torch.manual_seed(0)
    a = hp("ru", 1, sample=True)[0]
    b = hp("ru", 1, sample=True)[0]
    assert not torch.allclose(a, b)
    assert torch.equal(hp("ru", 1, sample=False)[0], hp("ru", 1, sample=False)[0])
