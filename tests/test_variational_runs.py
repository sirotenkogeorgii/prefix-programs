"""Design flat (variational, global prior) through run_one, and the hierarchical prefix."""
import math

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
        hp.dev_log_sigma["ru"].fill_(math.log(0.5))  # posterior == prior N(0, tau^2)
    assert hp.kl_dev_nats()["ru"].item() == pytest.approx(0.0, abs=1e-6)
    assert hp.kl_dev_nats()["de"].item() > 0


def test_hierarchical_sample_adds_noise_from_both_levels():
    hp = HierarchicalPrefix(2, 4, members=["ru"], prior=(torch.zeros(4), torch.zeros(4)), tau=1.0)
    torch.manual_seed(0)
    a = hp("ru", 1, sample=True)[0]
    b = hp("ru", 1, sample=True)[0]
    assert not torch.allclose(a, b)
    assert torch.equal(hp("ru", 1, sample=False)[0], hp("ru", 1, sample=False)[0])
