"""Regime 2: the input is replaced by m optimizable vectors per example; the target is scored with and without
the shared cell prefix. Reference points: real input (regime 1) and no input (m = 0)."""
import torch

from prefixlab.regime2 import Regime2Cfg, compress_inputs, pooled_init, run_regime2
from prefixlab.results import ResultsTable
from prefixlab.tasks import Cell

PAIRS = [
    ("The cat sat on the mat.", "Кошка сидела на коврике."),
    ("I like green tea.", "Я люблю зелёный чай."),
    ("Where is the station?", "Где находится вокзал?"),
    ("It is raining today.", "Сегодня идёт дождь."),
]


def test_pooled_init_averages_contiguous_groups_and_handles_short_inputs():
    emb = torch.arange(12.0).view(6, 2)  # 6 input tokens, d = 2
    out = pooled_init(emb, m=3)
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], emb[0:2].mean(0)) and torch.allclose(out[2], emb[4:6].mean(0))
    short = pooled_init(emb[:2], m=4)  # fewer tokens than slots: cycle
    assert short.shape == (4, 2) and torch.allclose(short[2], emb[0]) and torch.allclose(short[3], emb[1])


def test_compress_inputs_beats_no_input_and_reports_references(tok, tiny_model):
    cfg = Regime2Cfg(steps=30, lr=0.1, warmup=2)
    res = compress_inputs(tiny_model, tok, PAIRS, Cell("translation", "ru"), m=2, shared_prefix=None, cfg=cfg)
    n = len(PAIRS)
    assert len(res["bits_after"]) == n and len(res["bits_no_input"]) == n and len(res["bits_real_input"]) == n
    assert sum(res["bits_after"]) < sum(res["bits_no_input"]) - 1.0
    assert res["n_tokens"] == [len([m for m in e]) for e in res["target_masks"]] or sum(res["n_tokens"]) > 0
    assert res["m"] == 2 and res["k"] == 0


def test_run_regime2_one_row_per_m_and_skips_done(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "regime2.csv")
    cfg = Regime2Cfg(steps=3, lr=0.1, warmup=1)
    rows = run_regime2(tiny_model, tok, Cell("translation", "ru"), PAIRS, ms=[1, 2], shared_ckpt=None, k=0,
                       cond="none", cfg=cfg, results=results, seed=0)
    assert [r["m"] for r in rows] == [1, 2]
    assert all(r["design"] == "regime2" and r["k"] == 0 and r["cell"] == "translation-ru" for r in rows)
    assert all(r["bits_per_token_after"] > 0 and r["bits_per_token_no_input"] > 0 and r["bits_per_token_real_input"] > 0 for r in rows)
    assert run_regime2(tiny_model, tok, Cell("translation", "ru"), PAIRS, ms=[1, 2], shared_ckpt=None, k=0,
                       cond="none", cfg=cfg, results=results, seed=0) == []
