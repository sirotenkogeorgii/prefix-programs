"""One resumable run of the Design A sweep on the tiny model."""
from prefixlab.data.formats import Cond, build_example
from prefixlab.results import ResultsTable
from prefixlab.sweep import RunSpec, baseline_row, run_one
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


def test_run_one_writes_row_and_checkpoint_then_skips(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "sweep.csv")
    spec = RunSpec(cell=Cell("translation", "ru"), k=2, seed=0, cond="none")
    cfg = TrainCfg(steps=6, batch_size=4, lr=0.1, warmup=1, eval_every=3, seed=0)
    row = run_one(tiny_model, tok, spec, _data(tok), cfg, results, tmp_path / "ckpt")
    assert row is not None
    assert row["cell"] == "translation-ru" and row["k"] == 2 and row["cond"] == "none"
    assert row["test_bits_per_token"] > 0 and row["dev_bits_per_token"] > 0
    assert (tmp_path / "ckpt" / f"{row['run_id']}.pt").exists()
    assert results.is_done(cell="translation-ru", k=2, seed=0, cond="none", design="A")
    assert run_one(tiny_model, tok, spec, _data(tok), cfg, results, tmp_path / "ckpt") is None


def test_baseline_row_has_k_zero_and_no_checkpoint(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "sweep.csv")
    row = baseline_row(tiny_model, tok, Cell("translation", "ru"), "none", _data(tok), results)
    assert row["k"] == 0 and row["design"] == "none"
    assert row["test_bits_per_token"] > 0
    assert baseline_row(tiny_model, tok, Cell("translation", "ru"), "none", _data(tok), results) is None


import json
from pathlib import Path

from prefixlab.floor import FloorCfg
from prefixlab.sweep import floor_row
from prefixlab.train import evaluate


def test_run_one_saves_per_example_test_bits(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "sweep.csv")
    spec = RunSpec(cell=Cell("translation", "ru"), k=1, seed=0, cond="none")
    cfg = TrainCfg(steps=2, batch_size=4, lr=0.1, warmup=1, eval_every=2, seed=0)
    row = run_one(tiny_model, tok, spec, _data(tok), cfg, results, tmp_path / "ckpt")
    per = json.loads(Path(row["per_example"]).read_text())
    assert len(per["bits"]) == 2 and len(per["tokens"]) == 2
    assert abs(sum(per["bits"]) - row["test_bits_total"]) < 1e-3
    assert sum(per["tokens"]) == row["n_test_tokens"]
    assert Path(row["per_example"]).parent == tmp_path / "per_example"


def test_floor_row_records_design_floor_and_restores_model(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "sweep.csv")
    data = _data(tok)
    before = evaluate(tiny_model, None, data["test"], tok.pad_token_id).bits_per_token
    fcfg = FloorCfg(steps=4, batch_size=4, lr=1e-2, r=2, eval_every=2, warmup=1, seed=0)
    row = floor_row(tiny_model, tok, Cell("translation", "ru"), "none", data, results, fcfg)
    assert row["design"] == "floor" and row["k"] == 0 and row["seed"] == 0
    assert row["test_bits_per_token"] > 0 and Path(row["per_example"]).exists()
    after = evaluate(tiny_model, None, data["test"], tok.pad_token_id).bits_per_token
    assert abs(after - before) < 1e-6
    assert floor_row(tiny_model, tok, Cell("translation", "ru"), "none", data, results, fcfg) is None
