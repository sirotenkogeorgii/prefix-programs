"""Hierarchical design: one centre per group, one deviation per member, trained jointly."""
import json
import math
from pathlib import Path

from prefixlab.data.formats import Cond, build_example
from prefixlab.hier import HierSpec, run_hier
from prefixlab.results import ResultsTable
from prefixlab.tasks import Cell
from prefixlab.train import TrainCfg

RU = [("The cat sat on the mat.", "Кошка сидела на коврике."), ("I like green tea.", "Я люблю зелёный чай."),
      ("Where is the station?", "Где находится вокзал?"), ("It is raining today.", "Сегодня идёт дождь.")]
DE = [("The cat sat on the mat.", "Die Katze saß auf der Matte."), ("I like green tea.", "Ich mag grünen Tee."),
      ("Where is the station?", "Wo ist der Bahnhof?"), ("It is raining today.", "Heute regnet es.")]


def _data(tok, pairs):
    exs = [build_example(tok, x, y, Cond()) for x, y in pairs]
    return {"train": exs, "dev": exs[:2], "test": exs[2:]}


def test_run_hier_writes_one_row_per_member_with_both_kl_terms(tok, tiny_model, tmp_path):
    results = ResultsTable(tmp_path / "hier.csv")
    members = {"ru": Cell("translation", "ru"), "de": Cell("translation", "de")}
    data = {"ru": _data(tok, RU), "de": _data(tok, DE)}
    spec = HierSpec(group="task:translation", k=2, seed=0, cond="none", beta=1.0, tau=0.5)
    cfg = TrainCfg(steps=6, batch_size=4, lr=0.05, warmup=1, eval_every=3, seed=0)
    rows = run_hier(tiny_model, tok, spec, members, data, cfg, results, tmp_path / "ckpt")
    assert len(rows) == 2
    by_cell = {r["cell"]: r for r in rows}
    assert set(by_cell) == {"translation-ru", "translation-de"}
    for r in rows:
        assert r["design"] == "hier" and r["group"] == "task:translation" and r["tau"] == 0.5 and r["beta"] == 1.0
        assert r["kl_centre_bits"] > 0 and r["kl_dev_bits"] > 0 and math.isfinite(r["kl_dev_bits"])
        assert r["test_bits_per_token"] > 0 and r["dev_bits_per_token"] > 0
        assert r["kl_bits"] == r["kl_centre_bits"] + r["kl_dev_bits"]
    assert by_cell["translation-ru"]["kl_centre_bits"] == by_cell["translation-de"]["kl_centre_bits"]
    assert all(r["tau_units"] == "prior" and r["run_id"].startswith("hier_task-translation_none_k2_s0_b1_t0.5p_") for r in rows)
    hist = json.loads(Path(rows[0]["history"]).read_text())
    assert [h["step"] for h in hist] == [3, 6] and all("kl_centre_bits" in h and "kl_dev_bits" in h for h in hist)
    assert results.is_done(cell="translation-ru", design="hier", group="task:translation", k=2, seed=0, cond="none",
                           beta=1.0, tau=0.5, tau_units="prior")
    # the same tau in absolute units is a different run
    assert not results.is_done(cell="translation-ru", design="hier", group="task:translation", k=2, seed=0, cond="none",
                               beta=1.0, tau=0.5, tau_units="abs")
    assert run_hier(tiny_model, tok, spec, members, data, cfg, results, tmp_path / "ckpt") == []
