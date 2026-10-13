"""Warm-start transfer between cells: K(a|b) as prequential bits reaching cell a from cell b's prefix."""
import pandas as pd

from prefixlab.data.formats import Cond, build_example
from prefixlab.distance import run_distance
from prefixlab.prefix import SoftPrefix
from prefixlab.results import ResultsTable, save_prefix
from prefixlab.tasks import Cell
from prefixlab.train import TrainCfg

RU = [("The cat sat on the mat.", "Кошка сидела на коврике."), ("I like green tea.", "Я люблю зелёный чай."),
      ("Where is the station?", "Где находится вокзал?"), ("It is raining today.", "Сегодня идёт дождь.")]
DE = [("The cat sat on the mat.", "Die Katze saß auf der Matte."), ("I like green tea.", "Ich mag grünen Tee."),
      ("Where is the station?", "Wo ist der Bahnhof?"), ("It is raining today.", "Heute regnet es.")]


def _data(tok, pairs):
    exs = [build_example(tok, x, y, Cond()) for x, y in pairs]
    return {"train": exs, "dev": exs[:2], "test": exs[2:]}


def test_run_distance_covers_all_ordered_pairs_and_scratch(tok, tiny_model, tmp_path):
    E = tiny_model.get_input_embeddings().weight.detach()
    cells = {}
    for name, pairs, seed in (("translation-ru", RU, 0), ("translation-de", DE, 1)):
        ckpt = tmp_path / f"{name}.pt"
        save_prefix(SoftPrefix.from_vocab(E, 2, seed), ckpt)
        cells[name] = {"cell": Cell("translation", name.split("-")[1]), "ckpt": str(ckpt), "data": _data(tok, pairs)}
    results = ResultsTable(tmp_path / "distance.csv")
    cfg = TrainCfg(steps=2, batch_size=2, lr=0.1, warmup=1, seed=0)
    rows = run_distance(tiny_model, tok, cells, k=2, cond="none", cfg=cfg, results=results, budget=4, first_block=2)
    df = pd.read_csv(tmp_path / "distance.csv")
    assert len(rows) == 6 and len(df) == 6  # 2x2 ordered pairs + 2 from scratch
    assert set(df["src"]) == {"translation-ru", "translation-de", "scratch"}
    assert set(df["dst"]) == {"translation-ru", "translation-de"}
    assert (df["preq_bits"] > 0).all() and (df["zero_shot_test_bits_per_token"] > 0).all()
    assert (df["budget"] == 4).all()
    assert run_distance(tiny_model, tok, cells, k=2, cond="none", cfg=cfg, results=results, budget=4, first_block=2) == []
