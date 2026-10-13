"""Composed prefixes: one task part per task and one language part per language, trained jointly across cells;
held-out cells are evaluated zero-shot by composing parts never trained together."""
import pandas as pd

from prefixlab.compose import ComposeSpec, run_compose
from prefixlab.data.formats import Cond, build_example
from prefixlab.results import ResultsTable
from prefixlab.tasks import Cell
from prefixlab.train import TrainCfg

TR_RU = [("The cat sat on the mat.", "Кошка сидела на коврике."), ("I like green tea.", "Я люблю зелёный чай."),
         ("Where is the station?", "Где находится вокзал?"), ("It is raining today.", "Сегодня идёт дождь.")]
TR_DE = [("The cat sat on the mat.", "Die Katze saß auf der Matte."), ("I like green tea.", "Ich mag grünen Tee."),
         ("Where is the station?", "Wo ist der Bahnhof?"), ("It is raining today.", "Heute regnet es.")]
RC_RU = [("Dante was born in", "Флоренция"), ("Einstein was born in", "Ульм"), ("Mozart was born in", "Зальцбург"), ("Kant was born in", "Кёнигсберг")]
RC_DE = [("Dante was born in", "Florenz"), ("Einstein was born in", "Ulm"), ("Mozart was born in", "Salzburg"), ("Kant was born in", "Königsberg")]


def _data(tok, pairs, sep):
    exs = [build_example(tok, x, y, Cond(), sep=sep) for x, y in pairs]
    return {"train": exs, "dev": exs[:2], "test": exs[2:]}


def test_run_compose_trains_parts_jointly_and_scores_held_out_cell(tok, tiny_model, tmp_path):
    cells = {
        Cell("translation", "ru"): _data(tok, TR_RU, "\n"), Cell("translation", "de"): _data(tok, TR_DE, "\n"),
        Cell("recall", "ru"): _data(tok, RC_RU, " "), Cell("recall", "de"): _data(tok, RC_DE, " "),
    }
    results = ResultsTable(tmp_path / "compose.csv")
    spec = ComposeSpec(k_task=2, k_lang=1, seed=0, cond="none", holdout=(Cell("recall", "de"),))
    cfg = TrainCfg(steps=6, batch_size=4, lr=0.05, warmup=1, eval_every=3, seed=0)
    rows = run_compose(tiny_model, tok, cells, spec, cfg, results, tmp_path / "ckpt")
    df = pd.read_csv(tmp_path / "compose.csv")
    assert len(rows) == 4 and len(df) == 4
    assert set(df["design"]) == {"compose"} and (df["k"] == 3).all()
    held = df.set_index("cell")["held_out"]
    assert bool(held["recall-de"]) is True and not any(held.drop("recall-de"))
    assert (df["test_bits_per_token"] > 0).all()
    assert run_compose(tiny_model, tok, cells, spec, cfg, results, tmp_path / "ckpt") == []
