"""Mechanism rows: one per (probe, layer); an identity probe recovers zero bits in every block."""
import pandas as pd
import torch

from prefixlab.cli_hier import main as hier_main
from prefixlab.cli_mechanism import main as mechanism_main
from prefixlab.cli_sweep import main as sweep_main
from prefixlab.data.formats import Cond, build_example
from prefixlab.mechanism import run_mechanism
from prefixlab.prefix import SoftPrefix
from prefixlab.results import ResultsTable
from prefixlab.tasks import Cell

PAIRS = [("The cat sat on the mat.", "Кошка сидела на коврике."), ("I like green tea.", "Я люблю зелёный чай.")]


def test_run_mechanism_one_row_per_probe_and_layer(tok, tiny_model, tmp_path):
    exs = [build_example(tok, x, y, Cond()) for x, y in PAIRS]
    E = tiny_model.get_input_embeddings().weight.detach()
    a, b = SoftPrefix.from_vocab(E, 2, 0).weight.detach(), SoftPrefix.from_vocab(E, 2, 5).weight.detach()
    results = ResultsTable(tmp_path / "mechanism.csv")
    rows = run_mechanism(tiny_model, tok, Cell("translation", "ru"), exs, {"identity": (a, a), "A": (a, b)},
                         k=2, seed=0, cond="none", results=results, batch_size=2)
    L = tiny_model.config.num_hidden_layers
    df = pd.read_csv(tmp_path / "mechanism.csv")
    assert len(rows) == 2 * L and len(df) == 2 * L
    ident = df[df["probe"] == "identity"]
    assert sorted(ident["layer"]) == list(range(L)) and (ident["recovered_bits_per_token"].abs() < 1e-5).all()
    assert run_mechanism(tiny_model, tok, Cell("translation", "ru"), exs, {"identity": (a, a)},
                         k=2, seed=0, cond="none", results=results, batch_size=2) == []


def test_cli_mechanism_builds_probes_from_sweep_and_hier_tables(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    data = ["--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2"]
    sweep_main(["--tasks", "translation", "--langs", "ru,de", "--ks", "1", "--seeds", "0", "--conds", "none",
                "--steps", "2", "--batch-size", "4", "--eval-every", "2",
                "--results", str(tmp_path / "sweep.csv"), "--ckpt-dir", str(tmp_path / "ckpt")] + data, model=tiny_model, tok=tok)
    hier_main(["--group", "task", "--tasks", "translation", "--langs", "ru,de", "--ks", "1", "--seeds", "0", "--conds", "none",
               "--betas", "1", "--taus", "1", "--steps", "2", "--batch-size", "4", "--eval-every", "2",
               "--results", str(tmp_path / "hier_task.csv"), "--ckpt-dir", str(tmp_path / "ckpt")] + data, model=tiny_model, tok=tok)
    argv = ["--cells", "translation-ru", "--k", "1", "--seed", "0", "--cond", "none",
            "--sweep-results", str(tmp_path / "sweep.csv"), "--hier-results", str(tmp_path / "hier_task.csv"),
            "--hier-beta", "1", "--hier-tau", "1", "--max-examples", "2",
            "--results", str(tmp_path / "mechanism.csv")] + data
    mechanism_main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "mechanism.csv")
    L = tiny_model.config.num_hidden_layers
    assert set(df["probe"]) == {"A", "hier_task_centre", "hier_task_dev"} and len(df) == 3 * L
    assert (df["cell"] == "translation-ru").all()
    mechanism_main(argv, model=tiny_model, tok=tok)
    assert len(pd.read_csv(tmp_path / "mechanism.csv")) == 3 * L
