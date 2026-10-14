"""Regime 2 CLI: per-example input compression on a seeded subset of each cell's test pairs."""
import pandas as pd

from prefixlab.cli_regime2 import main as regime2_main
from prefixlab.cli_sweep import main as sweep_main


def test_cli_regime2_without_and_with_shared_prefix(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    common = ["--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--r2-steps", "2",
              "--results", str(tmp_path / "regime2.csv")]
    regime2_main(["--cells", "translation-ru,recall-ru", "--ms", "1,2", "--n-examples", "2", "--k", "0"] + common,
                 model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "regime2.csv")
    assert len(df) == 4 and set(df["cell"]) == {"translation-ru", "recall-ru"} and sorted(set(df["m"])) == [1, 2]
    assert (df["k"] == 0).all() and (df["n_examples"] == 2).all()

    sweep_main(["--tasks", "translation", "--langs", "ru", "--ks", "1", "--seeds", "0", "--conds", "none",
                "--steps", "2", "--batch-size", "4", "--eval-every", "2",
                "--results", str(tmp_path / "sweep.csv"), "--ckpt-dir", str(tmp_path / "ckpt"),
                "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2"], model=tiny_model, tok=tok)
    regime2_main(["--cells", "translation-ru", "--ms", "1", "--n-examples", "2", "--k", "1",
                  "--sweep-results", str(tmp_path / "sweep.csv")] + common, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "regime2.csv")
    assert len(df) == 5 and (df["k"] == 1).sum() == 1
