"""Distance CLI: warm-start prequential transfer between all ordered cell pairs, prefixes taken from the sweep table."""
import pandas as pd

from prefixlab.cli_distance import main as distance_main
from prefixlab.cli_sweep import main as sweep_main


def _sweep(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    sweep_main(["--tasks", "translation", "--langs", "ru,de", "--ks", "1", "--seeds", "0", "--conds", "none",
                "--steps", "2", "--batch-size", "4", "--eval-every", "2",
                "--results", str(tmp_path / "sweep.csv"), "--ckpt-dir", str(tmp_path / "ckpt"),
                "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2"], model=tiny_model, tok=tok)


def test_cli_distance_uses_sweep_checkpoints(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    _sweep(tok, tiny_model, tmp_path, flores_dir, mlama_dir)
    argv = ["--tasks", "translation", "--langs", "ru,de", "--k", "1", "--seed", "0", "--cond", "none",
            "--sweep-results", str(tmp_path / "sweep.csv"), "--budget", "4", "--first-block", "2", "--steps", "1",
            "--batch-size", "2", "--results", str(tmp_path / "distance.csv"),
            "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2"]
    distance_main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "distance.csv")
    assert len(df) == 6  # 2x2 ordered pairs + 2 from scratch
    assert set(df["src"]) == {"translation-ru", "translation-de", "scratch"}
    assert (df["k"] == 1).all() and (df["budget"] == 4).all()
    distance_main(argv, model=tiny_model, tok=tok)
    assert len(pd.read_csv(tmp_path / "distance.csv")) == 6
