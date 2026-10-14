"""Composition CLI: tied task and language parts, one joint run per (holdout set, k pair, seed, cond)."""
import pandas as pd

from prefixlab.cli_compose import main


def test_cli_compose_runs_each_holdout_set_and_is_idempotent(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    argv = ["--tasks", "translation,recall", "--langs", "ru,de", "--k-pairs", "1+1", "--seeds", "0", "--conds", "none",
            "--holdout-sets", "none,recall-de", "--steps", "2", "--batch-size", "4", "--eval-every", "2",
            "--results", str(tmp_path / "compose.csv"), "--ckpt-dir", str(tmp_path / "ckpt"),
            "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2"]
    main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "compose.csv")
    assert len(df) == 8  # 2 holdout sets x 4 cells
    assert set(df["holdout"]) == {"none", "recall-de"}
    held = df[df["held_out"]]
    assert list(held["cell"]) == ["recall-de"] and list(held["holdout"]) == ["recall-de"]
    assert (df["k_task"] == 1).all() and (df["k_lang"] == 1).all() and (df["k"] == 2).all()
    main(argv, model=tiny_model, tok=tok)
    assert len(pd.read_csv(tmp_path / "compose.csv")) == 8
