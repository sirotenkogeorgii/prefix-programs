"""End-to-end smoke test of the sweep command line with an injected tiny model."""
import pandas as pd

from prefixlab.cli_sweep import main


def test_cli_runs_grid_and_is_idempotent(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    argv = [
        "--tasks", "translation,recall,continuation", "--langs", "ru",
        "--ks", "1,2", "--seeds", "0", "--conds", "none,lang",
        "--steps", "4", "--batch-size", "4", "--eval-every", "2", "--baselines", "--floors", "--floor-steps", "2", "--floor-r", "2",
        "--results", str(tmp_path / "sweep.csv"), "--ckpt-dir", str(tmp_path / "ckpt"),
        "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2",
    ]
    main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "sweep.csv")
    # 3 cells x 2 conds x (2 ks + 1 baseline + 1 floor) = 24 rows
    assert len(df) == 24
    assert set(df["cell"]) == {"translation-ru", "recall-ru", "continuation-ru"}
    assert set(df["design"]) == {"A", "none", "floor"}
    assert (df["n_test"] <= 2).all()
    main(argv, model=tiny_model, tok=tok)
    assert len(pd.read_csv(tmp_path / "sweep.csv")) == 24
