"""Hierarchical CLI: centre per task with language members, or centre per language with task members."""
import pandas as pd

from prefixlab.cli_hier import main


def _base(tmp_path, flores_dir, mlama_dir):
    return ["--ks", "1", "--seeds", "0", "--conds", "none", "--betas", "1", "--taus", "0.5",
            "--steps", "2", "--batch-size", "4", "--eval-every", "2",
            "--results", str(tmp_path / "hier.csv"), "--ckpt-dir", str(tmp_path / "ckpt"),
            "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir), "--max-test", "2"]


def test_cli_hier_task_centre_with_language_members(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    argv = ["--group", "task", "--tasks", "translation", "--langs", "ru,de"] + _base(tmp_path, flores_dir, mlama_dir)
    main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "hier.csv")
    assert len(df) == 2
    assert set(df["cell"]) == {"translation-ru", "translation-de"}
    assert set(df["group"]) == {"task:translation"} and set(df["member"]) == {"ru", "de"}
    main(argv, model=tiny_model, tok=tok)
    assert len(pd.read_csv(tmp_path / "hier.csv")) == 2


def test_cli_hier_language_centre_with_task_members(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    argv = ["--group", "lang", "--tasks", "recall,continuation", "--langs", "ru"] + _base(tmp_path, flores_dir, mlama_dir)
    main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "hier.csv")
    assert len(df) == 2
    assert set(df["cell"]) == {"recall-ru", "continuation-ru"}
    assert set(df["group"]) == {"lang:ru"} and set(df["member"]) == {"recall", "continuation"}
