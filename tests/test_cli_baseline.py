"""Zero-rate baseline CLI: prior draws and random token rows scored without training."""
import json

import pandas as pd
import torch

from prefixlab.cli_baseline import draw_prefix, main
from prefixlab.prefix import gaussian_prior_from_embeddings


def test_cli_baseline_writes_one_row_per_cell_kind_k_and_is_idempotent(tok, tiny_model, tmp_path, flores_dir, mlama_dir):
    argv = ["--tasks", "translation,recall", "--langs", "ru", "--ks", "1,2", "--kinds", "prior,vocab", "--n-draws", "3",
            "--results", str(tmp_path / "baseline.csv"), "--flores-dir", str(flores_dir), "--mlama-dir", str(mlama_dir),
            "--max-test", "2"]
    main(argv, model=tiny_model, tok=tok)
    df = pd.read_csv(tmp_path / "baseline.csv")
    assert len(df) == 2 * 2 * 2  # cells x kinds x ks
    assert set(df.cell) == {"translation-ru", "recall-ru"} and set(df.kind) == {"prior", "vocab"} and set(df.k) == {1, 2}
    assert (df.n_draws == 3).all()
    for s, m in zip(df.test_bits_draws, df.test_bits_per_token):
        draws = json.loads(s)
        assert len(draws) == 3 and abs(sum(draws) / 3 - m) < 1e-5
    main(argv, model=tiny_model, tok=tok)
    assert len(pd.read_csv(tmp_path / "baseline.csv")) == 8


def test_prior_draw_matches_the_prior_and_vocab_draw_is_rows_of_e(tiny_model):
    E = tiny_model.get_input_embeddings().weight
    mu, log_sigma = gaussian_prior_from_embeddings(E)
    z = draw_prefix(E, "prior", 4000, 0).weight.detach()
    assert z.shape == (4000, E.shape[1])
    assert torch.allclose(z.mean(0), mu, atol=float(0.05 * log_sigma.exp().max()))
    assert torch.allclose(z.std(0), log_sigma.exp(), rtol=0.1)
    assert not torch.equal(draw_prefix(E, "prior", 2, 0).weight, draw_prefix(E, "prior", 2, 1).weight)
    v = draw_prefix(E, "vocab", 3, 0).weight.detach()
    assert all((E.detach().float() == row).all(1).any() for row in v)
