"""Prompt-tuning loop, evaluation, checkpoints and the resumable results table."""
import torch

from prefixlab.data.formats import Cond, build_example
from prefixlab.prefix import SoftPrefix, VariationalPrefix, gaussian_prior_from_embeddings
from prefixlab.results import ResultsTable, load_prefix, save_prefix
from prefixlab.train import TrainCfg, evaluate, train_prefix

PAIRS = [
    ("The cat sat on the mat.", "Кошка сидела на коврике."),
    ("I like green tea.", "Я люблю зелёный чай."),
    ("Where is the station?", "Где находится вокзал?"),
    ("It is raining today.", "Сегодня идёт дождь."),
    ("She reads every evening.", "Она читает каждый вечер."),
    ("We will come tomorrow.", "Мы придём завтра."),
]


def _examples(tok):
    return [build_example(tok, x, y, Cond()) for x, y in PAIRS]


def test_training_reduces_dev_bits_on_tiny_overfit(tok, tiny_model):
    exs = _examples(tok)
    E = tiny_model.get_input_embeddings().weight.detach()
    prefix = SoftPrefix.from_vocab(E, k=4, seed=0)
    before = evaluate(tiny_model, prefix, exs, tok.pad_token_id).bits_per_token
    cfg = TrainCfg(steps=40, batch_size=6, lr=0.1, warmup=5, eval_every=20, seed=0)
    res = train_prefix(tiny_model, prefix, exs, exs, tok.pad_token_id, cfg)
    assert res.best_dev_bits_per_token < before - 0.02
    assert res.steps_run == 40


def test_best_state_roundtrips_through_evaluate(tok, tiny_model):
    exs = _examples(tok)
    E = tiny_model.get_input_embeddings().weight.detach()
    prefix = SoftPrefix.from_vocab(E, k=2, seed=1)
    cfg = TrainCfg(steps=20, batch_size=6, lr=0.1, warmup=2, eval_every=10, seed=1)
    res = train_prefix(tiny_model, prefix, exs, exs, tok.pad_token_id, cfg)
    fresh = SoftPrefix(k=2, d=E.shape[1])
    fresh.load_state_dict(res.best_state)
    again = evaluate(tiny_model, fresh, exs, tok.pad_token_id).bits_per_token
    assert abs(again - res.best_dev_bits_per_token) < 1e-4
    assert [h["step"] for h in res.history if "dev_bits_per_token" in h] == [10, 20]


def test_same_seed_gives_same_result(tok, tiny_model):
    exs = _examples(tok)
    E = tiny_model.get_input_embeddings().weight.detach()
    cfg = TrainCfg(steps=10, batch_size=3, lr=0.1, warmup=2, eval_every=10, seed=3)
    r1 = train_prefix(tiny_model, SoftPrefix.from_vocab(E, 2, 3), exs, exs, tok.pad_token_id, cfg)
    r2 = train_prefix(tiny_model, SoftPrefix.from_vocab(E, 2, 3), exs, exs, tok.pad_token_id, cfg)
    assert r1.best_dev_bits_per_token == r2.best_dev_bits_per_token


def test_variational_training_reports_kl_and_uses_mean_at_eval(tok, tiny_model):
    exs = _examples(tok)
    E = tiny_model.get_input_embeddings().weight.detach()
    vp = VariationalPrefix.from_vocab(E, k=2, seed=0, prior=gaussian_prior_from_embeddings(E))
    cfg = TrainCfg(steps=10, batch_size=6, lr=0.05, warmup=2, eval_every=10, seed=0, beta=1.0, sample=True)
    res = train_prefix(tiny_model, vp, exs, exs, tok.pad_token_id, cfg)
    last = [h for h in res.history if "dev_bits_per_token" in h][-1]
    assert last["kl_bits"] > 0 and torch.isfinite(torch.tensor(last["kl_bits"]))
    fresh = VariationalPrefix(2, E.shape[1], *gaussian_prior_from_embeddings(E))
    fresh.load_state_dict(res.best_state)
    again = evaluate(tiny_model, fresh, exs, tok.pad_token_id).bits_per_token
    assert abs(again - res.best_dev_bits_per_token) < 1e-4


def test_evaluate_without_prefix_matches_scoring(tok, tiny_model):
    from prefixlab.scoring import collate, target_bits

    exs = _examples(tok)
    ev = evaluate(tiny_model, None, exs, tok.pad_token_id, batch_size=4)
    direct = sum(target_bits(tiny_model, collate([e], tok.pad_token_id), None).item() for e in exs)
    assert abs(ev.bits_total - direct) < 1e-3
    assert ev.n_tokens == sum(e.n_target_tokens for e in exs)
    assert len(ev.per_example_bits) == len(exs)


def test_results_table_skips_finished_runs(tmp_path):
    t = ResultsTable(tmp_path / "sweep.csv")
    key = dict(task="translation", lang="ru", k=4, seed=0)
    assert not t.is_done(**key)
    t.append({**key, "bits_per_token": 1.23})
    assert t.is_done(**key)
    assert not t.is_done(task="translation", lang="ru", k=8, seed=0)
    assert ResultsTable(tmp_path / "sweep.csv").is_done(**key)  # survives reopen
    assert t.load().shape == (1, 5)


def test_prefix_checkpoint_roundtrip(tmp_path):
    p = SoftPrefix(k=3, d=8)
    save_prefix(p, tmp_path / "p.pt")
    q = SoftPrefix(k=3, d=8)
    load_prefix(q, tmp_path / "p.pt")
    assert torch.equal(p.weight, q.weight)
