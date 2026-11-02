"""Zero-rate baselines, no training. Two kinds of untrained prefix are scored on every cell's test set:

    prior   k vectors drawn from the embedding prior pi_0 = N(mu_0, diag sigma_0^2): rate 0 under the variational code
            (q = pi_0), the true zero-bit program.
    vocab   k random rows of the token-embedding matrix: the recipient of the mechanism probes; not rate 0, since rows of
            E are not draws from pi_0, but a zero-training control.

One row per (cell, kind, k, seed, cond) with the mean and sd over n draws and the per-draw test bits per token.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .cli_common import (add_data_args, add_model_args, cell_data_from_args, csv_list, load_flores_for, resolve_model,
                         valid_cell)
from .prefix import SoftPrefix, gaussian_prior_from_embeddings
from .results import ResultsTable
from .tasks import Cell
from .train import evaluate

KINDS = ("prior", "vocab")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zero-rate prefix baselines: prior draws and random token rows, no training")
    add_model_args(p)
    add_data_args(p)
    p.add_argument("--ks", default="2,4")
    p.add_argument("--kinds", default="prior,vocab", help="comma list of: prior, vocab")
    p.add_argument("--n-draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--pad-to", type=int, default=None, help="bucket padded length to a multiple; use on MPS")
    p.add_argument("--results", default="results/baseline.csv")
    return p


def draw_prefix(E: torch.Tensor, kind: str, k: int, seed: int) -> SoftPrefix:
    """An untrained prefix: `prior` draws k vectors from the diagonal Gaussian fitted to E; `vocab` takes k rows of E."""
    if kind == "prior":
        mu, log_sigma = gaussian_prior_from_embeddings(E)
        g = torch.Generator().manual_seed(seed)
        z = mu.cpu() + log_sigma.exp().cpu() * torch.randn(k, E.shape[1], generator=g)
        return SoftPrefix(k, E.shape[1], init=z)
    if kind == "vocab":
        return SoftPrefix.from_vocab(E, k, seed)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    tasks, langs, conds = csv_list(args.tasks), csv_list(args.langs), csv_list(args.conds)
    ks, kinds = [int(k) for k in csv_list(args.ks)], csv_list(args.kinds)
    for kind in kinds:
        if kind not in KINDS:
            raise SystemExit(f"--kinds: unknown kind {kind!r}; choose from {KINDS}")
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)
    E = model.get_input_embeddings().weight
    device = E.device

    for t in tasks:
        for l in langs:
            if not valid_cell(t, l):
                continue
            cell = Cell(t, l)
            for cond in conds:
                data = None
                for kind in kinds:
                    for k in ks:
                        key = dict(design="baseline", cell=cell.key(), kind=kind, k=k, seed=args.seed, cond=cond)
                        if results.is_done(**key):
                            print(f"[{cell.key()} {cond}] {kind} k={k}: done, skipped", flush=True)
                            continue
                        if data is None:
                            data = cell_data_from_args(args, tok, cell, cond, dev_rows, devtest_rows)
                        t0, draws, n_tokens = time.time(), [], 0
                        for j in range(args.n_draws):
                            prefix = draw_prefix(E, kind, k, args.seed * 1000 + j).to(device)
                            ev = evaluate(model, prefix, data["test"], tok.pad_token_id, args.eval_batch_size, pad_to=args.pad_to)
                            draws.append(ev.bits_per_token)
                            n_tokens = ev.n_tokens
                        mean = sum(draws) / len(draws)
                        sd = (sum((x - mean) ** 2 for x in draws) / max(len(draws) - 1, 1)) ** 0.5
                        results.append({
                            **key, "model": getattr(model, "name_or_path", "") or getattr(model.config, "_name_or_path", ""),
                            "n_draws": len(draws), "n_test": len(data["test"]), "n_test_tokens": n_tokens,
                            "test_bits_per_token": mean, "test_bits_per_token_sd": sd,
                            "test_bits_draws": json.dumps([round(x, 6) for x in draws]),
                        })
                        print(f"[{cell.key()} {cond}] {kind} k={k}: {mean:.3f} ± {sd:.3f} bits/tok over {len(draws)} draws "
                              f"({time.time() - t0:.0f}s)", flush=True)
    return args.results
