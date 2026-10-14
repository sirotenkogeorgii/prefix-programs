"""Regime 2 (code -> output): per-example input compression on a seeded subset of each cell's test pairs,
with or without the cell's shared Design-A prefix (from the sweep table at k > 0)."""
from __future__ import annotations

import argparse
import time

from .cli_common import (add_data_args, add_model_args, cap_split, cell_pairs_from_args, csv_list, find_ckpt,
                         load_flores_for, parse_cells, resolve_model)
from .regime2 import Regime2Cfg, run_regime2
from .results import ResultsTable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regime 2: compress inputs into m vectors per example")
    add_model_args(p)
    add_data_args(p)
    p.add_argument("--cells", default="translation-ru,translation-de,recall-ru,continuation-ru")
    p.add_argument("--ms", default="1,2,4,8")
    p.add_argument("--n-examples", type=int, default=100)
    p.add_argument("--k", type=int, default=0, help="shared prefix slots; 0 = no shared prefix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cond", default="none", help="which sweep prefix to share (the examples themselves carry no conditioning)")
    p.add_argument("--sweep-results", default="results/sweep.csv")
    p.add_argument("--r2-steps", type=int, default=200)
    p.add_argument("--r2-lr", type=float, default=0.1)
    p.add_argument("--r2-warmup", type=int, default=10)
    p.add_argument("--r2-batch-size", type=int, default=32)
    p.add_argument("--pad-to", type=int, default=None)
    p.add_argument("--results", default="results/regime2.csv")
    return p


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    cells = parse_cells(args.cells)
    tasks, langs = sorted({c.task for c in cells}), sorted({c.lang for c in cells})
    ms = [int(m) for m in csv_list(args.ms)]
    cfg = Regime2Cfg(steps=args.r2_steps, lr=args.r2_lr, warmup=args.r2_warmup, batch_size=args.r2_batch_size, pad_to=args.pad_to)
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)
    for c in cells:
        pairs = cell_pairs_from_args(args, c, dev_rows, devtest_rows)["test"]
        pairs = [(p[0], p[1]) for p in cap_split(pairs, args.n_examples, args.seed)]
        ckpt = None if args.k == 0 else find_ckpt(args.sweep_results, design="A", cell=c.key(), k=args.k, seed=args.seed, cond=args.cond)
        t0 = time.time()
        rows = run_regime2(model, tok, c, pairs, ms, ckpt, args.k, args.cond, cfg, results, args.seed)
        for r in rows:
            print(f"[regime2 {c.key()} k={args.k}] m={r['m']}: after {r['bits_per_token_after']:.3f} | "
                  f"no input {r['bits_per_token_no_input']:.3f} | real input {r['bits_per_token_real_input']:.3f} bits/tok", flush=True)
        print(f"[regime2 {c.key()} k={args.k}]: {len(rows)} new rows ({time.time() - t0:.0f}s)", flush=True)
    return args.results
