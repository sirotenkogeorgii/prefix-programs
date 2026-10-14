"""Composed prefixes: tied task parts and language parts trained jointly over the grid, with held-out cells scored
zero-shot. One joint run per (holdout set, k pair, seed, cond); one CSV row per cell."""
from __future__ import annotations

import argparse
import time

from .cli_common import (add_data_args, add_model_args, add_train_args, all_cells, cell_data_from_args, csv_list,
                         load_flores_for, parse_cells, resolve_model, train_cfg_from_args)
from .compose import ComposeSpec, run_compose
from .results import ResultsTable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Composed task + language prefixes")
    add_model_args(p)
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--k-pairs", default="1+1,2+2,4+4", help="task slots + language slots per run")
    p.add_argument("--holdout-sets", default="none,recall-de+continuation-ru",
                   help="comma-separated holdout sets; cells within a set joined by '+'; 'none' = train on every cell")
    p.set_defaults(results="results/compose.csv")
    return p


def parse_k_pairs(s: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in item.split("+")) for item in csv_list(s)]


def parse_holdout_sets(s: str) -> list[tuple]:
    return [() if item == "none" else tuple(parse_cells(item.replace("+", ","))) for item in csv_list(s)]


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    tasks, langs, conds = csv_list(args.tasks), csv_list(args.langs), csv_list(args.conds)
    seeds = [int(s) for s in csv_list(args.seeds)]
    cfg = train_cfg_from_args(args)
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)
    cells = all_cells(tasks, langs)

    for cond in conds:
        data = {c: cell_data_from_args(args, tok, c, cond, dev_rows, devtest_rows) for c in cells}
        for holdout in parse_holdout_sets(args.holdout_sets):
            assert all(h in data for h in holdout), f"holdout cell outside the grid: {holdout}"
            for k_task, k_lang in parse_k_pairs(args.k_pairs):
                for seed in seeds:
                    spec = ComposeSpec(k_task, k_lang, seed, cond, holdout)
                    t0 = time.time()
                    rows = run_compose(model, tok, data, spec, cfg, results, args.ckpt_dir)
                    label = f"[compose {cond} holdout={spec.holdout_tag()}] k={k_task}+{k_lang} seed={seed}"
                    if rows:
                        summary = " ".join(f"{r['cell']}{'*' if r['held_out'] else ''}={r['test_bits_per_token']:.3f}" for r in rows)
                        print(f"{label}: test bits/tok {summary} ({time.time() - t0:.0f}s)", flush=True)
                    else:
                        print(f"{label}: done, skipped", flush=True)
    return args.results
