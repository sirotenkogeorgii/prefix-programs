"""Information distance between cells: warm-start from every cell's Design-A prefix (and from scratch) toward every
cell, prequential bits over a fixed budget. Prefixes are read from the sweep table at (k, seed, cond)."""
from __future__ import annotations

import argparse
import time

from .cli_common import (add_data_args, add_model_args, add_train_args, all_cells, cell_data_from_args, csv_list,
                         find_ckpt, load_flores_for, parse_cells, resolve_model, train_cfg_from_args)
from .distance import run_distance
from .results import ResultsTable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Warm-start information distance between cells")
    add_model_args(p)
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--cells", default=None, help="subset like translation-ru,recall-de; default: every valid cell")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cond", default="none")
    p.add_argument("--sweep-results", default="results/sweep.csv")
    p.add_argument("--budget", type=int, default=512, help="training examples of the destination cell")
    p.add_argument("--first-block", type=int, default=16)
    p.set_defaults(steps=50, results="results/distance.csv")  # steps per prequential block
    return p


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    cells = parse_cells(args.cells) if args.cells else all_cells(csv_list(args.tasks), csv_list(args.langs))
    tasks, langs = sorted({c.task for c in cells}), sorted({c.lang for c in cells})
    cfg = train_cfg_from_args(args)
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)
    info = {}
    for c in cells:
        ckpt = find_ckpt(args.sweep_results, design="A", cell=c.key(), k=args.k, seed=args.seed, cond=args.cond)
        info[c.key()] = {"cell": c, "ckpt": ckpt, "data": cell_data_from_args(args, tok, c, args.cond, dev_rows, devtest_rows)}
    t0 = time.time()
    rows = run_distance(model, tok, info, args.k, args.cond, cfg, results, args.budget, args.first_block, args.seed)
    for r in rows:
        print(f"[distance {r['src']} -> {r['dst']}] preq {r['preq_bits_per_token']:.3f} bits/tok, "
              f"zero-shot {r['zero_shot_test_bits_per_token']:.3f}, after {r['test_bits_per_token_after']:.3f}", flush=True)
    print(f"distance: {len(rows)} new rows ({time.time() - t0:.0f}s)", flush=True)
    return args.results
