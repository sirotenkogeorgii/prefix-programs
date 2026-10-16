"""Design A sweep (or flat variational with --design flat) over cells x conditioning x k x seed, resumable."""
from __future__ import annotations

import argparse
import time

from .cli_common import (add_data_args, add_model_args, add_train_args, cell_data_from_args, csv_list,
                         load_flores_for, resolve_model, train_cfg_from_args, valid_cell)
from .floor import FloorCfg
from .results import ResultsTable
from .sweep import RunSpec, baseline_row, floor_row, run_one, VARIATIONAL
from .tasks import Cell


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prefix sweep (Design A, or flat variational)")
    add_model_args(p)
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--design", default="A", choices=["A", "flat", "flat2"])
    p.add_argument("--betas", default="1", help="KL weights for --design flat / flat2")
    p.add_argument("--baselines", action="store_true", help="also record the no-prefix (k=0) row per cell and cond")
    p.add_argument("--floors", action="store_true", help="also train a LoRA floor per cell and cond (design=floor)")
    p.add_argument("--floor-steps", type=int, default=1500)
    p.add_argument("--floor-lr", type=float, default=2e-4)
    p.add_argument("--floor-r", type=int, default=16)
    p.add_argument("--floor-alpha", type=int, default=32)
    p.add_argument("--floor-eval-every", type=int, default=100)
    return p


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    tasks, langs, conds = csv_list(args.tasks), csv_list(args.langs), csv_list(args.conds)
    ks, seeds = [int(k) for k in csv_list(args.ks)], [int(s) for s in csv_list(args.seeds)]
    betas = [float(b) for b in csv_list(args.betas)] if args.design in VARIATIONAL else [0.0]
    cfg = train_cfg_from_args(args)
    fcfg = FloorCfg(steps=args.floor_steps, batch_size=args.batch_size, lr=args.floor_lr, r=args.floor_r,
                    alpha=args.floor_alpha, eval_every=args.floor_eval_every, eval_batch_size=args.eval_batch_size,
                    pad_to=args.pad_to)
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)

    for task in tasks:
        for lang in langs:
            if not valid_cell(task, lang):
                continue
            cell = Cell(task, lang)
            for cond in conds:
                data = cell_data_from_args(args, tok, cell, cond, dev_rows, devtest_rows)
                tag = f"[{cell.key()} {cond}]"
                if args.baselines:
                    row = baseline_row(model, tok, cell, cond, data, results, args.pad_to, args.eval_batch_size)
                    if row:
                        print(f"{tag} baseline k=0: test {row['test_bits_per_token']:.3f} bits/tok", flush=True)
                if args.floors:
                    t0 = time.time()
                    row = floor_row(model, tok, cell, cond, data, results, fcfg)
                    if row:
                        print(f"{tag} LoRA floor: test {row['test_bits_per_token']:.3f} bits/tok ({time.time() - t0:.0f}s)", flush=True)
                for k in ks:
                    for seed in seeds:
                        for beta in betas:
                            spec = RunSpec(cell, k, seed, cond, design=args.design, beta=beta)
                            t0 = time.time()
                            row = run_one(model, tok, spec, data, cfg, results, args.ckpt_dir)
                            label = f"{tag} {args.design} k={k} seed={seed}" + (f" beta={beta:g}" if args.design in VARIATIONAL else "")
                            if row:
                                extra = f" kl={row['kl_bits']:.1f} bits" if args.design in VARIATIONAL else ""
                                print(f"{label}: dev {row['dev_bits_per_token']:.3f} test {row['test_bits_per_token']:.3f} "
                                      f"bits/tok{extra} ({time.time() - t0:.0f}s)", flush=True)
                            else:
                                print(f"{label}: done, skipped", flush=True)
    return args.results
