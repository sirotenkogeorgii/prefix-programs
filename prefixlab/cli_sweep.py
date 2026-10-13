"""Design A sweep over cells x conditioning x k x seed, resumable. `main(argv, model, tok)` allows injection."""
from __future__ import annotations

import argparse
import time

from .data.flores import load_flores_rows
from .model import load_model
from .results import ResultsTable
from .floor import FloorCfg
from .sweep import RunSpec, baseline_row, floor_row, run_one
from .tasks import LANGS, Cell, cell_examples, cell_pairs
from .train import TrainCfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prefix sweep (Design A)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--tasks", default="translation,recall,continuation")
    p.add_argument("--langs", default="ru,de")
    p.add_argument("--ks", default="1,2,4,8,16,32")
    p.add_argument("--seeds", default="0")
    p.add_argument("--conds", default="none")
    p.add_argument("--baselines", action="store_true", help="also record the no-prefix (k=0) row per cell and cond")
    p.add_argument("--floors", action="store_true", help="also train a LoRA floor per cell and cond (design=floor)")
    p.add_argument("--floor-steps", type=int, default=1500)
    p.add_argument("--floor-lr", type=float, default=2e-4)
    p.add_argument("--floor-r", type=int, default=16)
    p.add_argument("--floor-alpha", type=int, default=32)
    p.add_argument("--floor-eval-every", type=int, default=100)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--results", default="results/sweep.csv")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--flores-dir", default=None, help="local mirror with <split>/<code>.jsonl; default downloads from the Hub")
    p.add_argument("--mlama-dir", default="data_raw/mlama")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-dev", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    return p


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    if model is None:
        model, tok = load_model(args.model, args.dtype, args.device)
    tasks, langs, conds = _csv(args.tasks), _csv(args.langs), _csv(args.conds)
    ks, seeds = [int(k) for k in _csv(args.ks)], [int(s) for s in _csv(args.seeds)]
    cfg = TrainCfg(steps=args.steps, batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                   eval_every=args.eval_every, eval_batch_size=args.eval_batch_size)
    results = ResultsTable(args.results)
    fcfg = FloorCfg(steps=args.floor_steps, batch_size=args.batch_size, lr=args.floor_lr, r=args.floor_r,
                    alpha=args.floor_alpha, eval_every=args.floor_eval_every, eval_batch_size=args.eval_batch_size)

    dev_rows = devtest_rows = None
    if any(t != "recall" for t in tasks):
        codes = sorted({LANGS["en"]["flores"], *(LANGS[l]["flores"] for l in langs)})
        dev_rows = load_flores_rows(args.flores_dir, codes, "dev", args.hf_token)
        devtest_rows = load_flores_rows(args.flores_dir, codes, "devtest", args.hf_token)

    for task in tasks:
        for lang in langs:
            if task == "translation" and lang == "en":
                continue
            cell = Cell(task, lang)
            pairs = cell_pairs(cell, dev_rows, devtest_rows, args.mlama_dir, args.dev_frac, args.split_seed)
            for split, cap in (("train", args.max_train), ("dev", args.max_dev), ("test", args.max_test)):
                if cap:
                    pairs[split] = pairs[split][:cap]
            for cond in conds:
                data = cell_examples(tok, pairs, cell, cond)
                if args.baselines:
                    row = baseline_row(model, tok, cell, cond, data, results)
                    if row:
                        print(f"[{cell.key()} {cond}] baseline k=0: test {row['test_bits_per_token']:.3f} bits/tok", flush=True)
                if args.floors:
                    t0 = time.time()
                    row = floor_row(model, tok, cell, cond, data, results, fcfg)
                    if row:
                        print(f"[{cell.key()} {cond}] LoRA floor: test {row['test_bits_per_token']:.3f} bits/tok ({time.time() - t0:.0f}s)", flush=True)
                for k in ks:
                    for seed in seeds:
                        spec = RunSpec(cell, k, seed, cond)
                        t0 = time.time()
                        row = run_one(model, tok, spec, data, cfg, results, args.ckpt_dir)
                        if row:
                            print(f"[{cell.key()} {cond}] k={k} seed={seed}: dev {row['dev_bits_per_token']:.3f} "
                                  f"test {row['test_bits_per_token']:.3f} bits/tok ({time.time() - t0:.0f}s)", flush=True)
                        else:
                            print(f"[{cell.key()} {cond}] k={k} seed={seed}: done, skipped", flush=True)
    return args.results
