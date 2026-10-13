"""Argument groups and data assembly shared by the sweep and hierarchical command lines."""
from __future__ import annotations

import argparse

from .data.flores import load_flores_rows
from .model import load_model
from .tasks import LANGS, Cell, cell_examples, cell_pairs
from .train import TrainCfg


def csv_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default=None)


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", default="translation,recall,continuation")
    p.add_argument("--langs", default="ru,de")
    p.add_argument("--conds", default="none")
    p.add_argument("--flores-dir", default=None, help="local mirror with <split>/<code>.jsonl; default downloads from the Hub")
    p.add_argument("--mlama-dir", default="data_raw/mlama")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-dev", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ks", default="1,2,4,8,16,32")
    p.add_argument("--seeds", default="0")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--pad-to", type=int, default=None, help="bucket padded length to a multiple; use on MPS")
    p.add_argument("--results", default="results/sweep.csv")
    p.add_argument("--ckpt-dir", default="checkpoints")


def train_cfg_from_args(args) -> TrainCfg:
    return TrainCfg(steps=args.steps, batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                    eval_every=args.eval_every, eval_batch_size=args.eval_batch_size, pad_to=args.pad_to)


def resolve_model(args, model=None, tok=None):
    if model is None:
        model, tok = load_model(args.model, args.dtype, args.device)
    return model, tok


def load_flores_for(args, tasks: list[str], langs: list[str]):
    if not any(t != "recall" for t in tasks):
        return None, None
    codes = sorted({LANGS["en"]["flores"], *(LANGS[l]["flores"] for l in langs)})
    dev = load_flores_rows(args.flores_dir, codes, "dev", args.hf_token)
    devtest = load_flores_rows(args.flores_dir, codes, "devtest", args.hf_token)
    return dev, devtest


def valid_cell(task: str, lang: str) -> bool:
    return not (task == "translation" and lang == "en")


def cell_pairs_from_args(args, cell: Cell, dev_rows, devtest_rows) -> dict:
    pairs = cell_pairs(cell, dev_rows, devtest_rows, args.mlama_dir, args.dev_frac, args.split_seed)
    for split, cap in (("train", args.max_train), ("dev", args.max_dev), ("test", args.max_test)):
        if cap:
            pairs[split] = pairs[split][:cap]
    return pairs


def cell_data_from_args(args, tok, cell: Cell, cond: str, dev_rows, devtest_rows) -> dict:
    return cell_examples(tok, cell_pairs_from_args(args, cell, dev_rows, devtest_rows), cell, cond)
