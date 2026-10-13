"""Hierarchical design: --group task (centre per task, members = languages) or --group lang (centre per language,
members = tasks). One joint training run per (group value, cond, k, seed, beta, tau); one CSV row per member."""
from __future__ import annotations

import argparse
import time

from .cli_common import (add_data_args, add_model_args, add_train_args, cell_data_from_args, csv_list,
                         load_flores_for, resolve_model, train_cfg_from_args, valid_cell)
from .hier import HierSpec, run_hier
from .results import ResultsTable
from .tasks import Cell


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hierarchical variational prefixes")
    add_model_args(p)
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--group", default="task", choices=["task", "lang"])
    p.add_argument("--betas", default="1")
    p.add_argument("--taus", default="1")
    p.set_defaults(results="results/hier.csv")
    return p


def groups(group: str, tasks: list[str], langs: list[str]) -> list[tuple[str, dict[str, Cell]]]:
    out = []
    if group == "task":
        for t in tasks:
            members = {l: Cell(t, l) for l in langs if valid_cell(t, l)}
            if members:
                out.append((f"task:{t}", members))
    else:
        for l in langs:
            members = {t: Cell(t, l) for t in tasks if valid_cell(t, l)}
            if members:
                out.append((f"lang:{l}", members))
    return out


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    tasks, langs, conds = csv_list(args.tasks), csv_list(args.langs), csv_list(args.conds)
    ks, seeds = [int(k) for k in csv_list(args.ks)], [int(s) for s in csv_list(args.seeds)]
    betas, taus = [float(b) for b in csv_list(args.betas)], [float(t) for t in csv_list(args.taus)]
    cfg = train_cfg_from_args(args)
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)

    for group_name, members in groups(args.group, tasks, langs):
        for cond in conds:
            data = {m: cell_data_from_args(args, tok, c, cond, dev_rows, devtest_rows) for m, c in members.items()}
            for k in ks:
                for seed in seeds:
                    for beta in betas:
                        for tau in taus:
                            spec = HierSpec(group_name, k, seed, cond, beta, tau)
                            t0 = time.time()
                            rows = run_hier(model, tok, spec, members, data, cfg, results, args.ckpt_dir)
                            label = f"[{group_name} {cond}] k={k} seed={seed} beta={beta:g} tau={tau:g}"
                            if rows:
                                summary = " ".join(f"{r['member']}={r['test_bits_per_token']:.3f}" for r in rows)
                                print(f"{label}: centre {rows[0]['kl_centre_bits']:.1f} bits; test bits/tok {summary} "
                                      f"({time.time() - t0:.0f}s)", flush=True)
                            else:
                                print(f"{label}: done, skipped", flush=True)
    return args.results
