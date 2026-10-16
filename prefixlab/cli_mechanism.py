"""Mechanism sweep: per-block window patching of the prefix positions for Design-A prefixes (from the sweep table)
and, when hierarchical tables are given, for the centre alone and for the member deviation on top of it."""
from __future__ import annotations

import argparse
import time

from .cli_common import (add_data_args, add_model_args, all_cells, cap_split, cell_data_from_args, csv_list, find_ckpt,
                         find_row, load_flores_for, parse_cells, resolve_model)
from .mechanism import hier_parts, random_prefix, run_mechanism
from .results import ResultsTable, load_prefix
from .prefix import SoftPrefix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Window patching of prefix positions, block by block")
    add_model_args(p)
    add_data_args(p)
    p.add_argument("--cells", default=None, help="subset like translation-ru,recall-de; default: every valid cell")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cond", default="none")
    p.add_argument("--sweep-results", default="results/sweep.csv")
    p.add_argument("--hier-results", default="", help="comma-separated hierarchical tables (task and/or lang grouping)")
    p.add_argument("--hier-beta", type=float, default=1.0)
    p.add_argument("--hier-tau", type=float, default=1.0)
    p.add_argument("--hier-tau-units", default="prior", choices=["prior", "abs"])
    p.add_argument("--hier-protocol", default="prereg", choices=["prereg", "corrected"])
    p.add_argument("--max-examples", type=int, default=256, help="seeded subsample of the test split per cell")
    p.add_argument("--hier-sample", action="store_true",
                   help="probe one posterior sample per hierarchical part (seeded) instead of the posterior means; "
                        "this is the default for --hier-protocol corrected, whose posteriors are as wide as the prior")
    p.add_argument("--hier-mean", action="store_true", help="force posterior means even for the corrected protocol")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--pad-to", type=int, default=None)
    p.add_argument("--results", default="results/mechanism.csv")
    return p


def main(argv=None, model=None, tok=None) -> str:
    args = build_parser().parse_args(argv)
    model, tok = resolve_model(args, model, tok)
    cells = parse_cells(args.cells) if args.cells else all_cells(csv_list(args.tasks), csv_list(args.langs))
    if args.hier_protocol == "corrected" and not args.hier_sample and not args.hier_mean:
        print("[mechanism] corrected protocol: probing posterior samples (pass --hier-mean to override)", flush=True)
        args.hier_sample = True
    tasks, langs = sorted({c.task for c in cells}), sorted({c.lang for c in cells})
    results = ResultsTable(args.results)
    dev_rows, devtest_rows = load_flores_for(args, tasks, langs)
    E = model.get_input_embeddings().weight.detach()
    for c in cells:
        data = cell_data_from_args(args, tok, c, args.cond, dev_rows, devtest_rows)
        examples = cap_split(data["test"], args.max_examples, args.seed)
        rand = random_prefix(E, args.k, args.seed)
        probes = {}
        ckpt = find_ckpt(args.sweep_results, design="A", cell=c.key(), k=args.k, seed=args.seed, cond=args.cond)
        probes["A"] = (rand, load_prefix(SoftPrefix(args.k, E.shape[1]), ckpt).weight.detach())
        for table in csv_list(args.hier_results):
            row = find_row(table, design="hier", cell=c.key(), k=args.k, seed=args.seed, cond=args.cond,
                           beta=args.hier_beta, tau=args.hier_tau, tau_units=args.hier_tau_units, protocol=args.hier_protocol)
            kind = str(row["group"]).split(":")[0]
            centre, member = hier_parts(row["ckpt"], str(row["member"]), sample=args.hier_sample, seed=args.seed)
            probes[f"hier_{kind}_centre"] = (rand, centre)
            probes[f"hier_{kind}_dev"] = (centre, member)
        t0 = time.time()
        rows = run_mechanism(model, tok, c, examples, probes, args.k, args.seed, args.cond, results,
                             args.eval_batch_size, args.pad_to, extra={"hier_beta": args.hier_beta, "hier_tau": args.hier_tau,
                                    "hier_tau_units": args.hier_tau_units, "hier_protocol": args.hier_protocol,
                                    "hier_sample": bool(args.hier_sample)})
        for name in sorted({r["probe"] for r in rows}):
            best = max((r for r in rows if r["probe"] == name), key=lambda r: r["recovered_bits_per_token"])
            print(f"[mechanism {c.key()} k={args.k}] {name}: recipient {best['recipient_bits_per_token']:.3f} -> donor "
                  f"{best['donor_bits_per_token']:.3f} bits/tok; best single block {best['layer']} recovers "
                  f"{best['recovered_bits_per_token']:.3f}", flush=True)
        print(f"[mechanism {c.key()} k={args.k}]: {len(rows)} new rows ({time.time() - t0:.0f}s)", flush=True)
    return args.results
