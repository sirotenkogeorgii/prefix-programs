"""What is recorded so far: row counts per table, design and conditioning, against the planned grid."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = ("sweep", "flat", "hier_task", "hier_lang", "compose", "distance", "regime2", "mechanism")


def table_summary(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return [f"{path.name}: missing"]
    df = pd.read_csv(path)
    cols = [c for c in ("design", "cond", "group", "probe") if c in df.columns]
    lines = [f"{path.name}: {len(df)} rows"]
    if cols:
        counts = df.groupby(cols).size()
        lines += [f"  {' '.join(str(x) for x in (idx if isinstance(idx, tuple) else (idx,)))}: {n}" for idx, n in counts.items()]
    return lines


def status_report(results_dir="results") -> str:
    d = Path(results_dir)
    lines = [f"results dir: {d.resolve()}"]
    for t in TABLES:
        lines += table_summary(d / f"{t}.csv")
    ckpts = Path("checkpoints")
    n = len(list(ckpts.glob("*.pt"))) if ckpts.exists() else 0
    lines.append(f"checkpoints: {n} files")
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Row counts of every results table")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)
    print(status_report(args.results_dir))
