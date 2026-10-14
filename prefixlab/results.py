"""Resumable results: one CSV row per finished run, plus tiny prefix checkpoints.

Scripts call `is_done(**key)` before a run and `append(row)` after it, so a
Colab session can die anywhere and a rerun skips what is already recorded.
Writes are atomic (temp file + replace).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import torch


class ResultsTable:
    def __init__(self, path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(self.path)

    def is_done(self, **key) -> bool:
        df = self.load()
        if df.empty or any(c not in df.columns for c in key):
            return False
        mask = pd.Series(True, index=df.index)
        for c, v in key.items():
            mask &= df[c].astype(str) == str(v)
        return bool(mask.any())

    def append(self, row: dict) -> None:
        self.append_many([row])

    def append_many(self, rows: list[dict]) -> None:
        """One atomic write for a group of rows that must land together (e.g. all layers of a probe)."""
        if not rows:
            return
        df = pd.concat([self.load(), pd.DataFrame(rows)], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)


def save_prefix(prefix: torch.nn.Module, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in prefix.state_dict().items()}, path)


def load_prefix(prefix: torch.nn.Module, path) -> torch.nn.Module:
    prefix.load_state_dict(torch.load(Path(path), map_location="cpu"))
    return prefix
