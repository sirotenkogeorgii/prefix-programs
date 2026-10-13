"""One resumable run of the prefix sweep: train, evaluate on test, record a CSV row + checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .prefix import SoftPrefix
from .results import ResultsTable, save_prefix
from .scoring import collate, target_bits_per_token
from .tasks import Cell
from .train import TrainCfg, evaluate, train_prefix


@dataclass(frozen=True)
class RunSpec:
    cell: Cell
    k: int
    seed: int
    cond: str = "none"  # which information the context already supplies: none | lang | task | both
    design: str = "A"  # A = plain sweep; later: flat / hier (variational), none = no-prefix baseline

    def key(self) -> dict:
        return dict(cell=self.cell.key(), k=self.k, seed=self.seed, cond=self.cond, design=self.design)

    def run_id(self) -> str:
        return f"{self.design}_{self.cell.key()}_{self.cond}_k{self.k}_s{self.seed}"


@torch.no_grad()
def copy_split_bits(model, prefix, examples, pad_id: int, batch_size: int = 32) -> dict:
    """Total test bits split into copyable and non-copyable target tokens."""
    bits_c = bits_n = 0.0
    n_c = n_n = 0
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        batch = collate(chunk, pad_id)
        pe = prefix(batch.input_ids.shape[0], sample=False) if prefix is not None else None
        pt = target_bits_per_token(model, batch, pe).cpu()
        cm = torch.zeros_like(batch.target_mask)
        for b, e in enumerate(chunk):
            cm[b, : len(e.copy_mask)] = torch.tensor(e.copy_mask, dtype=torch.bool)
        tm = batch.target_mask
        bits_c += pt[cm & tm].sum().item()
        bits_n += pt[tm & ~cm].sum().item()
        n_c += int((cm & tm).sum())
        n_n += int((tm & ~cm).sum())
    return {
        "test_bits_copy": bits_c,
        "n_test_tokens_copy": n_c,
        "test_bits_noncopy": bits_n,
        "n_test_tokens_noncopy": n_n,
        "test_bits_per_token_noncopy": bits_n / max(n_n, 1),
    }


def _common(model, data) -> dict:
    return {
        "model": getattr(model, "name_or_path", "") or getattr(model.config, "_name_or_path", ""),
        "n_train": len(data["train"]),
        "n_dev": len(data["dev"]),
        "n_test": len(data["test"]),
    }


def run_one(model, tok, spec: RunSpec, data: dict, cfg: TrainCfg, results: ResultsTable, ckpt_dir) -> dict | None:
    key = spec.key()
    if results.is_done(**key):
        return None
    E = model.get_input_embeddings().weight
    prefix = SoftPrefix.from_vocab(E, spec.k, spec.seed)
    res = train_prefix(model, prefix, data["train"], data["dev"], tok.pad_token_id, cfg)
    prefix.load_state_dict(res.best_state)
    test = evaluate(model, prefix, data["test"], tok.pad_token_id, cfg.eval_batch_size)
    ckpt = Path(ckpt_dir) / f"{spec.run_id()}.pt"
    save_prefix(prefix, ckpt)
    row = {
        **key,
        "run_id": spec.run_id(),
        **_common(model, data),
        "steps": cfg.steps,
        "best_step": res.best_step,
        "dev_bits_per_token": res.best_dev_bits_per_token,
        "test_bits_per_token": test.bits_per_token,
        "test_bits_total": test.bits_total,
        "n_test_tokens": test.n_tokens,
        **copy_split_bits(model, prefix, data["test"], tok.pad_token_id, cfg.eval_batch_size),
        "kl_bits": 0.0,
        "ckpt": str(ckpt),
    }
    results.append(row)
    return row


def baseline_row(model, tok, cell: Cell, cond: str, data: dict, results: ResultsTable) -> dict | None:
    """No-prefix evaluation of a cell under a conditioning mode (k = 0)."""
    key = dict(cell=cell.key(), k=0, seed=0, cond=cond, design="none")
    if results.is_done(**key):
        return None
    dev = evaluate(model, None, data["dev"], tok.pad_token_id)
    test = evaluate(model, None, data["test"], tok.pad_token_id)
    row = {
        **key,
        "run_id": f"none_{cell.key()}_{cond}_k0_s0",
        **_common(model, data),
        "steps": 0,
        "best_step": 0,
        "dev_bits_per_token": dev.bits_per_token,
        "test_bits_per_token": test.bits_per_token,
        "test_bits_total": test.bits_total,
        "n_test_tokens": test.n_tokens,
        **copy_split_bits(model, None, data["test"], tok.pad_token_id),
        "kl_bits": 0.0,
        "ckpt": "",
    }
    results.append(row)
    return row
