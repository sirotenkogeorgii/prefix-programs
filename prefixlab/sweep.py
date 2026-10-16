"""Resumable runs: prefix sweep (Design A), no-prefix baselines and LoRA floors.

Each finished run becomes one CSV row plus a per-example JSON (`<results dir>/per_example/<run_id>.json`
with test bits and token counts per example) for bootstrap intervals later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .floor import FloorCfg, train_lora_floor
from .prefix import SoftPrefix, VariationalPrefix, gaussian_prior_from_embeddings
from .results import ResultsTable, save_prefix
from .tasks import Cell
from .train import TrainCfg, copy_split_bits, evaluate, train_prefix


@dataclass(frozen=True)
class RunSpec:
    cell: Cell
    k: int
    seed: int
    cond: str = "none"  # which information the context already supplies: none | lang | task | both
    design: str = "A"  # A = plain sweep; flat = variational with global prior; flat2 = flat with the corrected
    # protocol (checkpoint by dev objective, dev scored with samples; run with --lr 0.01); none = no-prefix; floor = LoRA
    beta: float = 0.0  # KL weight (flat only)

    def key(self) -> dict:
        return dict(cell=self.cell.key(), k=self.k, seed=self.seed, cond=self.cond, design=self.design, beta=self.beta)

    def run_id(self) -> str:
        tag = f"_b{self.beta:g}" if self.design in VARIATIONAL else ""
        return f"{self.design}_{self.cell.key()}_{self.cond}_k{self.k}_s{self.seed}{tag}"


N_EVAL_SAMPLES = 8
VARIATIONAL = ("flat", "flat2")


def _common(model, data) -> dict:
    return {
        "model": getattr(model, "name_or_path", "") or getattr(model.config, "_name_or_path", ""),
        "n_train": len(data["train"]),
        "n_dev": len(data["dev"]),
        "n_test": len(data["test"]),
    }


def _save_per_example(results: ResultsTable, run_id: str, bits: list, tokens: list) -> str:
    d = results.path.parent / "per_example"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.json"
    p.write_text(json.dumps({"bits": [round(b, 6) for b in bits], "tokens": [int(t) for t in tokens]}))
    return str(p)


def _save_history(results: ResultsTable, run_id: str, history: list) -> str:
    """Per-eval-step trajectory (dev bits/token, KL bits, ...) so rate-distortion points other than the selected
    checkpoint can be examined post hoc."""
    d = results.path.parent / "history"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.json"
    p.write_text(json.dumps(history))
    return str(p)


def _test_fields(test, split: dict) -> dict:
    return {
        "test_bits_per_token": test.bits_per_token,
        "test_bits_total": test.bits_total,
        "n_test_tokens": test.n_tokens,
        **split,
    }


def run_one(model, tok, spec: RunSpec, data: dict, cfg: TrainCfg, results: ResultsTable, ckpt_dir) -> dict | None:
    key = spec.key()
    if results.is_done(**key):
        return None
    E = model.get_input_embeddings().weight
    if spec.design == "A":
        prefix = SoftPrefix.from_vocab(E, spec.k, spec.seed)
        cfg = replace(cfg, beta=0.0, sample=False)
    elif spec.design == "flat":
        prefix = VariationalPrefix.from_vocab(E, spec.k, spec.seed, prior=gaussian_prior_from_embeddings(E))
        cfg = replace(cfg, beta=spec.beta, sample=True)
    elif spec.design == "flat2":
        prefix = VariationalPrefix.from_vocab(E, spec.k, spec.seed, prior=gaussian_prior_from_embeddings(E))
        cfg = replace(cfg, beta=spec.beta, sample=True, select="objective", dev_sample=True)
    else:
        raise ValueError(f"run_one handles designs A, flat and flat2, not {spec.design}")
    res = train_prefix(model, prefix, data["train"], data["dev"], tok.pad_token_id, cfg)
    prefix.load_state_dict(res.best_state)
    test = evaluate(model, prefix, data["test"], tok.pad_token_id, cfg.eval_batch_size, pad_to=cfg.pad_to)
    ckpt = Path(ckpt_dir) / f"{spec.run_id()}.pt"
    save_prefix(prefix, ckpt)
    row = {
        **key,
        "run_id": spec.run_id(),
        **_common(model, data),
        "steps": cfg.steps,
        "lr": cfg.lr,
        "select": cfg.select,
        "dev_sample": cfg.dev_sample,
        "best_step": res.best_step,
        "dev_bits_per_token": res.best_dev_bits_per_token,
        **_test_fields(test, copy_split_bits(model, prefix, data["test"], tok.pad_token_id, cfg.eval_batch_size, cfg.pad_to)),
        "kl_bits": prefix.kl_bits().item(),
        "ckpt": str(ckpt),
        "per_example": _save_per_example(results, spec.run_id(), test.per_example_bits, test.per_example_tokens),
        "history": _save_history(results, spec.run_id(), res.history),
    }
    if spec.design in VARIATIONAL:
        sampled = [evaluate(model, prefix, data["test"], tok.pad_token_id, cfg.eval_batch_size, sample=True,
                            pad_to=cfg.pad_to).bits_per_token for _ in range(N_EVAL_SAMPLES)]
        row["test_bits_per_token_sampled"] = sum(sampled) / len(sampled)
    results.append(row)
    return row


def baseline_row(model, tok, cell: Cell, cond: str, data: dict, results: ResultsTable,
                 pad_to: int | None = None, batch_size: int = 32) -> dict | None:
    """No-prefix evaluation of a cell under a conditioning mode (k = 0)."""
    key = dict(cell=cell.key(), k=0, seed=0, cond=cond, design="none")
    if results.is_done(**key):
        return None
    run_id = f"none_{cell.key()}_{cond}_k0_s0"
    dev = evaluate(model, None, data["dev"], tok.pad_token_id, batch_size, pad_to=pad_to)
    test = evaluate(model, None, data["test"], tok.pad_token_id, batch_size, pad_to=pad_to)
    row = {
        **key,
        "run_id": run_id,
        **_common(model, data),
        "steps": 0,
        "best_step": 0,
        "dev_bits_per_token": dev.bits_per_token,
        **_test_fields(test, copy_split_bits(model, None, data["test"], tok.pad_token_id, batch_size, pad_to)),
        "kl_bits": 0.0,
        "ckpt": "",
        "per_example": _save_per_example(results, run_id, test.per_example_bits, test.per_example_tokens),
    }
    results.append(row)
    return row


def floor_row(model, tok, cell: Cell, cond: str, data: dict, results: ResultsTable, fcfg: FloorCfg) -> dict | None:
    """LoRA fine-tuning floor for a cell under a conditioning mode (k = 0, design = floor)."""
    key = dict(cell=cell.key(), k=0, seed=fcfg.seed, cond=cond, design="floor")
    if results.is_done(**key):
        return None
    run_id = f"floor_{cell.key()}_{cond}_k0_s{fcfg.seed}"
    res = train_lora_floor(model, tok, data, fcfg)
    row = {
        **key,
        "run_id": run_id,
        **_common(model, data),
        "steps": fcfg.steps,
        "best_step": res.best_step,
        "dev_bits_per_token": res.dev_bits_per_token,
        "test_bits_per_token": res.test_bits_per_token,
        "test_bits_total": res.test_bits_total,
        "n_test_tokens": res.n_test_tokens,
        **res.copy_split,
        "kl_bits": 0.0,
        "ckpt": "",
        "per_example": _save_per_example(results, run_id, res.per_example_bits, res.per_example_tokens),
        "lora_r": fcfg.r,
        "lora_lr": fcfg.lr,
    }
    results.append(row)
    return row
