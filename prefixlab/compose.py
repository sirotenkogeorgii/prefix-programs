"""Design compose: prefix(t, l) = [task part(t) | language part(l)], parts tied across the grid and trained jointly
on the non-held-out cells. Held-out cells are scored by composing parts that were never trained together,
which is the zero-shot composition test; the gap to a jointly trained prefix at the same total k is the
composition cost (algorithmic mutual information between task and language relative to the model)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from .joint import BoundPrefix, train_joint
from .prefix import _sample_vocab_rows
from .results import ResultsTable, save_prefix
from .sweep import _common, _save_per_example, _test_fields
from .tasks import Cell
from .train import TrainCfg, copy_split_bits, evaluate


@dataclass(frozen=True)
class ComposeSpec:
    k_task: int
    k_lang: int
    seed: int
    cond: str = "none"
    holdout: tuple = ()  # cells scored zero-shot, excluded from training

    def holdout_tag(self) -> str:
        return "+".join(sorted(c.key() for c in self.holdout)) or "none"

    def key(self) -> dict:
        return dict(design="compose", k=self.k_task + self.k_lang, k_task=self.k_task, k_lang=self.k_lang,
                    seed=self.seed, cond=self.cond, holdout=self.holdout_tag())

    def run_id(self) -> str:
        return f"compose_{self.cond}_kt{self.k_task}_kl{self.k_lang}_s{self.seed}_h{self.holdout_tag()}"


class ComposedPrefix(nn.Module):
    def __init__(self, tasks: list[str], langs: list[str], k_task: int, k_lang: int, E: torch.Tensor, seed: int):
        super().__init__()
        self.task_part = nn.ParameterDict({t: nn.Parameter(_sample_vocab_rows(E, k_task, seed * 1000 + i)) for i, t in enumerate(tasks)})
        self.lang_part = nn.ParameterDict({l: nn.Parameter(_sample_vocab_rows(E, k_lang, seed * 1000 + 500 + i)) for i, l in enumerate(langs)})

    def forward(self, cell: Cell, batch_size: int, sample: bool = False) -> torch.Tensor:
        z = torch.cat([self.task_part[cell.task], self.lang_part[cell.lang]], dim=0)
        return z.unsqueeze(0).expand(batch_size, -1, -1)

    def kl_nats(self) -> torch.Tensor:
        return next(iter(self.parameters())).new_zeros(())


def run_compose(model, tok, cells: dict, spec: ComposeSpec, cfg: TrainCfg, results: ResultsTable, ckpt_dir) -> list[dict]:
    """cells: {Cell: {train, dev, test}}. One row per cell not yet recorded."""
    base = spec.key()
    todo = [c for c in cells if not results.is_done(cell=c.key(), **base)]
    if not todo:
        return []
    E = model.get_input_embeddings().weight
    tasks, langs = sorted({c.task for c in cells}), sorted({c.lang for c in cells})
    prefix = ComposedPrefix(tasks, langs, spec.k_task, spec.k_lang, E, spec.seed)
    train_cells = [c for c in cells if c not in spec.holdout]
    cfg = replace(cfg, beta=0.0, sample=False)
    res = train_joint(model, prefix, train_cells, cells, tok.pad_token_id, cfg, forward=lambda c, b, s: prefix(c, b, s))
    prefix.load_state_dict(res.best_state)
    ckpt = Path(ckpt_dir) / f"{spec.run_id()}.pt"
    save_prefix(prefix, ckpt)
    pad, bs = tok.pad_token_id, cfg.eval_batch_size
    rows = []
    for cell in todo:
        view, data = BoundPrefix(prefix, cell), cells[cell]
        dev = evaluate(model, view, data["dev"], pad, bs, pad_to=cfg.pad_to)
        test = evaluate(model, view, data["test"], pad, bs, pad_to=cfg.pad_to)
        run_id = f"{spec.run_id()}_{cell.key()}"
        row = {
            "cell": cell.key(), **base, "held_out": cell in spec.holdout, "run_id": run_id, **_common(model, data),
            "steps": cfg.steps, "best_step": res.best_step, "dev_bits_per_token": dev.bits_per_token,
            **_test_fields(test, copy_split_bits(model, view, data["test"], pad, bs, cfg.pad_to)),
            "kl_bits": 0.0, "ckpt": str(ckpt),
            "per_example": _save_per_example(results, run_id, test.per_example_bits, test.per_example_tokens),
        }
        results.append(row)
        rows.append(row)
    return rows
