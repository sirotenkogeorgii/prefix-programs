"""Regime 2 (code -> output): the input is replaced by m optimizable vectors per example, placed after the shared
cell prefix (if any) and before `sep + target`. The m vectors are plain extra prefix positions for the scorer,
optimized per example. References per example: no input (m = 0) and the real input (regime 1), both with the
same shared prefix."""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from .data.formats import Cond, build_example
from .prefix import SoftPrefix
from .results import ResultsTable, load_prefix
from .scoring import collate, target_bits, target_nll_per_token
from .tasks import TASKS, Cell
from .train import _device_of, _lr_factor, evaluate


@dataclass
class Regime2Cfg:
    steps: int = 200
    lr: float = 0.1
    warmup: int = 10
    clip: float = 1.0
    batch_size: int = 32
    pad_to: int | None = None


def pooled_init(emb: torch.Tensor, m: int) -> torch.Tensor:
    """Average the input's token embeddings in m contiguous groups; cycle tokens when there are fewer than m."""
    T = emb.shape[0]
    if T >= m:
        return torch.stack([c.mean(0) for c in torch.tensor_split(emb, m)])
    return torch.stack([emb[i % T] for i in range(m)])


def _tails(tok, pairs, cell: Cell):
    sep = TASKS[cell.task]["sep"]
    return [build_example(tok, "", y, Cond(), sep=sep) for _, y in pairs]


def _real(tok, pairs, cell: Cell):
    sep = TASKS[cell.task]["sep"]
    return [build_example(tok, x, y, Cond(), sep=sep) for x, y in pairs]


def compress_inputs(model, tok, pairs, cell: Cell, m: int, shared_prefix: torch.Tensor | None, cfg: Regime2Cfg,
                    seed: int = 0) -> dict:
    device = _device_of(model)
    E = model.get_input_embeddings().weight.detach()
    pad = tok.pad_token_id
    tails, real = _tails(tok, pairs, cell), _real(tok, pairs, cell)
    k = 0 if shared_prefix is None else shared_prefix.shape[0]
    shared = None if shared_prefix is None else shared_prefix.detach().to(device)
    no_input = evaluate(model, (lambda b, sample=False: shared.unsqueeze(0).expand(b, -1, -1)) if shared is not None else None,
                        tails, pad, cfg.batch_size, pad_to=cfg.pad_to)
    real_ev = evaluate(model, (lambda b, sample=False: shared.unsqueeze(0).expand(b, -1, -1)) if shared is not None else None,
                       real, pad, cfg.batch_size, pad_to=cfg.pad_to)
    bits_after, n_tokens = [], []
    torch.manual_seed(seed)
    for i in range(0, len(pairs), cfg.batch_size):
        chunk_pairs, chunk_tails = pairs[i : i + cfg.batch_size], tails[i : i + cfg.batch_size]
        batch = collate(chunk_tails, pad, cfg.pad_to)
        B = batch.input_ids.shape[0]
        init = torch.stack([pooled_init(E[torch.tensor(tok(x, add_special_tokens=False)["input_ids"])].float(), m) for x, _ in chunk_pairs])
        C = torch.nn.Parameter(init.to(device))
        opt = torch.optim.Adam([C], lr=cfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg.warmup, cfg.steps))
        per_tok = batch.target_mask.sum(dim=1).clamp(min=1).to(device)

        def prefix_embeds():
            return torch.cat([shared.unsqueeze(0).expand(B, -1, -1), C], dim=1) if shared is not None else C

        for _ in range(cfg.steps):
            nll = target_nll_per_token(model, batch, prefix_embeds())
            loss = (nll.sum(dim=1) / per_tok).sum()  # examples are independent and equally weighted
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([C], cfg.clip)
            opt.step()
            sched.step()
        with torch.no_grad():
            bits_after.extend(target_bits(model, batch, prefix_embeds().detach()).tolist())
            n_tokens.extend(batch.target_mask.sum(dim=1).tolist())
    return {
        "m": m, "k": k,
        "bits_after": bits_after, "bits_no_input": no_input.per_example_bits, "bits_real_input": real_ev.per_example_bits,
        "n_tokens": n_tokens, "target_masks": [e.target_mask for e in tails],
    }


def run_regime2(model, tok, cell: Cell, pairs, ms: list[int], shared_ckpt, k: int, cond: str, cfg: Regime2Cfg,
                results: ResultsTable, seed: int = 0) -> list[dict]:
    shared = None
    if shared_ckpt is not None:
        E = model.get_input_embeddings().weight
        shared = load_prefix(SoftPrefix(k, E.shape[1]), shared_ckpt).weight.detach()
    rows = []
    for m in ms:
        key = dict(design="regime2", cell=cell.key(), m=m, k=k, cond=cond, seed=seed)
        if results.is_done(**key):
            continue
        res = compress_inputs(model, tok, pairs, cell, m, shared, cfg, seed)
        n = max(sum(res["n_tokens"]), 1)
        run_id = f"regime2_{cell.key()}_{cond}_k{k}_m{m}_s{seed}"
        per_dir = results.path.parent / "per_example"
        per_dir.mkdir(parents=True, exist_ok=True)
        per_path = per_dir / f"{run_id}.json"
        per_path.write_text(json.dumps({kk: [round(v, 6) for v in res[kk]] for kk in ("bits_after", "bits_no_input", "bits_real_input")}
                                       | {"tokens": res["n_tokens"]}))
        row = {
            **key, "run_id": run_id, "n_examples": len(pairs), "steps": cfg.steps, "lr": cfg.lr,
            "bits_per_token_after": sum(res["bits_after"]) / n,
            "bits_per_token_no_input": sum(res["bits_no_input"]) / n,
            "bits_per_token_real_input": sum(res["bits_real_input"]) / n,
            "n_tokens": n, "per_example": str(per_path),
        }
        results.append(row)
        rows.append(row)
    return rows
