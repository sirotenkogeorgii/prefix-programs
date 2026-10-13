"""Prequential (online) code length: each block is coded by the prefix trained on the blocks before it.

total − final is the cost of learning the prefix (Blier & Ollivier 2018; Voita & Titov 2020): the bits
spent before the prefix knew the data, relative to re-coding the same data with the final prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .train import TrainCfg, evaluate, train_steps


def doubling_blocks(n: int, first: int = 16) -> list[tuple[int, int]]:
    """Blocks whose cumulative extent doubles: [0,first), [first,2*first), [2*first,4*first), ... up to n."""
    blocks, start, end = [], 0, min(first, n)
    while start < n:
        blocks.append((start, end))
        start, end = end, min(2 * end, n)
    return blocks


@dataclass
class PrequentialResult:
    total_bits: float
    n_tokens: int
    block_bits: list[float]
    block_tokens: list[int]
    final_bits: float  # all examples re-coded with the final prefix
    learning_cost_bits: float  # total_bits - final_bits


def prequential_bits(model, prefix, examples, pad_id: int, cfg: TrainCfg, first_block: int = 16) -> PrequentialResult:
    total, bb, bt = 0.0, [], []
    for i, (s, e) in enumerate(doubling_blocks(len(examples), first_block)):
        ev = evaluate(model, prefix, examples[s:e], pad_id, cfg.eval_batch_size, pad_to=cfg.pad_to)
        total += ev.bits_total
        bb.append(ev.bits_total)
        bt.append(ev.n_tokens)
        train_steps(model, prefix, examples[:e], pad_id, replace(cfg, seed=cfg.seed + i))
    final = evaluate(model, prefix, examples, pad_id, cfg.eval_batch_size, pad_to=cfg.pad_to).bits_total
    return PrequentialResult(total, sum(bt), bb, bt, final, total - final)
