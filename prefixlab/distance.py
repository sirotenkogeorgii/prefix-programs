"""Information distance between cells: K(a|b) = prequential bits to reach cell a starting from cell b's prefix,
over a fixed budget of a's training examples. `scratch` rows give the from-nothing reference."""
from __future__ import annotations

import json

from .prefix import SoftPrefix
from .prequential import prequential_bits
from .results import ResultsTable, load_prefix
from .train import TrainCfg, _device_of, evaluate


def run_distance(model, tok, cells: dict, k: int, cond: str, cfg: TrainCfg, results: ResultsTable,
                 budget: int = 512, first_block: int = 16, seed: int = 0) -> list[dict]:
    """cells: {cell_key: {"cell": Cell, "ckpt": path to a Design-A prefix, "data": {train, dev, test}}}."""
    E = model.get_input_embeddings().weight
    pad, bs = tok.pad_token_id, cfg.eval_batch_size
    rows = []
    for dst, info in cells.items():
        for src in [*cells, "scratch"]:
            key = dict(design="distance", src=src, dst=dst, k=k, cond=cond, budget=budget, seed=seed)
            if results.is_done(**key):
                continue
            if src == "scratch":
                prefix = SoftPrefix.from_vocab(E, k, seed)
            else:
                prefix = load_prefix(SoftPrefix(k, E.shape[1]), cells[src]["ckpt"])
            prefix.to(_device_of(model))
            data = info["data"]
            zero = evaluate(model, prefix, data["test"], pad, bs, pad_to=cfg.pad_to).bits_per_token
            preq = prequential_bits(model, prefix, data["train"][:budget], pad, cfg, first_block)
            after = evaluate(model, prefix, data["test"], pad, bs, pad_to=cfg.pad_to).bits_per_token
            row = {
                **key, "cell": dst,
                "run_id": f"distance_{cond}_k{k}_{src}_to_{dst}_b{budget}_s{seed}",
                "preq_bits": preq.total_bits, "preq_tokens": preq.n_tokens,
                "preq_bits_per_token": preq.total_bits / max(preq.n_tokens, 1),
                "learning_cost_bits": preq.learning_cost_bits,
                "zero_shot_test_bits_per_token": zero, "test_bits_per_token_after": after,
                "n_blocks": len(preq.block_bits), "block_bits": json.dumps([round(b, 3) for b in preq.block_bits]),
                "steps_per_block": cfg.steps,
            }
            results.append(row)
            rows.append(row)
    return rows
