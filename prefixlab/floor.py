"""Per-cell LoRA fine-tuning floor: the best code length the frozen model's weights can be nudged to.

The adapter is injected into the shared model, trained, evaluated, and then unloaded, so the
model object is bit-identical afterwards (verified by tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .scoring import collate, target_nll_per_token
from .train import _lr_factor, copy_split_bits, evaluate


@dataclass
class FloorCfg:
    steps: int = 1500
    batch_size: int = 16
    lr: float = 2e-4
    warmup: int = 50
    clip: float = 1.0
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    eval_every: int = 100
    seed: int = 0
    eval_batch_size: int = 32
    pad_to: int | None = None


@dataclass
class FloorResult:
    dev_bits_per_token: float
    test_bits_per_token: float
    test_bits_total: float
    n_test_tokens: int
    best_step: int
    copy_split: dict
    history: list = field(default_factory=list)
    per_example_bits: list = field(default_factory=list)
    per_example_tokens: list = field(default_factory=list)


def train_lora_floor(model, tok, data: dict, cfg: FloorCfg) -> FloorResult:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    pad = tok.pad_token_id
    lcfg = LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=list(cfg.targets),
                      bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lcfg)
    try:
        peft_model.eval()
        params = [p for p in peft_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg.warmup, cfg.steps))
        train = data["train"]
        order: list[int] = []
        best_bits, best_state, best_step, history = math.inf, None, -1, []
        running, n_running = 0.0, 0
        for step in range(1, cfg.steps + 1):
            if len(order) < cfg.batch_size:
                order += torch.randperm(len(train), generator=gen).tolist()
            idx, order = order[: cfg.batch_size], order[cfg.batch_size :]
            batch = collate([train[i] for i in idx], pad, cfg.pad_to)
            nll = target_nll_per_token(model, batch, None)
            loss = nll.sum() / batch.n_targets
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            sched.step()
            running += loss.item(); n_running += 1
            if step % cfg.eval_every == 0 or step == cfg.steps:
                ev = evaluate(model, None, data["dev"], pad, cfg.eval_batch_size, pad_to=cfg.pad_to)
                history.append({"step": step, "train_nats_per_token": running / max(n_running, 1), "dev_bits_per_token": ev.bits_per_token})
                running, n_running = 0.0, 0
                if ev.bits_per_token < best_bits:
                    best_bits, best_step = ev.bits_per_token, step
                    best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(peft_model).items()}
        set_peft_model_state_dict(peft_model, best_state)
        with torch.no_grad():
            test = evaluate(model, None, data["test"], pad, cfg.eval_batch_size, pad_to=cfg.pad_to)
            split = copy_split_bits(model, None, data["test"], pad, cfg.eval_batch_size, cfg.pad_to)
    finally:
        base = peft_model.unload()
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
    return FloorResult(best_bits, test.bits_per_token, test.bits_total, test.n_tokens, best_step, split, history,
                       test.per_example_bits, test.per_example_tokens)
