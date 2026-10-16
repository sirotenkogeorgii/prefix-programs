"""Prompt tuning of a soft prefix against a frozen model.

Loss per step = mean target NLL (nats/token) + beta * KL(prefix) / N_train_target_tokens,
which at beta = 1 is the two-part MDL objective per token. Evaluation uses the
posterior mean. The best dev checkpoint is returned as a CPU state_dict.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch

from .scoring import LN2, collate, target_bits, target_bits_per_token, target_nll_per_token


@dataclass
class TrainCfg:
    steps: int = 1500
    batch_size: int = 16
    lr: float = 5e-2
    warmup: int = 50
    clip: float = 1.0
    eval_every: int = 100
    seed: int = 0
    beta: float = 0.0  # weight on the prefix KL (variational prefixes)
    sample: bool = True  # variational: inject posterior noise during training
    select: str = "dev"  # checkpoint rule: "dev" = lowest dev bits/token; "objective" = lowest dev bits + beta*KL/n_train_tokens
    dev_sample: bool = False  # score dev with posterior samples (the coded object) instead of the posterior mean
    eval_batch_size: int = 32
    pad_to: int | None = None  # bucket padded length to a multiple (fixed shapes for MPS)


@dataclass
class EvalResult:
    bits_total: float
    n_tokens: int
    per_example_bits: list[float] = field(default_factory=list)
    per_example_tokens: list[int] = field(default_factory=list)

    @property
    def bits_per_token(self) -> float:
        return self.bits_total / max(self.n_tokens, 1)


@dataclass
class TrainResult:
    best_state: dict
    best_dev_bits_per_token: float
    best_step: int
    history: list[dict]
    steps_run: int


def _device_of(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def evaluate(model, prefix, examples, pad_id: int, batch_size: int = 32, sample: bool = False,
             pad_to: int | None = None) -> EvalResult:
    model.eval()
    total, n_tok, per_ex, per_tok = 0.0, 0, [], []
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i : i + batch_size], pad_id, pad_to)
        pe = prefix(batch.input_ids.shape[0], sample=sample) if prefix is not None else None
        bits = target_bits(model, batch, pe)
        per_ex.extend(bits.tolist())
        per_tok.extend(batch.target_mask.sum(dim=1).tolist())
        total += bits.sum().item()
        n_tok += batch.n_targets
    return EvalResult(total, n_tok, per_ex, per_tok)


@torch.no_grad()
def copy_split_bits(model, prefix, examples, pad_id: int, batch_size: int = 32, pad_to: int | None = None) -> dict:
    """Total test bits split into copyable and non-copyable target tokens."""
    bits_c = bits_n = 0.0
    n_c = n_n = 0
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        batch = collate(chunk, pad_id, pad_to)
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


def _lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def train_prefix(model, prefix, train_examples, dev_examples, pad_id: int, cfg: TrainCfg) -> TrainResult:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = _device_of(model)
    prefix.to(device)
    model.eval()
    opt = torch.optim.AdamW(prefix.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg.warmup, cfg.steps))
    n_train_tokens = max(sum(e.n_target_tokens for e in train_examples), 1)

    order: list[int] = []
    def next_batch():
        nonlocal order
        if len(order) < cfg.batch_size:
            order += torch.randperm(len(train_examples), generator=gen).tolist()
        idx, order = order[: cfg.batch_size], order[cfg.batch_size :]
        return collate([train_examples[i] for i in idx], pad_id, cfg.pad_to)

    history: list[dict] = []
    best_score, best_bits, best_state, best_step = math.inf, math.inf, None, -1
    running, n_running = 0.0, 0
    for step in range(1, cfg.steps + 1):
        batch = next_batch()
        pe = prefix(batch.input_ids.shape[0], sample=cfg.sample)
        nll = target_nll_per_token(model, batch, pe)
        loss = nll.sum() / batch.n_targets + cfg.beta * prefix.kl_nats() / n_train_tokens
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix.parameters(), cfg.clip)
        opt.step()
        sched.step()
        running += loss.item(); n_running += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(model, prefix, dev_examples, pad_id, cfg.eval_batch_size, sample=cfg.dev_sample, pad_to=cfg.pad_to)
            kl_bits = prefix.kl_bits().item()
            objective = ev.bits_per_token + cfg.beta * kl_bits / n_train_tokens
            entry = {
                "step": step,
                "train_nats_per_token": running / max(n_running, 1),
                "dev_bits_per_token": ev.bits_per_token,
                "kl_bits": kl_bits,
                "dev_objective": objective,
            }
            history.append(entry)
            running, n_running = 0.0, 0
            score = objective if cfg.select == "objective" else ev.bits_per_token
            if score < best_score:
                best_score, best_bits, best_step = score, ev.bits_per_token, step
                best_state = {k: v.detach().cpu().clone() for k, v in prefix.state_dict().items()}
    return TrainResult(best_state, best_bits, best_step, history, cfg.steps)


def train_steps(model, prefix, examples, pad_id: int, cfg: TrainCfg) -> float:
    """Optimize the prefix in place for cfg.steps on `examples` with no evaluation or model selection.
    Used by the prequential coder, where peeking at held-out data would leak. Returns the mean train loss."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = _device_of(model)
    prefix.to(device)
    model.eval()
    opt = torch.optim.AdamW(prefix.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg.warmup, cfg.steps))
    n_tokens = max(sum(e.n_target_tokens for e in examples), 1)
    order: list[int] = []
    total = 0.0
    for step in range(cfg.steps):
        if len(order) < cfg.batch_size:
            order += torch.randperm(len(examples), generator=gen).tolist()
        idx, order = order[: cfg.batch_size], order[cfg.batch_size :]
        batch = collate([examples[i] for i in idx], pad_id, cfg.pad_to)
        nll = target_nll_per_token(model, batch, prefix(batch.input_ids.shape[0], sample=cfg.sample))
        loss = nll.sum() / batch.n_targets + cfg.beta * prefix.kl_nats() / n_tokens
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix.parameters(), cfg.clip)
        opt.step()
        sched.step()
        total += loss.item()
    return total / max(cfg.steps, 1)
