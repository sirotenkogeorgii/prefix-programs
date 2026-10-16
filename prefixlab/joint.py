"""Joint training of one module over several members (cells) with round-robin batches.

Shared by the hierarchical design (centre + deviations) and the composed design (task part + language part).
`forward(member, batch_size, sample)` yields that member's prefix; `kl_for(member)` its KL term in nats, already
divided by the token counts it is amortized over; `log_extra()` adds fields to each history entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .scoring import collate, target_nll_per_token
from .train import TrainCfg, _device_of, _lr_factor, evaluate


@dataclass
class JointResult:
    best_state: dict
    best_objective: float  # token-weighted dev bits/token over members, posterior mean
    best_step: int
    history: list


class BoundPrefix:
    """Binds a joint module to one member so it quacks like a plain prefix: bound(batch_size, sample)."""

    def __init__(self, forward: Callable, member):
        self.forward, self.member = forward, member

    def __call__(self, batch_size: int, sample: bool = False) -> torch.Tensor:
        return self.forward(self.member, batch_size, sample)


def train_joint(model, module: torch.nn.Module, members: list, data_by_member: dict, pad_id: int, cfg: TrainCfg,
                forward: Callable, kl_for: Callable | None = None, log_extra: Callable[[], dict] | None = None,
                eval_members: list | None = None) -> JointResult:
    """cfg.select == "objective" selects the checkpoint by the token-weighted dev bits/token plus beta times the
    members' KL terms (kl_for, nats per token, converted to bits), i.e. the training objective on dev; cfg.dev_sample
    scores dev with posterior samples. The defaults ("dev", False) are the preregistered rule."""
    torch.manual_seed(cfg.seed)
    gens = {m: torch.Generator().manual_seed(cfg.seed + i) for i, m in enumerate(members)}
    device = _device_of(model)
    module.to(device)
    model.eval()
    opt = torch.optim.AdamW(module.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg.warmup, cfg.steps))
    orders = {m: [] for m in members}
    eval_members = eval_members or members

    def next_batch(m):
        train = data_by_member[m]["train"]
        if len(orders[m]) < cfg.batch_size:
            orders[m] += torch.randperm(len(train), generator=gens[m]).tolist()
        idx, orders[m] = orders[m][: cfg.batch_size], orders[m][cfg.batch_size :]
        return collate([train[i] for i in idx], pad_id, cfg.pad_to)

    def dev_objective():
        bits = toks = 0.0
        per = {}
        kl_term = 0.0
        for m in eval_members:
            ev = evaluate(model, BoundPrefix(forward, m), data_by_member[m]["dev"], pad_id, cfg.eval_batch_size,
                          sample=cfg.dev_sample, pad_to=cfg.pad_to)
            per[str(m)] = ev.bits_per_token
            bits += ev.bits_total; toks += ev.n_tokens
            if kl_for is not None:
                kl_term += ev.n_tokens * float(kl_for(m)) / math.log(2)  # member's KL rate in bits/token, weighted by its dev tokens
        dev_bits = bits / max(toks, 1)
        return dev_bits, per, dev_bits + cfg.beta * kl_term / max(toks, 1)

    history, best_score, best, best_state, best_step = [], math.inf, math.inf, None, -1
    running, n_running = 0.0, 0
    for step in range(1, cfg.steps + 1):
        m = members[(step - 1) % len(members)]
        batch = next_batch(m)
        nll = target_nll_per_token(model, batch, forward(m, batch.input_ids.shape[0], cfg.sample))
        loss = nll.sum() / batch.n_targets
        if kl_for is not None:
            loss = loss + cfg.beta * kl_for(m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.clip)
        opt.step()
        sched.step()
        running += loss.item(); n_running += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            dev_bits, per, objective = dev_objective()
            entry = {"step": step, "train_loss": running / max(n_running, 1), "dev_bits_per_token": dev_bits,
                     "dev_objective": objective, "per_member": per}
            if log_extra is not None:
                entry.update(log_extra())
            history.append(entry)
            running, n_running = 0.0, 0
            score = objective if cfg.select == "objective" else dev_bits
            if score < best_score:
                best_score, best, best_step = score, dev_bits, step
                best_state = {kk: v.detach().cpu().clone() for kk, v in module.state_dict().items()}
    return JointResult(best_state, best, best_step, history)
