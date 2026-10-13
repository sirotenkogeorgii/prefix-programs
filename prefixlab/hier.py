"""Design hier: a shared centre per group (e.g. a task) with one deviation per member (e.g. a language),
trained jointly with round-robin member batches. Loss per step for member m:

    mean NLL_m + beta * ( KL(centre) / N_total_tokens + KL(dev_m) / N_m_tokens )

so at beta = 1 the centre is paid once over all members' data and each deviation once over its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .prefix import HierarchicalPrefix, _sample_vocab_rows, gaussian_prior_from_embeddings
from .results import ResultsTable, save_prefix
from .scoring import collate, target_nll_per_token
from .sweep import N_EVAL_SAMPLES, _common, _save_per_example, _test_fields
from .tasks import Cell
from .train import TrainCfg, _device_of, _lr_factor, copy_split_bits, evaluate


@dataclass(frozen=True)
class HierSpec:
    group: str  # e.g. "task:translation" (members are languages) or "lang:ru" (members are tasks)
    k: int
    seed: int
    cond: str = "none"
    beta: float = 1.0
    tau: float = 1.0

    def key(self) -> dict:
        return dict(design="hier", group=self.group, k=self.k, seed=self.seed, cond=self.cond, beta=self.beta, tau=self.tau)

    def run_id(self) -> str:
        return f"hier_{self.group.replace(':', '-')}_{self.cond}_k{self.k}_s{self.seed}_b{self.beta:g}_t{self.tau:g}"


class MemberView:
    """Binds a HierarchicalPrefix to one member so it quacks like a plain prefix: view(batch_size, sample)."""

    def __init__(self, prefix: HierarchicalPrefix, member: str):
        self.prefix, self.member = prefix, member

    def __call__(self, batch_size: int, sample: bool = False) -> torch.Tensor:
        return self.prefix(self.member, batch_size, sample)


@dataclass
class HierResult:
    best_state: dict
    best_objective: float  # token-weighted dev bits/token over members, at the posterior mean
    best_step: int
    history: list


def train_hier(model, prefix: HierarchicalPrefix, data_by_member: dict, pad_id: int, cfg: TrainCfg) -> HierResult:
    torch.manual_seed(cfg.seed)
    members = prefix.members
    gens = {m: torch.Generator().manual_seed(cfg.seed + i) for i, m in enumerate(members)}
    device = _device_of(model)
    prefix.to(device)
    model.eval()
    opt = torch.optim.AdamW(prefix.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg.warmup, cfg.steps))
    n_tok = {m: max(sum(e.n_target_tokens for e in data_by_member[m]["train"]), 1) for m in members}
    n_total = sum(n_tok.values())
    orders = {m: [] for m in members}

    def next_batch(m):
        train = data_by_member[m]["train"]
        if len(orders[m]) < cfg.batch_size:
            orders[m] += torch.randperm(len(train), generator=gens[m]).tolist()
        idx, orders[m] = orders[m][: cfg.batch_size], orders[m][cfg.batch_size :]
        return collate([train[i] for i in idx], pad_id)

    def dev_objective():
        bits = toks = 0.0
        per = {}
        for m in members:
            ev = evaluate(model, MemberView(prefix, m), data_by_member[m]["dev"], pad_id, cfg.eval_batch_size)
            per[m] = ev.bits_per_token
            bits += ev.bits_total; toks += ev.n_tokens
        return bits / max(toks, 1), per

    history, best, best_state, best_step = [], math.inf, None, -1
    running, n_running = 0.0, 0
    for step in range(1, cfg.steps + 1):
        m = members[(step - 1) % len(members)]
        batch = next_batch(m)
        pe = prefix(m, batch.input_ids.shape[0], sample=cfg.sample)
        nll = target_nll_per_token(model, batch, pe)
        kl = prefix.kl_centre_nats() / n_total + prefix.kl_dev_nats()[m] / n_tok[m]
        loss = nll.sum() / batch.n_targets + cfg.beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix.parameters(), cfg.clip)
        opt.step()
        sched.step()
        running += loss.item(); n_running += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            obj, per = dev_objective()
            history.append({"step": step, "train_loss": running / max(n_running, 1), "dev_bits_per_token": obj,
                            "per_member": per, "kl_centre_bits": prefix.kl_centre_bits().item(),
                            "kl_dev_bits": {mm: v.item() for mm, v in prefix.kl_dev_bits().items()}})
            running, n_running = 0.0, 0
            if obj < best:
                best, best_step = obj, step
                best_state = {kk: v.detach().cpu().clone() for kk, v in prefix.state_dict().items()}
    return HierResult(best_state, best, best_step, history)


def run_hier(model, tok, spec: HierSpec, members: dict[str, Cell], data_by_member: dict, cfg: TrainCfg,
             results: ResultsTable, ckpt_dir) -> list[dict]:
    base = spec.key()
    todo = {m: c for m, c in members.items() if not results.is_done(cell=c.key(), **base)}
    if not todo:
        return []
    E = model.get_input_embeddings().weight
    prefix = HierarchicalPrefix(spec.k, E.shape[1], members=list(members), prior=gaussian_prior_from_embeddings(E),
                                tau=spec.tau, init_centre=_sample_vocab_rows(E, spec.k, spec.seed))
    cfg = replace(cfg, beta=spec.beta, sample=True)
    res = train_hier(model, prefix, data_by_member, tok.pad_token_id, cfg)
    prefix.load_state_dict(res.best_state)
    ckpt = Path(ckpt_dir) / f"{spec.run_id()}.pt"
    save_prefix(prefix, ckpt)
    kl_c = prefix.kl_centre_bits().item()
    kl_d = {m: v.item() for m, v in prefix.kl_dev_bits().items()}
    pad, bs = tok.pad_token_id, cfg.eval_batch_size
    rows = []
    for m, cell in todo.items():
        view, data = MemberView(prefix, m), data_by_member[m]
        dev = evaluate(model, view, data["dev"], pad, bs)
        test = evaluate(model, view, data["test"], pad, bs)
        sampled = [evaluate(model, view, data["test"], pad, bs, sample=True).bits_per_token for _ in range(N_EVAL_SAMPLES)]
        run_id = f"{spec.run_id()}_{m}"
        row = {
            "cell": cell.key(), **base, "member": m, "run_id": run_id, **_common(model, data),
            "steps": cfg.steps, "best_step": res.best_step,
            "dev_bits_per_token": dev.bits_per_token,
            **_test_fields(test, copy_split_bits(model, view, data["test"], pad, bs)),
            "kl_centre_bits": kl_c, "kl_dev_bits": kl_d[m], "kl_bits": kl_c + kl_d[m],
            "test_bits_per_token_sampled": sum(sampled) / len(sampled),
            "ckpt": str(ckpt),
            "per_example": _save_per_example(results, run_id, test.per_example_bits, test.per_example_tokens),
        }
        results.append(row)
        rows.append(row)
    return rows
