"""Design hier: a shared centre per group (e.g. a task) with one deviation per member (e.g. a language),
trained jointly with round-robin member batches. Loss per step for member m:

    mean NLL_m + beta * ( KL(centre) / N_total_tokens + KL(dev_m) / N_m_tokens )

so at beta = 1 the centre is paid once over all members' data and each deviation once over its own.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .joint import BoundPrefix, JointResult, train_joint
from .prefix import HierarchicalPrefix, _sample_vocab_rows, gaussian_prior_from_embeddings
from .results import ResultsTable, save_prefix
from .sweep import N_EVAL_SAMPLES, _common, _save_history, _save_per_example, _test_fields
from .tasks import Cell
from .train import TrainCfg, copy_split_bits, evaluate


@dataclass(frozen=True)
class HierSpec:
    group: str  # e.g. "task:translation" (members are languages) or "lang:ru" (members are tasks)
    k: int
    seed: int
    cond: str = "none"
    beta: float = 1.0
    tau: float = 1.0
    tau_units: str = "prior"  # "prior": tau in units of the prior sigma; "abs": absolute embedding units (old runs)
    protocol: str = "prereg"  # "prereg": checkpoint by mean-dev distortion; "corrected": by dev objective, dev on samples (run with --lr 0.01)

    def key(self) -> dict:
        return dict(design="hier", group=self.group, k=self.k, seed=self.seed, cond=self.cond, beta=self.beta,
                    tau=self.tau, tau_units=self.tau_units, protocol=self.protocol)

    def run_id(self) -> str:
        return (f"hier_{self.group.replace(':', '-')}_{self.cond}_k{self.k}_s{self.seed}_b{self.beta:g}"
                f"_t{self.tau:g}{'p' if self.tau_units == 'prior' else 'a'}{'_c' if self.protocol == 'corrected' else ''}")


class MemberView(BoundPrefix):
    """Binds a HierarchicalPrefix to one member so it quacks like a plain prefix: view(batch_size, sample)."""

    def __init__(self, prefix: HierarchicalPrefix, member: str):
        super().__init__(prefix, member)


HierResult = JointResult


def train_hier(model, prefix: HierarchicalPrefix, data_by_member: dict, pad_id: int, cfg: TrainCfg) -> JointResult:
    members = prefix.members
    n_tok = {m: max(sum(e.n_target_tokens for e in data_by_member[m]["train"]), 1) for m in members}
    n_total = sum(n_tok.values())
    return train_joint(
        model, prefix, members, data_by_member, pad_id, cfg,
        forward=lambda m, b, s: prefix(m, b, s),
        kl_for=lambda m: prefix.kl_centre_nats() / n_total + prefix.kl_dev_nats()[m] / n_tok[m],
        log_extra=lambda: {"kl_centre_bits": prefix.kl_centre_bits().item(),
                           "kl_dev_bits": {mm: v.item() for mm, v in prefix.kl_dev_bits().items()}},
    )


def run_hier(model, tok, spec: HierSpec, members: dict[str, Cell], data_by_member: dict, cfg: TrainCfg,
             results: ResultsTable, ckpt_dir) -> list[dict]:
    base = spec.key()
    todo = {m: c for m, c in members.items() if not results.is_done(cell=c.key(), **base)}
    if not todo:
        return []
    E = model.get_input_embeddings().weight
    prefix = HierarchicalPrefix(spec.k, E.shape[1], members=list(members), prior=gaussian_prior_from_embeddings(E),
                                tau=spec.tau, tau_units=spec.tau_units, init_centre=_sample_vocab_rows(E, spec.k, spec.seed))
    cfg = replace(cfg, beta=spec.beta, sample=True,
                  **({"select": "objective", "dev_sample": True} if spec.protocol == "corrected" else {}))
    res = train_hier(model, prefix, data_by_member, tok.pad_token_id, cfg)
    prefix.load_state_dict(res.best_state)
    ckpt = Path(ckpt_dir) / f"{spec.run_id()}.pt"
    save_prefix(prefix, ckpt)
    kl_c = prefix.kl_centre_bits().item()
    kl_d = {m: v.item() for m, v in prefix.kl_dev_bits().items()}
    pad, bs = tok.pad_token_id, cfg.eval_batch_size
    history = _save_history(results, spec.run_id(), res.history)
    rows = []
    for m, cell in todo.items():
        view, data = MemberView(prefix, m), data_by_member[m]
        dev = evaluate(model, view, data["dev"], pad, bs, pad_to=cfg.pad_to)
        test = evaluate(model, view, data["test"], pad, bs, pad_to=cfg.pad_to)
        sampled = [evaluate(model, view, data["test"], pad, bs, sample=True, pad_to=cfg.pad_to).bits_per_token
                   for _ in range(N_EVAL_SAMPLES)]
        run_id = f"{spec.run_id()}_{m}"
        row = {
            "cell": cell.key(), **base, "member": m, "run_id": run_id, **_common(model, data),
            "steps": cfg.steps, "lr": cfg.lr, "select": cfg.select, "dev_sample": cfg.dev_sample, "best_step": res.best_step,
            "dev_bits_per_token": dev.bits_per_token,
            **_test_fields(test, copy_split_bits(model, view, data["test"], pad, bs, cfg.pad_to)),
            "kl_centre_bits": kl_c, "kl_dev_bits": kl_d[m], "kl_bits": kl_c + kl_d[m],
            "test_bits_per_token_sampled": sum(sampled) / len(sampled),
            "ckpt": str(ckpt),
            "per_example": _save_per_example(results, run_id, test.per_example_bits, test.per_example_tokens),
            "history": history,
        }
        results.append(row)
        rows.append(row)
    return rows
