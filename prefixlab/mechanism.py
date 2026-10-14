"""Mechanism: in which blocks the prefix's information is read. For each probe (recipient prefix, donor prefix) the
attention input at the prefix positions is replaced by the donor's in one block at a time; the bits recovered
relative to the recipient run, per block, locate the read-out in depth. Probes:

    A                recipient = random prefix of the same k, donor = trained Design-A prefix
    hier_<g>_centre  recipient = random, donor = the hierarchical centre alone (shared part, e.g. the task)
    hier_<g>_dev     recipient = centre alone, donor = centre + member deviation (member part, e.g. the language)
"""
from __future__ import annotations

import torch

from .patching import window_patch_curve
from .prefix import SoftPrefix
from .results import ResultsTable
from .tasks import Cell
from .train import _device_of

RECIPIENT_SEED_OFFSET = 100


def random_prefix(E: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    return SoftPrefix.from_vocab(E, k, seed + RECIPIENT_SEED_OFFSET).weight.detach()


def hier_parts(ckpt_path, member: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(centre, centre + deviation[member]) posterior means from a HierarchicalPrefix checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu")
    centre = state["centre_mu"]
    return centre, centre + state[f"dev_mu.{member}"]


def run_mechanism(model, tok, cell: Cell, examples: list, probes: dict, k: int, seed: int, cond: str,
                  results: ResultsTable, batch_size: int = 32, pad_to: int | None = None, extra: dict | None = None) -> list[dict]:
    """probes: {name: (recipient_prefix, donor_prefix)}; one row per (probe, layer), written together."""
    device = _device_of(model)
    rows = []
    for name, (rec, don) in probes.items():
        key = dict(design="mechanism", cell=cell.key(), probe=name, k=k, seed=seed, cond=cond)
        if results.is_done(**key):
            continue
        curve = window_patch_curve(model, examples, tok.pad_token_id, rec.to(device), don.to(device), batch_size, pad_to)
        run_id = f"mechanism_{cell.key()}_{cond}_k{k}_s{seed}_{name}"
        new = [{
            **key, "layer": layer, "run_id": run_id, "n_examples": len(examples), "n_tokens": curve["n_tokens"],
            "recipient_bits_per_token": curve["recipient_bits_per_token"],
            "donor_bits_per_token": curve["donor_bits_per_token"],
            "all_layers_bits_per_token": curve["all_layers_bits_per_token"],
            "patched_bits_per_token": bits,
            "recovered_bits_per_token": curve["recovered_bits_per_token"][layer],
            **(extra or {}),
        } for layer, bits in curve["patched_bits_per_token"].items()]
        results.append_many(new)
        rows.extend(new)
    return rows
