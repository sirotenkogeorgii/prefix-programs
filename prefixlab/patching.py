"""Window patching of the prefix positions: in block l only, the attention sublayer's input at positions 0..k-1
is replaced by a donor run's values, so text positions read the donor's keys and values in that block and the
recipient's everywhere else. Patching every block at once reproduces the donor run exactly (tested)."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .scoring import LN2, collate, target_bits


def _layers(model):
    return model.model.layers


@contextmanager
def attn_input_prehooks(model, fns: dict):
    """fns: {layer: fn(hidden_states) -> hidden_states} applied to the attention input of each listed block."""
    handles = []
    for layer, fn in fns.items():
        def hook(module, args, kwargs, fn=fn):
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs, hidden_states=fn(kwargs["hidden_states"]))
            else:
                args = (fn(args[0]), *args[1:])
            return args, kwargs
        handles.append(_layers(model)[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture_attn_inputs(model, batch, prefix: torch.Tensor, layers: list[int], k: int) -> dict:
    store = {}
    fns = {}
    for l in layers:
        def fn(h, l=l):
            store[l] = h[:, :k].detach().clone()
            return h
        fns[l] = fn
    with attn_input_prehooks(model, fns):
        target_bits(model, batch, prefix)
    return store


@torch.no_grad()
def window_patch_bits(model, batch, recipient_prefix: torch.Tensor, donor_prefix: torch.Tensor, layer: int | list[int]) -> torch.Tensor:
    layers = [layer] if isinstance(layer, int) else list(layer)
    k = donor_prefix.shape[0]
    donor = capture_attn_inputs(model, batch, donor_prefix, layers, k)
    fns = {}
    for l in layers:
        def fn(h, l=l):
            h = h.clone()
            h[:, :k] = donor[l].to(h.dtype)
            return h
        fns[l] = fn
    with attn_input_prehooks(model, fns):
        return target_bits(model, batch, recipient_prefix)


@torch.no_grad()
def window_patch_curve(model, examples, pad_id: int, recipient_prefix: torch.Tensor, donor_prefix: torch.Tensor,
                       batch_size: int = 32, pad_to: int | None = None) -> dict:
    L = len(_layers(model))
    tot = {l: 0.0 for l in range(L)}
    rec = don = allp = 0.0
    n_tok = 0
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i : i + batch_size], pad_id, pad_to)
        n_tok += batch.n_targets
        rec += target_bits(model, batch, recipient_prefix).sum().item()
        don += target_bits(model, batch, donor_prefix).sum().item()
        allp += window_patch_bits(model, batch, recipient_prefix, donor_prefix, list(range(L))).sum().item()
        for l in range(L):
            tot[l] += window_patch_bits(model, batch, recipient_prefix, donor_prefix, l).sum().item()
    n = max(n_tok, 1)
    return {
        "n_tokens": n_tok,
        "recipient_bits_per_token": rec / n,
        "donor_bits_per_token": don / n,
        "all_layers_bits_per_token": allp / n,
        "patched_bits_per_token": {l: tot[l] / n for l in range(L)},
        "recovered_bits_per_token": {l: (rec - tot[l]) / n for l in range(L)},
    }
