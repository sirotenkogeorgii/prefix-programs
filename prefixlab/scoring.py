"""Teacher-forced code length of target tokens, with an optional soft prefix.

Positions: the prefix occupies 0..k-1, the text token i sits at k+i and is
predicted by the logits at k+i-1. Code length is computed in float32 from the
model's logits whatever their dtype. Log-softmax is taken in chunks over
positions so the float32 logits never exceed a bounded memory footprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
_CHUNK = 32  # positions per log-softmax chunk


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, T] right-padded
    attention_mask: torch.Tensor  # [B, T]
    target_mask: torch.Tensor  # [B, T] bool

    def to(self, device) -> "Batch":
        return Batch(self.input_ids.to(device), self.attention_mask.to(device), self.target_mask.to(device))

    @property
    def n_targets(self) -> int:
        return int(self.target_mask.sum().item())


def collate(examples, pad_id: int) -> Batch:
    T = max(len(e.input_ids) for e in examples)
    B = len(examples)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    tmask = torch.zeros((B, T), dtype=torch.bool)
    for b, e in enumerate(examples):
        n = len(e.input_ids)
        ids[b, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attn[b, :n] = 1
        tmask[b, :n] = torch.tensor(e.target_mask, dtype=torch.bool)
    return Batch(ids, attn, tmask)


def forward_logits(model, batch: Batch, prefix_embeds: torch.Tensor | None):
    """Run the model on [prefix | text]. Returns (logits [B, k+T, V], k)."""
    device = model.get_input_embeddings().weight.device
    batch = batch.to(device)
    emb = model.get_input_embeddings()(batch.input_ids)
    attn = batch.attention_mask
    k = 0
    if prefix_embeds is not None:
        if prefix_embeds.dim() == 2:
            prefix_embeds = prefix_embeds.unsqueeze(0)
        if prefix_embeds.shape[0] == 1 and emb.shape[0] > 1:
            prefix_embeds = prefix_embeds.expand(emb.shape[0], -1, -1)
        k = prefix_embeds.shape[1]
        emb = torch.cat([prefix_embeds.to(device=device, dtype=emb.dtype), emb], dim=1)
        attn = torch.cat([torch.ones((emb.shape[0], k), dtype=attn.dtype, device=device), attn], dim=1)
    out = model(inputs_embeds=emb, attention_mask=attn)
    return out.logits, k


def target_nll_per_token(model, batch: Batch, prefix_embeds: torch.Tensor | None) -> torch.Tensor:
    """Negative log-likelihood in nats per text position, float32, zero off-target. [B, T]"""
    logits, k = forward_logits(model, batch, prefix_embeds)
    device = logits.device
    ids = batch.input_ids.to(device)
    tmask = batch.target_mask.to(device)
    B, T = ids.shape
    nll = torch.zeros((B, T), dtype=torch.float32, device=device)
    # text position i is predicted by logits position k+i-1; position 0 has none when k == 0
    first = 1 if k == 0 else 0
    for start in range(first, T, _CHUNK):
        stop = min(start + _CHUNK, T)
        pred = logits[:, k + start - 1 : k + stop - 1].float()
        tgt = ids[:, start:stop]
        lp = torch.log_softmax(pred, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll[:, start:stop] = -lp
    return nll * tmask


def target_bits_per_token(model, batch: Batch, prefix_embeds: torch.Tensor | None) -> torch.Tensor:
    return target_nll_per_token(model, batch, prefix_embeds) / LN2


def target_bits(model, batch: Batch, prefix_embeds: torch.Tensor | None) -> torch.Tensor:
    """Total code length of each example's target in bits. [B]"""
    return target_bits_per_token(model, batch, prefix_embeds).sum(dim=1)
