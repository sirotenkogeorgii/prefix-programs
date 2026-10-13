"""Teacher-forced code length in bits, with and without a soft prefix."""
import math

import torch

from prefixlab.data.formats import Cond, build_example
from prefixlab.scoring import collate, target_bits, target_bits_per_token


def _manual_bits(model, ids, mask, prefix=None):
    """Independent computation: -log2 softmax at the predicting position."""
    ids_t = torch.tensor([ids])
    emb = model.get_input_embeddings()(ids_t)
    k = 0
    if prefix is not None:
        emb = torch.cat([prefix.unsqueeze(0), emb], dim=1)
        k = prefix.shape[0]
    logits = model(inputs_embeds=emb).logits[0].float()
    total = 0.0
    for i, m in enumerate(mask):
        if m:
            lp = torch.log_softmax(logits[k + i - 1], dim=-1)[ids[i]].item()
            total += -lp / math.log(2)
    return total


def test_bits_match_manual_log_softmax_without_prefix(tok, tiny_model):
    ex = build_example(tok, "The cat sat on the mat.", "Кошка сидела на коврике.", Cond())
    batch = collate([ex], tok.pad_token_id)
    got = target_bits(tiny_model, batch, None)
    want = _manual_bits(tiny_model, ex.input_ids, ex.target_mask)
    assert got.shape == (1,)
    assert abs(got.item() - want) < 1e-3


def test_prefix_shifts_predicting_positions(tok, tiny_model):
    ex = build_example(tok, "The cat sat.", "Кошка сидела.", Cond())
    torch.manual_seed(1)
    prefix = torch.randn(3, tiny_model.config.hidden_size)
    batch = collate([ex], tok.pad_token_id)
    got = target_bits(tiny_model, batch, prefix.unsqueeze(0))
    want = _manual_bits(tiny_model, ex.input_ids, ex.target_mask, prefix)
    assert abs(got.item() - want) < 1e-3
    # and the prefix actually changes the answer
    assert abs(got.item() - _manual_bits(tiny_model, ex.input_ids, ex.target_mask)) > 1e-6


def test_padding_does_not_change_bits(tok, tiny_model):
    a = build_example(tok, "Short.", "Коротко.", Cond())
    b = build_example(tok, "A much longer input sentence here.", "Гораздо более длинное предложение здесь.", Cond())
    batch = collate([a, b], tok.pad_token_id)
    got = target_bits(tiny_model, batch, None)
    want_a = _manual_bits(tiny_model, a.input_ids, a.target_mask)
    want_b = _manual_bits(tiny_model, b.input_ids, b.target_mask)
    assert abs(got[0].item() - want_a) < 1e-3
    assert abs(got[1].item() - want_b) < 1e-3


def test_per_token_bits_sum_to_total_and_vanish_off_target(tok, tiny_model):
    ex = build_example(tok, "The cat sat.", "Кошка сидела.", Cond(lang_label="Russian:"))
    batch = collate([ex], tok.pad_token_id)
    per_tok = target_bits_per_token(tiny_model, batch, None)
    total = target_bits(tiny_model, batch, None)
    assert per_tok.shape == batch.input_ids.shape
    assert torch.allclose(per_tok.sum(dim=1), total, atol=1e-4)
    assert torch.all(per_tok[0][~batch.target_mask[0]] == 0)
    assert torch.all(per_tok[0][batch.target_mask[0]] > 0)
