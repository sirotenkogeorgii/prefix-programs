"""Window patching: replace the attention input at the prefix positions in one block with a donor's."""
import torch

from prefixlab.data.formats import Cond, build_example
from prefixlab.patching import window_patch_bits, window_patch_curve
from prefixlab.scoring import collate, target_bits
from prefixlab.prefix import SoftPrefix

PAIRS = [("The cat sat on the mat.", "Кошка сидела на коврике."), ("I like green tea.", "Я люблю зелёный чай.")]


def test_identity_patch_recovers_exactly_the_recipient_bits(tok, tiny_model):
    exs = [build_example(tok, x, y, Cond()) for x, y in PAIRS]
    batch = collate(exs, tok.pad_token_id)
    E = tiny_model.get_input_embeddings().weight.detach()
    p = SoftPrefix.from_vocab(E, 3, 0).weight.detach()
    base = target_bits(tiny_model, batch, p)
    for layer in range(tiny_model.config.num_hidden_layers):
        patched = window_patch_bits(tiny_model, batch, recipient_prefix=p, donor_prefix=p, layer=layer)
        assert torch.allclose(patched, base, atol=1e-5)


def test_donor_patch_changes_bits_and_curve_has_one_entry_per_layer(tok, tiny_model):
    exs = [build_example(tok, x, y, Cond()) for x, y in PAIRS]
    batch = collate(exs, tok.pad_token_id)
    E = tiny_model.get_input_embeddings().weight.detach()
    a = SoftPrefix.from_vocab(E, 3, 0).weight.detach()
    b = SoftPrefix.from_vocab(E, 3, 7).weight.detach()
    base = target_bits(tiny_model, batch, a)
    patched = window_patch_bits(tiny_model, batch, recipient_prefix=a, donor_prefix=b, layer=0)
    assert not torch.allclose(patched, base, atol=1e-4)
    curve = window_patch_curve(tiny_model, exs, tok.pad_token_id, recipient_prefix=a, donor_prefix=b, batch_size=2)
    L = tiny_model.config.num_hidden_layers
    assert sorted(curve["recovered_bits_per_token"]) == list(range(L))
    assert curve["recipient_bits_per_token"] > 0 and curve["donor_bits_per_token"] > 0
    # patching every layer at once must reproduce the donor run exactly
    assert abs(curve["all_layers_bits_per_token"] - curve["donor_bits_per_token"]) < 1e-5
