"""Shared fixtures: the real Qwen2.5 tokenizer and a tiny random-init Qwen2 model.

No model weights are downloaded; only the tokenizer files (a few MB).
"""
import pytest
import torch

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="session")
def tiny_model(tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
