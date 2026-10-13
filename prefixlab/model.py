"""Load a frozen causal LM and its tokenizer."""
from __future__ import annotations

import torch


def pick_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    table = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    if name in table:
        return table[name]
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_model(name: str, dtype: str = "auto", device: str | None = None, revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    dt = resolve_dtype(dtype, dev)
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dt, revision=revision)
    except TypeError:  # transformers < 5
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt, revision=revision)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
