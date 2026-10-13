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


def _write_jsonl(path, rows):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture(scope="session")
def mlama_dir(tmp_path_factory):
    """Synthetic mLAMA layout: en/ru/de, P19 ([Y]-final) and P36 ([Y] not final)."""
    root = tmp_path_factory.mktemp("mlama")
    _write_jsonl(root / "en" / "templates.jsonl", [
        {"relation": "P19", "template": "[X] was born in [Y] ."},
        {"relation": "P36", "template": "[Y] is the capital of [X] ."},
    ])
    for lang, born in [("ru", "родился в"), ("de", "wurde geboren in")]:
        _write_jsonl(root / lang / "templates.jsonl", [
            {"relation": "P19", "template": f"[X] {born} [Y] ."},
            {"relation": "P36", "template": "[Y] ... [X] ."},
        ])
    en = [
        {"uuid": "u1", "lineid": 1, "sub_uri": "Q1", "sub_label": "Dante", "obj_uri": "Q2", "obj_label": "Florence"},
        {"uuid": "u2", "lineid": 2, "sub_uri": "Q3", "sub_label": "Einstein", "obj_uri": "Q4", "obj_label": "Ulm"},
        {"uuid": "u3", "lineid": 3, "sub_uri": "Q5", "sub_label": "Nobody", "obj_uri": "Q6", "obj_label": "Nowhere"},
    ]
    ru = [
        {"uuid": "u1", "lineid": 1, "sub_uri": "Q1", "sub_label": "Данте", "obj_uri": "Q2", "obj_label": "Флоренция"},
        {"uuid": "u2", "lineid": 2, "sub_uri": "Q3", "sub_label": "Эйнштейн", "obj_uri": "Q4", "obj_label": "Ulm"},
    ]
    de = [
        {"uuid": "u1", "lineid": 1, "sub_uri": "Q1", "sub_label": "Dante", "obj_uri": "Q2", "obj_label": "Florenz"},
        {"uuid": "u2", "lineid": 2, "sub_uri": "Q3", "sub_label": "Einstein", "obj_uri": "Q4", "obj_label": "Ulm"},
    ]
    for i in range(4, 34):  # 30 generic triples present in all languages
        en.append({"uuid": f"u{i}", "lineid": i, "sub_uri": f"S{i}", "sub_label": f"Person {i}", "obj_uri": f"O{i}", "obj_label": f"Town {i}"})
        ru.append({"uuid": f"u{i}", "lineid": i, "sub_uri": f"S{i}", "sub_label": f"Персона {i}", "obj_uri": f"O{i}", "obj_label": f"Город {i}"})
        de.append({"uuid": f"u{i}", "lineid": i, "sub_uri": f"S{i}", "sub_label": f"Person {i}", "obj_uri": f"O{i}", "obj_label": f"Stadt {i}"})
    _write_jsonl(root / "en" / "P19.jsonl", en)
    _write_jsonl(root / "ru" / "P19.jsonl", ru)
    _write_jsonl(root / "de" / "P19.jsonl", de)
    for lang in ("en", "ru", "de"):
        _write_jsonl(root / lang / "P36.jsonl", [{"uuid": "u99", "lineid": 99, "sub_uri": "Q9", "sub_label": "France", "obj_uri": "Q10", "obj_label": "Paris"}])
    return root


@pytest.fixture(scope="session")
def flores_dir(tmp_path_factory):
    """Synthetic FLORES+ layout: dev (8 rows) and devtest (3 rows) for eng_Latn, rus_Cyrl, deu_Latn."""
    root = tmp_path_factory.mktemp("flores")
    def rows(code, split, ids):
        word = {"eng_Latn": "word", "rus_Cyrl": "слово", "deu_Latn": "Wort"}[code]
        return [{"id": i, "iso_639_3": code[:3], "iso_15924": code[4:], "text": " ".join(f"{word}{i}_{j}" for j in range(8)) + "."} for i in ids]
    for code in ("eng_Latn", "rus_Cyrl", "deu_Latn"):
        _write_jsonl(root / "dev" / f"{code}.jsonl", rows(code, "dev", range(1, 9)))
        _write_jsonl(root / "devtest" / f"{code}.jsonl", rows(code, "devtest", range(1, 4)))
    return root
