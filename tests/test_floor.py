"""LoRA fine-tuning floor per cell: improves the cell and leaves the shared frozen model untouched."""
from prefixlab.data.formats import Cond, build_example
from prefixlab.floor import FloorCfg, train_lora_floor
from prefixlab.train import evaluate

PAIRS = [
    ("The cat sat on the mat.", "Кошка сидела на коврике."),
    ("I like green tea.", "Я люблю зелёный чай."),
    ("Where is the station?", "Где находится вокзал?"),
    ("It is raining today.", "Сегодня идёт дождь."),
    ("She reads every evening.", "Она читает каждый вечер."),
    ("We will come tomorrow.", "Мы придём завтра."),
]


def test_lora_floor_improves_cell_and_restores_model(tok, tiny_model):
    exs = [build_example(tok, x, y, Cond()) for x, y in PAIRS]
    data = {"train": exs, "dev": exs, "test": exs}
    before = evaluate(tiny_model, None, exs, tok.pad_token_id).bits_per_token
    cfg = FloorCfg(steps=30, batch_size=6, lr=1e-2, r=4, eval_every=10, warmup=2, seed=0)
    res = train_lora_floor(tiny_model, tok, data, cfg)
    assert res.test_bits_per_token < before - 0.05
    assert res.dev_bits_per_token < before
    assert res.n_test_tokens == sum(e.n_target_tokens for e in exs)
    after = evaluate(tiny_model, None, exs, tok.pad_token_id).bits_per_token
    assert abs(after - before) < 1e-6
    assert not any(p.requires_grad for p in tiny_model.parameters())
    assert not any("lora" in n.lower() for n, _ in tiny_model.named_modules())
