"""Online (prequential) code length of a prefix over doubling blocks, and the warm-start distance runner."""
from prefixlab.data.formats import Cond, build_example
from prefixlab.prefix import SoftPrefix
from prefixlab.prequential import doubling_blocks, prequential_bits
from prefixlab.train import TrainCfg

PAIRS = [
    ("The cat sat on the mat.", "Кошка сидела на коврике."),
    ("I like green tea.", "Я люблю зелёный чай."),
    ("Where is the station?", "Где находится вокзал?"),
    ("It is raining today.", "Сегодня идёт дождь."),
    ("She reads every evening.", "Она читает каждый вечер."),
    ("We will come tomorrow.", "Мы придём завтра."),
    ("The book is on the table.", "Книга лежит на столе."),
    ("He plays the piano well.", "Он хорошо играет на пианино."),
]


def test_doubling_blocks_cover_the_range_once():
    assert doubling_blocks(100, first=16) == [(0, 16), (16, 32), (32, 64), (64, 100)]
    assert doubling_blocks(10, first=16) == [(0, 10)]
    assert doubling_blocks(32, first=16) == [(0, 16), (16, 32)]


def test_prequential_total_is_sum_of_blocks_and_learning_cost_is_nonnegative(tok, tiny_model):
    exs = [build_example(tok, x, y, Cond()) for x, y in PAIRS]
    E = tiny_model.get_input_embeddings().weight.detach()
    prefix = SoftPrefix.from_vocab(E, k=2, seed=0)
    cfg = TrainCfg(steps=8, batch_size=4, lr=0.1, warmup=1, seed=0)
    res = prequential_bits(tiny_model, prefix, exs, tok.pad_token_id, cfg, first_block=2)
    assert len(res.block_bits) == len(doubling_blocks(len(exs), 2)) == 3
    assert abs(sum(res.block_bits) - res.total_bits) < 1e-6
    assert res.n_tokens == sum(e.n_target_tokens for e in exs) == sum(res.block_tokens)
    assert res.final_bits > 0
    assert res.learning_cost_bits == res.total_bits - res.final_bits
    assert res.learning_cost_bits > 0  # the final prefix fits the seen data better than the online code did
