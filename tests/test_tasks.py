"""Cells, separators, conditioning strings and pair construction."""
from prefixlab.tasks import Cell, cond_for, pairs_to_examples, split_half, translation_pairs, continuation_pairs


def test_split_half_splits_on_words_and_rejects_short_sentences():
    assert split_half("one two three four five six seven") == ("one two three", "four five six seven")
    assert split_half("too short here") is None


def test_translation_pairs_use_english_source_and_target_language():
    rows = [{"id": 1, "eng_Latn": "Hello.", "rus_Cyrl": "Привет.", "deu_Latn": "Hallo."}]
    assert translation_pairs(rows, "ru") == [("Hello.", "Привет.")]
    assert translation_pairs(rows, "de") == [("Hello.", "Hallo.")]


def test_continuation_pairs_split_sentences_in_the_target_language():
    rows = [{"id": 1, "eng_Latn": "a b c d e f g h", "rus_Cyrl": "один два три четыре пять шесть", "deu_Latn": "x y"}]
    assert continuation_pairs(rows, "ru") == [("один два три", "четыре пять шесть")]
    assert continuation_pairs(rows, "de") == []  # too short, dropped


def test_cond_for_each_conditioning_mode():
    cell = Cell("translation", "ru")
    assert cond_for(cell, "none").lang_label is None and cond_for(cell, "none").task_instruction is None
    assert cond_for(cell, "lang").lang_label == "Russian:"
    assert cond_for(cell, "task").task_instruction == "Translate:"
    both = cond_for(cell, "both")
    assert both.lang_label == "Russian:" and both.task_instruction == "Translate:"


def test_pairs_to_examples_uses_task_separator(tok):
    exs_tr = pairs_to_examples(tok, [("Hello there", "Привет")], Cell("translation", "ru"), "none")
    exs_rc = pairs_to_examples(tok, [("Dante was born in", "Флоренция")], Cell("recall", "ru"), "lang")
    assert exs_tr[0].text == "Hello there\nПривет"
    assert exs_rc[0].text == "Russian: Dante was born in Флоренция"


from prefixlab.tasks import cell_examples, cell_pairs
from prefixlab.data.flores import load_flores_rows


def test_cell_pairs_translation_splits_dev_and_uses_devtest_for_test(flores_dir):
    dev = load_flores_rows(flores_dir, ["eng_Latn", "rus_Cyrl"], "dev")
    devtest = load_flores_rows(flores_dir, ["eng_Latn", "rus_Cyrl"], "devtest")
    pairs = cell_pairs(Cell("translation", "ru"), dev, devtest, mlama_root=None, dev_frac=0.25, seed=0)
    assert len(pairs["train"]) == 6 and len(pairs["dev"]) == 2 and len(pairs["test"]) == 3
    assert not (set(pairs["train"]) & set(pairs["dev"]))
    assert pairs == cell_pairs(Cell("translation", "ru"), dev, devtest, mlama_root=None, dev_frac=0.25, seed=0)
    assert all("слово" in y and "word" in x for x, y in pairs["test"])


def test_cell_pairs_recall_uses_mlama_splits(mlama_dir):
    pairs = cell_pairs(Cell("recall", "ru"), None, None, mlama_root=mlama_dir)
    n = sum(len(v) for v in pairs.values())
    assert n == 32 and all(pairs[s] for s in ("train", "dev", "test"))


def test_cell_examples_builds_every_split_with_task_format(tok, flores_dir):
    dev = load_flores_rows(flores_dir, ["eng_Latn", "rus_Cyrl"], "dev")
    devtest = load_flores_rows(flores_dir, ["eng_Latn", "rus_Cyrl"], "devtest")
    pairs = cell_pairs(Cell("continuation", "ru"), dev, devtest, mlama_root=None)
    exs = cell_examples(tok, pairs, Cell("continuation", "ru"), "task")
    assert set(exs) == {"train", "dev", "test"}
    assert exs["test"][0].text.startswith("Continue: слово")
    assert "\n" not in exs["test"][0].text


from prefixlab.cli_common import cap_split


def test_cap_split_is_a_seeded_random_subsample_not_the_head():
    items = list(range(100))
    a = cap_split(items, 10, seed=0)
    assert len(a) == 10 and set(a) <= set(items) and a != items[:10]
    assert a == cap_split(items, 10, seed=0) and a != cap_split(items, 10, seed=1)
    assert cap_split(items, None, seed=0) == items and cap_split(items, 500, seed=0) == items
