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
