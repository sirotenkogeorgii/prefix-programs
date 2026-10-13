"""Cells of the task x language grid, their text format, and conditioning strings."""
from __future__ import annotations

from dataclasses import dataclass

from .data.formats import Cond, Example, build_example

LANGS = {
    "en": {"name": "English", "flores": "eng_Latn", "mlama": "en"},
    "ru": {"name": "Russian", "flores": "rus_Cyrl", "mlama": "ru"},
    "de": {"name": "German", "flores": "deu_Latn", "mlama": "de"},
    "zh": {"name": "Chinese", "flores": "cmn_Hans", "mlama": "zh"},
}
TASKS = {
    "translation": {"sep": "\n", "instruction": "Translate:"},
    "recall": {"sep": " ", "instruction": "Complete:"},
    "continuation": {"sep": " ", "instruction": "Continue:"},
}
COND_MODES = ("none", "lang", "task", "both")
MIN_WORDS_FOR_SPLIT = 6


@dataclass(frozen=True)
class Cell:
    task: str
    lang: str

    def key(self) -> str:
        return f"{self.task}-{self.lang}"


def lang_label(lang: str) -> str:
    return f"{LANGS[lang]['name']}:"


def cond_for(cell: Cell, mode: str) -> Cond:
    assert mode in COND_MODES, mode
    return Cond(
        lang_label=lang_label(cell.lang) if mode in ("lang", "both") else None,
        task_instruction=TASKS[cell.task]["instruction"] if mode in ("task", "both") else None,
    )


def split_half(text: str):
    words = text.split()
    if len(words) < MIN_WORDS_FOR_SPLIT:
        return None
    m = len(words) // 2
    return " ".join(words[:m]), " ".join(words[m:])


def translation_pairs(rows: list[dict], lang: str) -> list[tuple[str, str]]:
    src, tgt = LANGS["en"]["flores"], LANGS[lang]["flores"]
    return [(r[src], r[tgt]) for r in rows]


def continuation_pairs(rows: list[dict], lang: str) -> list[tuple[str, str]]:
    code = LANGS[lang]["flores"]
    out = []
    for r in rows:
        halves = split_half(r[code])
        if halves is not None:
            out.append(halves)
    return out


def pairs_to_examples(tok, pairs, cell: Cell, mode: str) -> list[Example]:
    cond, sep = cond_for(cell, mode), TASKS[cell.task]["sep"]
    return [build_example(tok, x, y, cond, sep=sep) for x, y in pairs]
