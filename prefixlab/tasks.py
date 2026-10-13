"""Cells of the task x language grid, their text format, and conditioning strings."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .data.formats import Cond, Example, build_example
from .data.mlama import cross_lingual_recall, find_mlama_root, split_by_uuid

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
    """Pairs are (input, target) or (input, target, copy_all); copy_all marks every target token copyable."""
    cond, sep = cond_for(cell, mode), TASKS[cell.task]["sep"]
    out = []
    for pr in pairs:
        ex = build_example(tok, pr[0], pr[1], cond, sep=sep)
        if len(pr) > 2 and pr[2]:
            ex.copy_mask = list(ex.target_mask)
        out.append(ex)
    return out


def shuffle_split(items: list, dev_frac: float, seed: int) -> tuple[list, list]:
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_dev = int(round(len(items) * dev_frac))
    dev_idx = set(idx[:n_dev])
    return [items[i] for i in idx if i not in dev_idx], [items[i] for i in idx if i in dev_idx]


def cell_pairs(cell: Cell, flores_dev_rows, flores_devtest_rows, mlama_root, dev_frac: float = 0.2, seed: int = 0) -> dict:
    """{train, dev, test} pairs. FLORES tasks: dev split -> train/dev, devtest -> test. Recall: hash split of mLAMA."""
    if cell.task == "recall":
        rows = cross_lingual_recall(find_mlama_root(mlama_root), LANGS[cell.lang]["mlama"])
        parts = split_by_uuid(rows, seed=seed)
        return {s: [(r["input"], r["target"], r["copyable"]) for r in part] for s, part in zip(("train", "dev", "test"), parts)}
    fn = translation_pairs if cell.task == "translation" else continuation_pairs
    train, dev = shuffle_split(fn(flores_dev_rows, cell.lang), dev_frac, seed)
    return {"train": train, "dev": dev, "test": fn(flores_devtest_rows, cell.lang)}


def cell_examples(tok, pairs_by_split: dict, cell: Cell, mode: str) -> dict[str, list[Example]]:
    return {s: pairs_to_examples(tok, prs, cell, mode) for s, prs in pairs_by_split.items()}
