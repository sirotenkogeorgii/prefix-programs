"""Build scored examples: `[task_instruction] input ⏎ [lang_label] target`.

The soft prefix is *not* part of the text; it is prepended as embeddings by the
scoring code. Target tokens are those whose character span overlaps the target
text, so a first target token that carries the preceding space is scored and a
token that only covers the newline or a label is not.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cond:
    """Conditioning strings. None means the prefix must supply that information."""

    lang_label: str | None = None  # inserted before the target, e.g. "Russian:"
    task_instruction: str | None = None  # inserted before the input, e.g. "Translate:"


@dataclass
class Example:
    input_ids: list[int]
    target_mask: list[bool]
    text: str
    target_char_start: int
    n_target_tokens: int
    boundary_clean: bool  # first target token starts at the target or at the single space before it
    copy_mask: list[bool]  # target tokens whose surface form occurs in the input text


def build_text(input_text: str, target_text: str, cond: Cond, sep: str = "\n") -> tuple[str, int]:
    """Newline tasks: `[instr ]input⏎[label ]target`. Same-line tasks: `[instr ][label ]input target`."""
    context = ""
    if cond.task_instruction:
        context += cond.task_instruction + " "
    if sep == "\n":
        context += input_text + "\n"
        if cond.lang_label:
            context += cond.lang_label + " "
    else:
        if cond.lang_label:
            context += cond.lang_label + " "
        context += input_text + sep
    return context + target_text, len(context)


def _is_copyable(piece: str, input_text: str) -> bool:
    piece = piece.strip()
    return len(piece) >= 2 and any(ch.isalnum() for ch in piece) and piece in input_text


def build_example(tok, input_text: str, target_text: str, cond: Cond = Cond(), sep: str = "\n") -> Example:
    text, start = build_text(input_text, target_text, cond, sep)
    end = len(text)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    mask = [bool(e > start and s < end) for (s, e) in offsets]
    copy = [m and _is_copyable(text[s:e], input_text) for (s, e), m in zip(offsets, mask)]
    first = next((s for (s, _), m in zip(offsets, mask) if m), None)
    clean = first is not None and (first == start or (first == start - 1 and text[start - 1] == " "))
    return Example(
        input_ids=ids,
        target_mask=mask,
        text=text,
        target_char_start=start,
        n_target_tokens=sum(mask),
        boundary_clean=clean,
        copy_mask=copy,
    )
