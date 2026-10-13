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


def build_text(input_text: str, target_text: str, cond: Cond) -> tuple[str, int]:
    context = ""
    if cond.task_instruction:
        context += cond.task_instruction + " "
    context += input_text + "\n"
    if cond.lang_label:
        context += cond.lang_label + " "
    return context + target_text, len(context)


def build_example(tok, input_text: str, target_text: str, cond: Cond = Cond()) -> Example:
    text, start = build_text(input_text, target_text, cond)
    end = len(text)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    mask = [bool(e > start and s < end) for (s, e) in offsets]
    first = next((s for (s, _), m in zip(offsets, mask) if m), None)
    clean = first is not None and (first == start or (first == start - 1 and text[start - 1] == " "))
    return Example(
        input_ids=ids,
        target_mask=mask,
        text=text,
        target_char_start=start,
        n_target_tokens=sum(mask),
        boundary_clean=clean,
    )
