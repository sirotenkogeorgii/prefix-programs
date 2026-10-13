"""FLORES+ (openlanguagedata/flores_plus): one jsonl per split and language, aligned by `id`.

The dataset is gated: accept the terms on Hugging Face once and pass a token
(or be logged in). Nothing here needs the `datasets` library.
"""
from __future__ import annotations

import json
from pathlib import Path

FLORES_REPO = "openlanguagedata/flores_plus"


def read_flores_jsonl(path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["id"])] = r["text"]
    return out


def download_flores(lang_code: str, split: str, token: str | None = None, cache_dir=None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(FLORES_REPO, f"{split}/{lang_code}.jsonl", repo_type="dataset", token=token, cache_dir=cache_dir)
    )


def load_flores(lang_codes: list[str], split: str, token: str | None = None, cache_dir=None) -> dict[str, dict[int, str]]:
    return {c: read_flores_jsonl(download_flores(c, split, token, cache_dir)) for c in lang_codes}


def aligned_rows(per_lang: dict[str, dict[int, str]]) -> list[dict]:
    """Rows {id, <code>: text, ...} for ids present in every language, sorted by id."""
    codes = list(per_lang)
    common = set.intersection(*(set(d) for d in per_lang.values())) if codes else set()
    return [{"id": i, **{c: per_lang[c][i] for c in codes}} for i in sorted(common)]


def load_flores_rows(flores_dir, lang_codes: list[str], split: str, token: str | None = None, cache_dir=None) -> list[dict]:
    """Aligned rows from a local mirror (`<dir>/<split>/<code>.jsonl`) or, if `flores_dir` is None, from the Hub."""
    if flores_dir is not None:
        per = {c: read_flores_jsonl(Path(flores_dir) / split / f"{c}.jsonl") for c in lang_codes}
    else:
        per = load_flores(lang_codes, split, token, cache_dir)
    return aligned_rows(per)
