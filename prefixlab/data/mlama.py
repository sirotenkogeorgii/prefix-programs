"""mLAMA (Kassner et al. 2021) read directly from the released zip: `<lang>/templates.jsonl`,
`<lang>/<relation>.jsonl`. Cross-lingual recall = English template and subject, object in language l.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MLAMA_URL = "http://cistern.cis.lmu.de/mlama/mlama1.1.zip"


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_mlama_root(root) -> Path:
    """Directory that directly contains `en/templates.jsonl` (the zip nests one level)."""
    root = Path(root)
    if (root / "en" / "templates.jsonl").exists():
        return root
    for cand in root.rglob("en/templates.jsonl"):
        return cand.parent.parent
    raise FileNotFoundError(f"no mLAMA data under {root}")


def ensure_mlama(root) -> Path:
    root = Path(root)
    try:
        return find_mlama_root(root)
    except FileNotFoundError:
        root.mkdir(parents=True, exist_ok=True)
        zpath = root / "mlama1.1.zip"
        if not zpath.exists():
            urlretrieve(MLAMA_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
        return find_mlama_root(root)


def read_templates(root, lang: str) -> dict[str, str]:
    return {r["relation"]: r["template"] for r in _read_jsonl(Path(root) / lang / "templates.jsonl")}


def read_triples(root, lang: str, relation: str) -> list[dict]:
    p = Path(root) / lang / f"{relation}.jsonl"
    return _read_jsonl(p) if p.exists() else []


def template_ends_with_object(template: str) -> bool:
    t = template.strip()
    if t.endswith("."):
        t = t[:-1].rstrip()
    return t.endswith("[Y]")


def cross_lingual_recall(root, obj_lang: str, relations: list[str] | None = None) -> list[dict]:
    """Rows {uuid, relation, input, target, copyable}: English cloze prefix -> object label in obj_lang."""
    root = Path(root)
    en_templates = read_templates(root, "en")
    rows: list[dict] = []
    for rel in relations or sorted(en_templates):
        tpl = en_templates.get(rel)
        if not tpl or not template_ends_with_object(tpl):
            continue
        en_rows = {r["uuid"]: r for r in read_triples(root, "en", rel)}
        obj_rows = en_rows if obj_lang == "en" else {r["uuid"]: r for r in read_triples(root, obj_lang, rel)}
        for uuid, en in en_rows.items():
            obj = obj_rows.get(uuid)
            if obj is None:
                continue
            prompt = tpl.replace("[X]", en["sub_label"]).split("[Y]")[0].rstrip()
            rows.append(
                {
                    "uuid": uuid,
                    "relation": rel,
                    "input": prompt,
                    "target": obj["obj_label"],
                    "copyable": bool(obj_lang != "en" and obj["obj_label"] == en["obj_label"]),
                }
            )
    return rows


def split_by_uuid(rows: list[dict], fractions=(0.7, 0.1, 0.2), seed: int = 0):
    """Deterministic disjoint train/dev/test split keyed on a hash of the uuid."""
    a, b, _ = fractions
    out = ([], [], [])
    for r in rows:
        h = hashlib.sha1(f"{seed}:{r['uuid']}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        out[0 if u < a else 1 if u < a + b else 2].append(r)
    return out
