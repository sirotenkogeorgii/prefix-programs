"""FLORES+ jsonl reading/alignment and the mLAMA directory loader, on synthetic fixtures."""
import json

from prefixlab.data.flores import aligned_rows, read_flores_jsonl
from prefixlab.data.mlama import cross_lingual_recall, split_by_uuid


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_flores_read_and_align_drops_ids_missing_in_any_language(tmp_path):
    en = [{"id": i, "iso_639_3": "eng", "iso_15924": "Latn", "text": f"en {i}"} for i in [3, 1, 2, 4]]
    ru = [{"id": i, "iso_639_3": "rus", "iso_15924": "Cyrl", "text": f"ru {i}"} for i in [1, 2, 4]]
    _write_jsonl(tmp_path / "dev" / "eng_Latn.jsonl", en)
    _write_jsonl(tmp_path / "dev" / "rus_Cyrl.jsonl", ru)
    d_en = read_flores_jsonl(tmp_path / "dev" / "eng_Latn.jsonl")
    d_ru = read_flores_jsonl(tmp_path / "dev" / "rus_Cyrl.jsonl")
    assert d_en[3] == "en 3"
    rows = aligned_rows({"eng_Latn": d_en, "rus_Cyrl": d_ru})
    assert [r["id"] for r in rows] == [1, 2, 4]
    assert rows[0]["rus_Cyrl"] == "ru 1"


def test_cross_lingual_recall_uses_english_template_and_target_language_object(mlama_dir):
    rows = cross_lingual_recall(mlama_dir, obj_lang="ru")
    by_uuid = {r["uuid"]: r for r in rows}
    assert "u3" not in by_uuid  # no Russian object
    assert "u99" not in by_uuid  # P36 has [Y] not at the end
    assert len(by_uuid) == 32
    assert by_uuid["u1"]["input"] == "Dante was born in"
    assert by_uuid["u1"]["target"] == "Флоренция"
    assert by_uuid["u1"]["relation"] == "P19"
    assert by_uuid["u1"]["copyable"] is False
    assert by_uuid["u2"]["copyable"] is True  # 'Ulm' identical in both languages


def test_cross_lingual_recall_english_objects_for_english_cell(mlama_dir):
    rows = cross_lingual_recall(mlama_dir, obj_lang="en")
    targets = {r["target"] for r in rows}
    assert {"Florence", "Ulm", "Nowhere"} <= targets and len(rows) == 33
    assert not any(r["copyable"] for r in rows)


def test_split_by_uuid_is_deterministic_and_disjoint():
    rows = [{"uuid": f"u{i}", "input": "x", "target": "y"} for i in range(1000)]
    tr, dv, te = split_by_uuid(rows, fractions=(0.7, 0.1, 0.2))
    assert len(tr) + len(dv) + len(te) == 1000
    assert 600 < len(tr) < 800 and 50 < len(dv) < 150
    assert not ({r["uuid"] for r in tr} & {r["uuid"] for r in te})
    tr2, _, _ = split_by_uuid(rows, fractions=(0.7, 0.1, 0.2))
    assert [r["uuid"] for r in tr] == [r["uuid"] for r in tr2]
