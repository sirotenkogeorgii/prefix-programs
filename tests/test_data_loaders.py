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


def _mlama_fixture(root):
    _write_jsonl(root / "en" / "templates.jsonl", [
        {"relation": "P19", "template": "[X] was born in [Y] ."},
        {"relation": "P36", "template": "[Y] is the capital of [X] ."},
    ])
    _write_jsonl(root / "ru" / "templates.jsonl", [
        {"relation": "P19", "template": "[X] родился в [Y] ."},
        {"relation": "P36", "template": "[Y] столица [X] ."},
    ])
    en_p19 = [
        {"uuid": "u1", "lineid": 1, "sub_uri": "Q1", "sub_label": "Dante", "obj_uri": "Q2", "obj_label": "Florence"},
        {"uuid": "u2", "lineid": 2, "sub_uri": "Q3", "sub_label": "Einstein", "obj_uri": "Q4", "obj_label": "Ulm"},
        {"uuid": "u3", "lineid": 3, "sub_uri": "Q5", "sub_label": "Nobody", "obj_uri": "Q6", "obj_label": "Nowhere"},
    ]
    ru_p19 = [
        {"uuid": "u1", "lineid": 1, "sub_uri": "Q1", "sub_label": "Данте", "obj_uri": "Q2", "obj_label": "Флоренция"},
        {"uuid": "u2", "lineid": 2, "sub_uri": "Q3", "sub_label": "Эйнштейн", "obj_uri": "Q4", "obj_label": "Ulm"},
    ]
    _write_jsonl(root / "en" / "P19.jsonl", en_p19)
    _write_jsonl(root / "ru" / "P19.jsonl", ru_p19)
    _write_jsonl(root / "en" / "P36.jsonl", [{"uuid": "u9", "lineid": 9, "sub_uri": "Q9", "sub_label": "France", "obj_uri": "Q10", "obj_label": "Paris"}])
    _write_jsonl(root / "ru" / "P36.jsonl", [{"uuid": "u9", "lineid": 9, "sub_uri": "Q9", "sub_label": "Франция", "obj_uri": "Q10", "obj_label": "Париж"}])


def test_cross_lingual_recall_uses_english_template_and_target_language_object(tmp_path):
    _mlama_fixture(tmp_path)
    rows = cross_lingual_recall(tmp_path, obj_lang="ru")
    by_uuid = {r["uuid"]: r for r in rows}
    assert set(by_uuid) == {"u1", "u2"}  # u3 has no Russian object; P36 has [Y] not at the end
    assert by_uuid["u1"]["input"] == "Dante was born in"
    assert by_uuid["u1"]["target"] == "Флоренция"
    assert by_uuid["u1"]["relation"] == "P19"
    assert by_uuid["u1"]["copyable"] is False
    assert by_uuid["u2"]["copyable"] is True  # 'Ulm' identical in both languages


def test_cross_lingual_recall_english_objects_for_english_cell(tmp_path):
    _mlama_fixture(tmp_path)
    rows = cross_lingual_recall(tmp_path, obj_lang="en")
    assert {r["target"] for r in rows} == {"Florence", "Ulm", "Nowhere"}


def test_split_by_uuid_is_deterministic_and_disjoint():
    rows = [{"uuid": f"u{i}", "input": "x", "target": "y"} for i in range(1000)]
    tr, dv, te = split_by_uuid(rows, fractions=(0.7, 0.1, 0.2))
    assert len(tr) + len(dv) + len(te) == 1000
    assert 600 < len(tr) < 800 and 50 < len(dv) < 150
    assert not ({r["uuid"] for r in tr} & {r["uuid"] for r in te})
    tr2, _, _ = split_by_uuid(rows, fractions=(0.7, 0.1, 0.2))
    assert [r["uuid"] for r in tr] == [r["uuid"] for r in tr2]
