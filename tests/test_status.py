import pandas as pd

from prefixlab.status import status_report


def test_status_counts_rows_per_design_and_cond(tmp_path):
    pd.DataFrame([{"design": "A", "cond": "none", "k": 1}, {"design": "A", "cond": "none", "k": 2},
                  {"design": "floor", "cond": "lang", "k": 0}]).to_csv(tmp_path / "sweep.csv", index=False)
    out = status_report(tmp_path)
    assert "sweep.csv: 3 rows" in out and "A none: 2" in out and "floor lang: 1" in out and "flat.csv: missing" in out
