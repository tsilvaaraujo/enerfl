import csv

import pytest

from enerfl.csv_writer import MetricsCSVWriter, _safe_csv_path


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_rejects_non_csv_path(tmp_path):
    with pytest.raises(ValueError):
        _safe_csv_path(str(tmp_path / "out.txt"))


def test_append_rows_and_header_growth(tmp_path):
    path = tmp_path / "out.csv"
    w = MetricsCSVWriter(str(path), base_fields=["algorithm", "round"])
    w.append_row({"algorithm": "A", "round": 1, "time": 10.0})
    w.append_row({"algorithm": "A", "round": 2, "time": 20.0, "extra": 5})

    rows = _read(path)
    assert len(rows) == 2
    assert rows[0]["time"] == "10.0"
    assert "extra" in rows[0]
    assert rows[1]["extra"] == "5"


def test_rewrite_rows_replaces_content(tmp_path):
    path = tmp_path / "out.csv"
    w = MetricsCSVWriter(str(path), base_fields=["algorithm", "round"])
    w.append_row({"algorithm": "A", "round": 1})
    w.rewrite_rows([{"algorithm": "B", "round": 7, "acc": 0.9}])

    rows = _read(path)
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "B"
    assert rows[0]["acc"] == "0.9"


def test_reset_truncates_to_header(tmp_path):
    path = tmp_path / "out.csv"
    w = MetricsCSVWriter(str(path))
    w.append_row({"algorithm": "A", "round": 1})
    w.reset()
    assert _read(path) == []
