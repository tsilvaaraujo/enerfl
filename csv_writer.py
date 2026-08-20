import csv
from pathlib import Path
from typing import Any

_ALLOWED_SUFFIXES = {".csv"}


def _safe_csv_path(path: str) -> Path:
    p = Path(path).resolve()
    if p.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"CSV path must end in .csv, got: {path}")
    return p


class MetricsCSVWriter:
    def __init__(self, path: str, base_fields: list[str] | None = None) -> None:
        self._path = _safe_csv_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._base_fields = list(base_fields or ["algorithm", "round"])
        self._fieldnames: list[str] = list(self._base_fields)
        self._rows: list[dict[str, Any]] = []
        self._write_header_only()

    def append_row(self, row: dict[str, Any]) -> None:
        normalized = {str(k): v for k, v in row.items()}
        self._rows.append(normalized)

        new_fields = [k for k in normalized.keys() if k not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(new_fields)
            self._rewrite_all_rows()
            return

        with self._path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            writer.writerow(normalized)

    def rewrite_rows(self, rows: list[dict[str, Any]]) -> None:
        normalized_rows = [{str(k): v for k, v in row.items()} for row in rows]
        self._rows = normalized_rows

        fields = list(self._base_fields)
        seen = set(fields)
        for row in self._rows:
            for key in row.keys():
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        self._fieldnames = fields
        self._rewrite_all_rows()

    def reset(self) -> None:
        self._fieldnames = list(self._base_fields)
        self._rows = []
        self._write_header_only()

    def _write_header_only(self) -> None:
        with self._path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

    def _rewrite_all_rows(self) -> None:
        with self._path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            if self._rows:
                writer.writerows(self._rows)
