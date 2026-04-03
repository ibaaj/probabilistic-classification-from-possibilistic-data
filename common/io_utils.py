from __future__ import annotations

"""Small shared JSON/CSV I/O helpers."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(x: Any) -> Any:
    """Convert nested Python/NumPy objects into JSON-serializable structures."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)


def load_json(path: str | Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def save_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of row dictionaries to CSV.

    The column order is the first-seen order across the input rows.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        p.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)