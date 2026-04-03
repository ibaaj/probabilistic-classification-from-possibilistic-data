#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

THRESHOLD_FILENAMES = (
    "chaosnli_ambiguity_thresholds.json",
    "chaosnli_thresholds.json",
)


def find_threshold_files(paths: list[str]) -> list[Path]:
    found: list[Path] = []

    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name in THRESHOLD_FILENAMES:
            found.append(p.resolve())
            continue

        if p.is_dir():
            for filename in THRESHOLD_FILENAMES:
                found.extend(sorted(q.resolve() for q in p.rglob(filename)))

    dedup: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_from_path(path: Path) -> str:
    """
    Try to recover the protocol name from a path such as
    out/real_nlp/chaosnli/val_full/ambiguity/chaosnli_ambiguity_thresholds.json
    """
    parts = path.parts
    for name in ("val_full", "val_S_amb", "val_S_easy"):
        if name in parts:
            return name
    return ""


def extract_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)

    row = {
        "file": str(path),
        "protocol": protocol_from_path(path),
        "reference_split": payload.get("reference_split", ""),
        "reference_subset": payload.get("reference_subset", ""),
        "n_reference": payload.get("n_reference", ""),
        "T_low_peak": payload.get("T_low_peak", ""),
        "T_high_peak": payload.get("T_high_peak", ""),
        "T_low_H": payload.get("T_low_H", ""),
        "T_high_H": payload.get("T_high_H", ""),
    }
    return row


def format_float(x: Any) -> str:
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No threshold files found.")
        return

    headers = [
        "protocol",
        "reference_split",
        "reference_subset",
        "n_reference",
        "T_low_peak",
        "T_high_peak",
        "T_low_H",
        "T_high_H",
    ]

    widths: dict[str, int] = {}
    for h in headers:
        widths[h] = max(
            len(h),
            max(len(format_float(row[h])) for row in rows),
        )

    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))

    for row in rows:
        print(
            "  ".join(
                format_float(row[h]).ljust(widths[h])
                for h in headers
            )
        )


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "reference_split",
        "reference_subset",
        "n_reference",
        "T_low_peak",
        "T_high_peak",
        "T_low_H",
        "T_high_H",
        "file",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ChaosNLI ambiguity thresholds from saved JSON files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Files or directories to scan.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    files = find_threshold_files(list(args.inputs))
    rows = [extract_row(path) for path in files]
    rows.sort(key=lambda row: (str(row["protocol"]), str(row["file"])))

    print_table(rows)

    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
        print()
        print(f"wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
