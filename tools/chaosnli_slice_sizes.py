#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ItemRec:
    split: str                  # train / train_full / validation / test
    p_max: Optional[float] = None
    h_norm: Optional[float] = None
    has_unique_majority: Optional[bool] = None
    is_amb: Optional[bool] = None
    is_easy: Optional[bool] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small table with ChaosNLI split/slice sizes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit input CSV. If omitted, the script searches under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out/real_nlp/chaosnli"),
        help="Root directory used to auto-discover the input CSV when --input is not given.",
    )
    parser.add_argument("--out-tex", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def discover_input_csv(root: Path) -> Path:
    candidates = sorted(p for p in root.rglob("*.csv") if p.is_file())

    preferred: List[Path] = []
    for p in candidates:
        name = p.name.lower()
        path_text = str(p).lower()
        if "slice" in name or "slice" in path_text or "item" in name or "item" in path_text:
            preferred.append(p)

    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No CSV file found under {root}")


def canonical_split(value: str) -> str:
    x = str(value).strip().lower()
    if x in {"tr", "train", "training"}:
        return "train"
    if x in {"train_full"}:
        return "train_full"
    if x in {"val", "validation", "dev", "val_full"}:
        return "validation"
    if x in {"te", "test", "testing", "test_full"}:
        return "test"
    raise ValueError(f"Unknown split value: {value!r}")


def safe_float(x: object) -> Optional[float]:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def safe_bool(x: object) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def normalized_entropy(probs: List[float]) -> float:
    denom = math.log(3.0)
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p)
    return total / denom


def stats_from_counts(counts: Dict[str, int]) -> tuple[float, float, bool]:
    keys = ["entailment", "neutral", "contradiction"]
    vals = [int(counts.get(k, 0)) for k in keys]
    total = sum(vals)
    if total <= 0:
        raise ValueError("All label counts are zero.")
    probs = [v / total for v in vals]
    top = max(vals)
    has_unique_majority = sum(v == top for v in vals) == 1
    return max(probs), normalized_entropy(probs), has_unique_majority


def find_count_columns(row: Dict[str, object]) -> Optional[Dict[str, str]]:
    candidates = {
        "entailment": [
            "count_entailment", "entailment_count", "entailment", "e_count", "nli_entailment"
        ],
        "neutral": [
            "count_neutral", "neutral_count", "neutral", "n_count", "nli_neutral"
        ],
        "contradiction": [
            "count_contradiction", "contradiction_count", "contradiction", "c_count", "nli_contradiction"
        ],
    }
    out: Dict[str, str] = {}
    for label, names in candidates.items():
        for name in names:
            if name in row and str(row[name]).strip() != "":
                out[label] = name
                break
    return out if len(out) == 3 else None


def first_present(row: Dict[str, object], names: List[str]) -> Optional[object]:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return None


def extract_split(row: Dict[str, object]) -> str:
    value = first_present(
        row,
        ["split", "split_key", "split_name", "partition", "subset"]
    )
    if value is None:
        raise ValueError("Missing split column.")
    return canonical_split(str(value))


def extract_membership_flag(row: Dict[str, object], kind: str) -> Optional[bool]:
    assert kind in {"amb", "easy"}

    explicit_names = {
        "amb": [
            "is_S_amb", "in_S_amb", "slice_S_amb", "is_amb", "in_amb",
            "amb_member", "is_ambiguous_slice"
        ],
        "easy": [
            "is_S_easy", "in_S_easy", "slice_S_easy", "is_easy", "in_easy",
            "easy_member"
        ],
    }

    value = first_present(row, explicit_names[kind])
    parsed = safe_bool(value)
    if parsed is not None:
        return parsed

    for key, raw_value in row.items():
        name = str(key).strip().lower()
        val = safe_bool(raw_value)
        if val is None:
            continue

        if kind == "amb":
            if "amb" in name and "easy" not in name and any(tok in name for tok in ["is", "in", "member", "slice"]):
                return val
        else:
            if "easy" in name and "amb" not in name and any(tok in name for tok in ["is", "in", "member", "slice"]):
                return val

    return None


def row_to_item(row: Dict[str, object]) -> ItemRec:
    split = extract_split(row)

    is_amb = extract_membership_flag(row, "amb")
    is_easy = extract_membership_flag(row, "easy")

    p_max = safe_float(first_present(row, ["p_max", "peak_vote_proportion", "peak_prob", "peak"]))
    h_norm = safe_float(first_present(row, ["h_norm", "normalized_entropy", "entropy_norm", "Hnorm"]))
    has_unique_majority = safe_bool(
        first_present(row, ["has_unique_majority", "unique_majority", "is_unique_majority", "unique_top"])
    )

    if p_max is not None and h_norm is not None and has_unique_majority is not None:
        return ItemRec(
            split=split,
            p_max=p_max,
            h_norm=h_norm,
            has_unique_majority=has_unique_majority,
            is_amb=is_amb,
            is_easy=is_easy,
        )

    if "label_counter" in row and str(row["label_counter"]).strip() != "":
        raw = json.loads(str(row["label_counter"]))
        counts = {str(k).strip().lower(): int(v) for k, v in raw.items()}
        p_max, h_norm, has_unique_majority = stats_from_counts(counts)
        return ItemRec(
            split=split,
            p_max=p_max,
            h_norm=h_norm,
            has_unique_majority=has_unique_majority,
            is_amb=is_amb,
            is_easy=is_easy,
        )

    count_cols = find_count_columns(row)
    if count_cols is not None:
        counts = {
            label: int(float(row[col]))
            for label, col in count_cols.items()
        }
        p_max, h_norm, has_unique_majority = stats_from_counts(counts)
        return ItemRec(
            split=split,
            p_max=p_max,
            h_norm=h_norm,
            has_unique_majority=has_unique_majority,
            is_amb=is_amb,
            is_easy=is_easy,
        )

    if is_amb is not None and is_easy is not None:
        return ItemRec(
            split=split,
            is_amb=is_amb,
            is_easy=is_easy,
        )

    raise ValueError(
        "Could not extract item statistics or direct slice membership. "
        "Expected one of: "
        "(split, p_max, h_norm, has_unique_majority), "
        "(split, label_counter), "
        "(split plus three count columns), "
        "or direct membership flags for S_amb / S_easy."
    )


def read_csv(path: Path) -> List[ItemRec]:
    rows: List[ItemRec] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")

        for line_no, raw in enumerate(reader, start=2):
            try:
                rows.append(row_to_item(dict(raw)))
            except Exception as exc:
                columns = ", ".join(reader.fieldnames)
                raise ValueError(
                    f"{path}:{line_no}: {exc}\nAvailable columns: {columns}"
                ) from exc
    return rows


def read_jsonl(path: Path) -> List[ItemRec]:
    rows: List[ItemRec] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no}: expected JSON object")
            rows.append(row_to_item(raw))
    return rows


def read_items(path: Path) -> List[ItemRec]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return read_jsonl(path)
    raise ValueError(f"Unsupported input format: {path}")


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _reference_train_items(by_split: Dict[str, List[ItemRec]]) -> List[ItemRec]:
    return list(by_split.get("train_full") or by_split.get("train") or [])


def compute_sizes(items: Iterable[ItemRec]) -> dict:
    items = list(items)

    if not items:
        raise ValueError("No items found.")

    by_split = {
        "train": [],
        "train_full": [],
        "validation": [],
        "test": [],
    }
    for x in items:
        by_split[x.split].append(x)

    train_reference = _reference_train_items(by_split)
    if not train_reference:
        raise ValueError("No training items found (expected split=train or split=train_full).")

    has_any_direct_membership = any(
        x.is_amb is not None or x.is_easy is not None for x in items
    )

    if has_any_direct_membership:
        incomplete = [
            x for x in items if x.is_amb is None or x.is_easy is None
        ]
        if incomplete:
            raise ValueError(
                "Some rows have direct slice-membership information but others do not."
            )

        return {
            "thresholds": None,
            "sizes": {
                "train_full": len(train_reference),
                "train_S_amb": sum(bool(x.is_amb) for x in train_reference),
                "train_S_easy": sum(bool(x.is_easy) for x in train_reference),
                "val_full": len(by_split["validation"]),
                "val_S_amb": sum(bool(x.is_amb) for x in by_split["validation"]),
                "val_S_easy": sum(bool(x.is_easy) for x in by_split["validation"]),
                "test_full": len(by_split["test"]),
                "test_S_amb": sum(bool(x.is_amb) for x in by_split["test"]),
                "test_S_easy": sum(bool(x.is_easy) for x in by_split["test"]),
            },
        }

    missing_raw = [
        x for x in items
        if x.p_max is None or x.h_norm is None or x.has_unique_majority is None
    ]
    if missing_raw:
        raise ValueError(
            "Input does not contain enough information to compute slice thresholds."
        )

    train_unique = [
        x for x in train_reference
        if bool(x.has_unique_majority)
    ]
    if not train_unique:
        raise ValueError("No unique-majority items found in the training split.")

    low_peak = percentile([x.p_max for x in train_unique if x.p_max is not None], 0.30)
    high_peak = percentile([x.p_max for x in train_unique if x.p_max is not None], 0.70)
    low_h = percentile([x.h_norm for x in train_unique if x.h_norm is not None], 0.30)
    high_h = percentile([x.h_norm for x in train_unique if x.h_norm is not None], 0.70)

    def is_amb(x: ItemRec) -> bool:
        return (
            bool(x.has_unique_majority)
            and x.p_max is not None
            and x.h_norm is not None
            and x.p_max <= low_peak
            and x.h_norm >= high_h
        )

    def is_easy(x: ItemRec) -> bool:
        return (
            bool(x.has_unique_majority)
            and x.p_max is not None
            and x.h_norm is not None
            and x.p_max >= high_peak
            and x.h_norm <= low_h
        )

    return {
        "thresholds": {
            "T_low_peak": low_peak,
            "T_high_peak": high_peak,
            "T_low_H": low_h,
            "T_high_H": high_h,
        },
        "sizes": {
            "train_full": len(train_reference),
            "train_S_amb": sum(is_amb(x) for x in train_reference),
            "train_S_easy": sum(is_easy(x) for x in train_reference),
            "val_full": len(by_split["validation"]),
            "val_S_amb": sum(is_amb(x) for x in by_split["validation"]),
            "val_S_easy": sum(is_easy(x) for x in by_split["validation"]),
            "test_full": len(by_split["test"]),
            "test_S_amb": sum(is_amb(x) for x in by_split["test"]),
            "test_S_easy": sum(is_easy(x) for x in by_split["test"]),
        },
    }


def latex_text(result: dict) -> str:
    s = result["sizes"]
    t = result["thresholds"]

    lines: List[str] = []
    lines.append("% Auto-generated by tools/chaosnli_slice_sizes.py")
    lines.append(rf"\newcommand{{\ChaosTrainFullN}}{{{s['train_full']}}}")
    lines.append(rf"\newcommand{{\ChaosTrainAmbN}}{{{s['train_S_amb']}}}")
    lines.append(rf"\newcommand{{\ChaosTrainEasyN}}{{{s['train_S_easy']}}}")
    lines.append(rf"\newcommand{{\ChaosValFullN}}{{{s['val_full']}}}")
    lines.append(rf"\newcommand{{\ChaosValAmbN}}{{{s['val_S_amb']}}}")
    lines.append(rf"\newcommand{{\ChaosValEasyN}}{{{s['val_S_easy']}}}")
    lines.append(rf"\newcommand{{\ChaosTestFullN}}{{{s['test_full']}}}")
    lines.append(rf"\newcommand{{\ChaosTestAmbN}}{{{s['test_S_amb']}}}")
    lines.append(rf"\newcommand{{\ChaosTestEasyN}}{{{s['test_S_easy']}}}")
    lines.append("")
    lines.append(r"% Suggested sentence for the paper:")
    lines.append(
        r"The resulting section sizes are "
        rf"\texttt{{train\_full}} ($n={s['train_full']}$), "
        rf"\texttt{{train\_S\_amb}} ($n={s['train_S_amb']}$), and "
        rf"\texttt{{train\_S\_easy}} ($n={s['train_S_easy']}$) for training; "
        rf"\texttt{{val\_full}} ($n={s['val_full']}$), "
        rf"\texttt{{val\_S\_amb}} ($n={s['val_S_amb']}$), and "
        rf"\texttt{{val\_S\_easy}} ($n={s['val_S_easy']}$) for validation; and "
        rf"\texttt{{test\_full}} ($n={s['test_full']}$), "
        rf"\texttt{{test\_S\_amb}} ($n={s['test_S_amb']}$), and "
        rf"\texttt{{test\_S\_easy}} ($n={s['test_S_easy']}$) for testing."
    )
    lines.append("")

    if t is not None:
        lines.append(r"% Thresholds:")
        lines.append(
            rf"% T_low_peak={t['T_low_peak']:.6f}, "
            rf"T_high_peak={t['T_high_peak']:.6f}, "
            rf"T_low_H={t['T_low_H']:.6f}, "
            rf"T_high_H={t['T_high_H']:.6f}"
        )
    else:
        lines.append(r"% Thresholds not emitted: input already contained direct slice membership.")

    return "\n".join(lines) + "\n"


def write_csv_summary(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = result["sizes"]
    t = result["thresholds"]

    fieldnames = ["name", "value"]
    rows = []

    if t is not None:
        rows.extend(
            [
                ("T_low_peak", t["T_low_peak"]),
                ("T_high_peak", t["T_high_peak"]),
                ("T_low_H", t["T_low_H"]),
                ("T_high_H", t["T_high_H"]),
            ]
        )

    rows.extend(
        [
            ("train_full", s["train_full"]),
            ("train_S_amb", s["train_S_amb"]),
            ("train_S_easy", s["train_S_easy"]),
            ("val_full", s["val_full"]),
            ("val_S_amb", s["val_S_amb"]),
            ("val_S_easy", s["val_S_easy"]),
            ("test_full", s["test_full"]),
            ("test_S_amb", s["test_S_amb"]),
            ("test_S_easy", s["test_S_easy"]),
        ]
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, value in rows:
            writer.writerow({"name": name, "value": value})


def main() -> int:
    args = parse_args()

    if args.input is not None:
        input_path = args.input.resolve()
    else:
        input_path = discover_input_csv(args.root.resolve())

    if not input_path.exists():
        print(f"Missing input CSV: {input_path}", file=sys.stderr, flush=True)
        return 1

    print(f"USING_INPUT={input_path}", flush=True)

    items = read_items(input_path)
    result = compute_sizes(items)

    s = result["sizes"]
    t = result["thresholds"]

    if t is not None:
        print(
            "thresholds:",
            f"T_low_peak={t['T_low_peak']:.6f}",
            f"T_high_peak={t['T_high_peak']:.6f}",
            f"T_low_H={t['T_low_H']:.6f}",
            f"T_high_H={t['T_high_H']:.6f}",
            flush=True,
        )
    else:
        print("thresholds: unavailable (direct slice-membership input)", flush=True)

    print(
        "sizes:",
        f"train_full={s['train_full']}",
        f"train_S_amb={s['train_S_amb']}",
        f"train_S_easy={s['train_S_easy']}",
        f"val_full={s['val_full']}",
        f"val_S_amb={s['val_S_amb']}",
        f"val_S_easy={s['val_S_easy']}",
        f"test_full={s['test_full']}",
        f"test_S_amb={s['test_S_amb']}",
        f"test_S_easy={s['test_S_easy']}",
        flush=True,
    )

    if args.out_tex is not None:
        out_tex = args.out_tex.resolve()
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(latex_text(result), encoding="utf-8")
        print(f"WROTE_TEX={out_tex}", flush=True)

    if args.out_csv is not None:
        out_csv = args.out_csv.resolve()
        write_csv_summary(out_csv, result)
        print(f"WROTE_CSV={out_csv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
