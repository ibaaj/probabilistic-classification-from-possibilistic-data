#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import save_json, save_rows_csv, to_jsonable
from nlpbench.chaosnli.slices import (
    compute_slice_stats_for_split,
    slice_counts,
    slice_stats_rows,
    slice_stats_summary,
)
from nlpbench.chaosnli import add_chaosnli_data_args, load_chaosnli_splits


def cmd_analyze(args: argparse.Namespace) -> None:
    data = load_chaosnli_splits(args)
    n_classes = int(data["C"])

    train_stats = compute_slice_stats_for_split(list(data["train_full"]), n_classes=n_classes)
    val_stats = compute_slice_stats_for_split(list(data["val_full"]), n_classes=n_classes)
    test_stats = compute_slice_stats_for_split(list(data["test_full"]), n_classes=n_classes)

    if "slice_thresholds" in data and isinstance(data["slice_thresholds"], dict):
        thresholds = dict(data["slice_thresholds"])
    elif "train_section_thresholds" in data and isinstance(data["train_section_thresholds"], dict):
        thresholds = dict(data["train_section_thresholds"])
    else:
        raise KeyError("ChaosNLI loader payload is missing canonical slice thresholds.")

    all_rows = (
        slice_stats_rows(train_stats, split_name="train")
        + slice_stats_rows(val_stats, split_name="val_full")
        + slice_stats_rows(test_stats, split_name="test_full")
    )

    summary = {
        "dataset": "chaosnli",
        "label_names": list(data.get("label_names", [])),
        "thresholds": thresholds,
        "counts": {
            "train": slice_counts(train_stats, thresholds),
            "val_full": slice_counts(val_stats, thresholds),
            "test_full": slice_counts(test_stats, thresholds),
        },
        "split_summaries": {
            "train": slice_stats_summary(train_stats),
            "val_full": slice_stats_summary(val_stats),
            "test_full": slice_stats_summary(test_stats),
        },
        "thresholds_ignore_train_subsample": True,
        "requested_train_size_from_cli": int(getattr(args, "max_train_samples", 0)),
        "effective_train_size_used_for_thresholds": int(len(train_stats)),
        "data_provenance": {
            key: to_jsonable(value)
            for key, value in data.items()
            if key not in {"train", "train_full", "val_full", "test_full", "C", "D"}
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out_dir / "chaosnli_ambiguity_rows.csv", all_rows)
    save_json(out_dir / "chaosnli_ambiguity_thresholds.json", thresholds)
    save_json(out_dir / "chaosnli_ambiguity_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze ChaosNLI vote ambiguity and derive fixed slice thresholds.")
    add_chaosnli_data_args(parser)
    parser.add_argument("--out-dir", required=True)
    parser.set_defaults(fn=cmd_analyze)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
