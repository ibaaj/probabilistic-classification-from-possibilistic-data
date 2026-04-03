#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.chaosnli_slices import (
    compute_slice_stats_for_split,
    compute_slice_thresholds,
    slice_masks_from_stats,
)
from nlpbench.chaosnli.loader import load_chaosnli_splits


def _to_row(sample: Any, split_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "sample_id": str(sample.sample_id),
        "subset": str(sample.sample_id).split("::", 1)[0] if "::" in str(sample.sample_id) else "",
        "y": int(sample.y),
        "votes_entailment": int(sample.votes[0]),
        "votes_neutral": int(sample.votes[1]),
        "votes_contradiction": int(sample.votes[2]),
        "n_raters": int(sample.n_raters),
        "top_votes": int(sample.top_votes),
        "second_votes": int(sample.second_votes),
        "top_margin": int(sample.top_margin),
        "majority_label_index": int(sample.y),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the exact ChaosNLI train/validation/test protocol and ambiguity slices."
    )
    parser.add_argument("--dataset-url", default="https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1")
    parser.add_argument("--data-root", default="data/chaosnli")
    parser.add_argument("--emb-dir", default="out/chaosnli_emb")
    parser.add_argument("--source-subsets", nargs="+", default=["snli", "mnli"])
    parser.add_argument("--train-section", default="train_full")
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--pi-eps", type=float, default=1e-6)
    parser.add_argument("--tie-tol", type=float, default=0.0)
    parser.add_argument("--eps-cap", type=float, default=0.05)
    parser.add_argument("--embedding-model", default="roberta-base")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--embedding-storage-dtype", default="float16")
    parser.add_argument("--eval-apply-keep-filter", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--train-subset-seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_chaosnli_splits(SimpleNamespace(**vars(args)))

    train_used = list(data["train"])
    train_full = list(data["train_full"])
    val_full = list(data["val_full"])
    test_full = list(data["test_full"])

    train_used_ids = {str(sample.sample_id) for sample in train_used}
    train_full_ids = {str(sample.sample_id) for sample in train_full}
    train_differs = train_used_ids != train_full_ids

    split_rows = []
    split_rows.extend(_to_row(sample, "train") for sample in train_used)
    if train_differs:
        split_rows.extend(_to_row(sample, "train_full") for sample in train_full)
    split_rows.extend(_to_row(sample, "validation") for sample in val_full)
    split_rows.extend(_to_row(sample, "test") for sample in test_full)

    _write_csv(out_dir / "chaosnli_split_manifest.csv", split_rows)
    _write_jsonl(out_dir / "chaosnli_split_manifest.jsonl", split_rows)

    n_classes = int(data["C"])
    train_reference_stats = compute_slice_stats_for_split(train_full, n_classes=n_classes)
    train_used_stats = compute_slice_stats_for_split(train_used, n_classes=n_classes)
    val_stats = compute_slice_stats_for_split(val_full, n_classes=n_classes)
    test_stats = compute_slice_stats_for_split(test_full, n_classes=n_classes)

    thresholds = compute_slice_thresholds(train_reference_stats)

    (out_dir / "chaosnli_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    def slice_rows(stats_list: list[Any], split_name: str) -> list[dict[str, Any]]:
        masks = slice_masks_from_stats(stats_list, thresholds)
        rows: list[dict[str, Any]] = []
        for stat, is_amb, is_easy in zip(
            stats_list,
            masks["S_amb"].tolist(),
            masks["S_easy"].tolist(),
        ):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": stat.sample_id,
                    "subset": stat.subset,
                    "y": int(stat.y),
                    "n_raters": int(stat.n_raters),
                    "support": int(stat.support),
                    "top_votes": int(stat.top_votes),
                    "second_votes": int(stat.second_votes),
                    "top_margin": int(stat.top_margin),
                    "margin_rate": float(stat.margin_rate),
                    "peak": float(stat.peak),
                    "entropy": float(stat.entropy),
                    "Hnorm": float(stat.Hnorm),
                    "unique_top": int(bool(stat.unique_top)),
                    "in_S_amb": int(bool(is_amb)),
                    "in_S_easy": int(bool(is_easy)),
                }
            )
        return rows

    all_slice_rows = []
    all_slice_rows.extend(slice_rows(train_used_stats, "train"))
    if train_differs:
        all_slice_rows.extend(slice_rows(train_reference_stats, "train_full"))
    all_slice_rows.extend(slice_rows(val_stats, "validation"))
    all_slice_rows.extend(slice_rows(test_stats, "test"))

    _write_csv(out_dir / "chaosnli_slice_membership.csv", all_slice_rows)
    _write_jsonl(out_dir / "chaosnli_slice_membership.jsonl", all_slice_rows)

    metadata = {
        "source_subsets": list(data["source_subsets"]),
        "split_seed": int(data["split_seed"]),
        "train_frac": float(data["train_frac"]),
        "val_frac": float(data["val_frac"]),
        "label_names": list(data["label_names"]),
        "train_section": str(data.get("train_section", getattr(args, "train_section", "train_full"))),
        "train_section_sizes": data.get("train_section_sizes", {}),
        "train_section_thresholds": data.get("train_section_thresholds", {}),
        "threshold_reference": {
            "split": "train_full",
            "n_reference": int(thresholds.get("n_reference", 0)),
            "reference_subset": str(thresholds.get("reference_subset", "")),
        },
        "raw_counts": data["raw_counts"],
        "kept_counts": data["kept_counts"],
        "requested_train_size": int(data["requested_train_size"]),
        "effective_train_size": int(data["effective_train_size"]),
        "full_train_size_before_subsample": int(data["full_train_size_before_subsample"]),
        "train_section_size_before_subsample": int(data.get("train_section_size_before_subsample", len(train_used))),
        "train_subset_seed": int(data["train_subset_seed"]),
        "emb_dir": str(data["emb_dir"]),
    }
    (out_dir / "chaosnli_protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote split manifest to: {out_dir / 'chaosnli_split_manifest.csv'}")
    print(f"Wrote thresholds to:     {out_dir / 'chaosnli_thresholds.json'}")
    print(f"Wrote slice membership:  {out_dir / 'chaosnli_slice_membership.csv'}")
    print(f"Wrote metadata to:       {out_dir / 'chaosnli_protocol_metadata.json'}")


if __name__ == "__main__":
    main()
