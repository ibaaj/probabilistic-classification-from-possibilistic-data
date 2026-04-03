#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np


TRAIN_SECTIONS = ("train_full", "train_S_amb", "train_S_easy")
NUMERIC_THRESHOLD_KEYS = ("T_low_peak", "T_high_peak", "T_low_H", "T_high_H")
META_THRESHOLD_KEYS = ("reference_split", "reference_subset", "n_reference")


def _add_project_root(project_root: Path) -> None:
    root = project_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _make_loader_args(cli_args: argparse.Namespace, *, train_section: str) -> Namespace:
    return Namespace(
        dataset_url=str(cli_args.dataset_url),
        data_root=str(cli_args.data_root),
        emb_dir=str(cli_args.emb_dir),
        source_subsets=list(cli_args.source_subsets),
        train_section=str(train_section),
        split_seed=int(cli_args.split_seed),
        train_frac=float(cli_args.train_frac),
        val_frac=float(cli_args.val_frac),
        pi_eps=float(cli_args.pi_eps),
        tie_tol=float(cli_args.tie_tol),
        eps_cap=float(cli_args.eps_cap),
        eval_apply_keep_filter=bool(cli_args.eval_apply_keep_filter),
        max_train_samples=int(cli_args.max_train_samples),
        train_subset_seed=int(cli_args.train_subset_seed),
        embedding_model=str(cli_args.embedding_model),
        embedding_batch_size=int(cli_args.embedding_batch_size),
        embedding_max_length=int(cli_args.embedding_max_length),
        embedding_storage_dtype=str(cli_args.embedding_storage_dtype),
    )


def _sample_ids(samples: list[Any]) -> list[str]:
    out: list[str] = []
    for i, sample in enumerate(samples):
        sid = getattr(sample, "sample_id", None)
        if sid is None:
            raise ValueError(f"Sample at index {i} is missing sample_id.")
        out.append(str(sid))
    return out


def _float_close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= float(tol)


def _compare_thresholds(loader_thresholds: dict[str, Any], exp_thresholds: dict[str, Any], tol: float) -> list[str]:
    errors: list[str] = []

    for key in META_THRESHOLD_KEYS:
        if loader_thresholds.get(key) != exp_thresholds.get(key):
            errors.append(
                f"Threshold metadata mismatch for {key}: "
                f"loader={loader_thresholds.get(key)!r}, exp={exp_thresholds.get(key)!r}"
            )

    for key in NUMERIC_THRESHOLD_KEYS:
        lhs = float(loader_thresholds.get(key, float("nan")))
        rhs = float(exp_thresholds.get(key, float("nan")))
        if not _float_close(lhs, rhs, tol):
            errors.append(f"Threshold mismatch for {key}: loader={lhs:.12g}, exp={rhs:.12g}")

    return errors


def _format_id_diff(name: str, expected: list[str], actual: list[str], max_show: int = 10) -> str:
    expected_set = set(expected)
    actual_set = set(actual)
    only_expected = sorted(expected_set - actual_set)
    only_actual = sorted(actual_set - expected_set)
    return (
        f"{name} membership mismatch: expected_n={len(expected)}, actual_n={len(actual)}, "
        f"only_expected={only_expected[:max_show]}, only_actual={only_actual[:max_show]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the loader-side train-section logic and the experiment-side "
            "slice logic yield the same ChaosNLI results on the FULL_REPRO split."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-url", default="https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1")
    parser.add_argument("--data-root", default="data/chaosnli")
    parser.add_argument("--emb-dir", default="out/chaosnli_emb")
    parser.add_argument("--source-subsets", nargs="+", default=["snli", "mnli"])
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--pi-eps", type=float, default=1e-6)
    parser.add_argument("--tie-tol", type=float, default=0.0)
    parser.add_argument("--eps-cap", type=float, default=0.05)
    parser.add_argument("--eval-apply-keep-filter", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--train-subset-seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="roberta-base")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--embedding-storage-dtype", default="float16", choices=["float16", "float32", "float64"])
    parser.add_argument("--float-tol", type=float, default=1e-12)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON in addition to the summary.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    _add_project_root(project_root)

    from experiment.chaosnli_slices import compute_slice_stats_for_split, compute_slice_thresholds, slice_masks_from_stats
    from nlpbench.chaosnli.loader import load_chaosnli_splits, load_raw_items
    from nlpbench.chaosnli.splits import split_raw_items
    from nlpbench.chaosnli.votes import VoteDerivationConfig, build_items_for_split

    # 1) Load the FULL_REPRO train_full payload. This carries the loader-side thresholds
    # actually used by the pipeline and the train_full samples used downstream.
    base_data = load_chaosnli_splits(_make_loader_args(args, train_section="train_full"))

    if int(base_data["missing_embedding_counts"]["train_full"]) != 0:
        raise RuntimeError(
            "train_full has missing embeddings. Run the embedding build step first so the check "
            "is performed on the exact FULL_REPRO sample set."
        )

    train_full_samples = list(base_data["train_full"])
    n_classes = int(base_data["C"])
    loader_thresholds = dict(base_data["train_section_thresholds"])

    # 2) Recompute the slice thresholds and masks with the experiment-side implementation.
    exp_stats = compute_slice_stats_for_split(train_full_samples, n_classes=n_classes)
    exp_thresholds = compute_slice_thresholds(exp_stats)
    exp_masks = slice_masks_from_stats(exp_stats, exp_thresholds)

    expected_memberships = {
        "train_full": _sample_ids(train_full_samples),
        "train_S_amb": [sid for sid, keep in zip(_sample_ids(train_full_samples), exp_masks["S_amb"]) if bool(keep)],
        "train_S_easy": [sid for sid, keep in zip(_sample_ids(train_full_samples), exp_masks["S_easy"]) if bool(keep)],
    }

    # 3) Load the actual loader-selected train section for each FULL_REPRO train_section.
    actual_memberships: dict[str, list[str]] = {}
    train_full_memberships_across_calls: dict[str, list[str]] = {}
    for section in TRAIN_SECTIONS:
        section_data = load_chaosnli_splits(_make_loader_args(args, train_section=section))
        actual_memberships[section] = _sample_ids(list(section_data["train"]))
        train_full_memberships_across_calls[section] = _sample_ids(list(section_data["train_full"]))

    # 4) Rebuild train_full items without embeddings so we can compare the item-level inputs to the
    # two slicing procedures. This diagnoses whether the entropy / Hnorm inputs differ.
    raw_items = load_raw_items(
        dataset_url=str(args.dataset_url),
        data_root=str(args.data_root),
        source_subsets=list(args.source_subsets),
    )
    raw_splits = split_raw_items(
        raw_items,
        split_seed=int(args.split_seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
    )
    train_full_items, _ = build_items_for_split(
        list(raw_splits["train"]),
        config=VoteDerivationConfig(
            pi_eps=float(args.pi_eps),
            tie_tol=float(args.tie_tol),
            eps_cap=float(args.eps_cap),
        ),
        apply_keep_filter=True,
    )

    loader_item_stats: dict[str, dict[str, Any]] = {}
    log_c = math.log(float(n_classes)) if n_classes > 1 else 1.0
    for item in train_full_items:
        uid = str(item.uid)
        loader_item_stats[uid] = {
            "peak": float(np.max(np.asarray(item.vote_p, dtype=np.float64))),
            "Hnorm": float(float(item.entropy) / log_c) if n_classes > 1 else 0.0,
            "unique_top": bool(int(item.top_votes) > int(item.second_votes)),
            "top_votes": int(item.top_votes),
            "second_votes": int(item.second_votes),
            "n_raters": int(item.n_raters),
        }

    exp_item_stats: dict[str, dict[str, Any]] = {}
    for row in exp_stats:
        exp_item_stats[str(row.sample_id)] = {
            "peak": float(row.peak),
            "Hnorm": float(row.Hnorm),
            "unique_top": bool(row.unique_top),
            "top_votes": int(row.top_votes),
            "second_votes": int(row.second_votes),
            "n_raters": int(row.n_raters),
        }

    errors: list[str] = []
    warnings: list[str] = []

    # Threshold comparison.
    errors.extend(_compare_thresholds(loader_thresholds, exp_thresholds, tol=float(args.float_tol)))

    # train_full should be identical across the three load_chaosnli_splits calls.
    full_ref = train_full_memberships_across_calls["train_full"]
    for section in TRAIN_SECTIONS[1:]:
        if train_full_memberships_across_calls[section] != full_ref:
            errors.append(
                _format_id_diff(
                    f"train_full consistency across load_chaosnli_splits calls (vs {section})",
                    full_ref,
                    train_full_memberships_across_calls[section],
                )
            )

    # Membership comparison for the actual loader-selected train sections.
    for section in TRAIN_SECTIONS:
        expected = expected_memberships[section]
        actual = actual_memberships[section]
        if actual != expected:
            errors.append(_format_id_diff(section, expected, actual))

    # Item-level stats comparison.
    loader_ids = set(loader_item_stats)
    exp_ids = set(exp_item_stats)
    if loader_ids != exp_ids:
        errors.append(
            _format_id_diff(
                "train_full item ids (loader raw/items vs experiment samples)",
                sorted(loader_ids),
                sorted(exp_ids),
            )
        )
    else:
        peak_diffs: list[float] = []
        hnorm_diffs: list[float] = []
        for uid in sorted(loader_ids):
            lhs = loader_item_stats[uid]
            rhs = exp_item_stats[uid]
            for key in ("unique_top", "top_votes", "second_votes", "n_raters"):
                if lhs[key] != rhs[key]:
                    errors.append(f"Per-item mismatch for uid={uid!r}, field={key}: loader={lhs[key]!r}, exp={rhs[key]!r}")
            peak_diffs.append(abs(float(lhs["peak"]) - float(rhs["peak"])))
            hnorm_diffs.append(abs(float(lhs["Hnorm"]) - float(rhs["Hnorm"])))

        peak_max = float(max(peak_diffs)) if peak_diffs else 0.0
        hnorm_max = float(max(hnorm_diffs)) if hnorm_diffs else 0.0
        peak_bad = int(sum(diff > float(args.float_tol) for diff in peak_diffs))
        hnorm_bad = int(sum(diff > float(args.float_tol) for diff in hnorm_diffs))
        if peak_bad:
            errors.append(f"Per-item peak differs for {peak_bad} items; max_abs_diff={peak_max:.12g}")
        if hnorm_bad:
            errors.append(f"Per-item Hnorm differs for {hnorm_bad} items; max_abs_diff={hnorm_max:.12g}")
        if not peak_bad and peak_max > 0.0:
            warnings.append(f"Per-item peak matched within tolerance; max_abs_diff={peak_max:.12g}")
        if not hnorm_bad and hnorm_max > 0.0:
            warnings.append(f"Per-item Hnorm matched within tolerance; max_abs_diff={hnorm_max:.12g}")

    report = {
        "project_root": str(project_root),
        "config": {
            "dataset_url": str(args.dataset_url),
            "data_root": str(args.data_root),
            "emb_dir": str(args.emb_dir),
            "source_subsets": list(args.source_subsets),
            "split_seed": int(args.split_seed),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "pi_eps": float(args.pi_eps),
            "tie_tol": float(args.tie_tol),
            "eps_cap": float(args.eps_cap),
            "eval_apply_keep_filter": bool(args.eval_apply_keep_filter),
            "max_train_samples": int(args.max_train_samples),
            "train_subset_seed": int(args.train_subset_seed),
            "embedding_model": str(args.embedding_model),
            "embedding_batch_size": int(args.embedding_batch_size),
            "embedding_max_length": int(args.embedding_max_length),
            "embedding_storage_dtype": str(args.embedding_storage_dtype),
            "float_tol": float(args.float_tol),
        },
        "loader_thresholds": {k: loader_thresholds.get(k) for k in META_THRESHOLD_KEYS + NUMERIC_THRESHOLD_KEYS},
        "experiment_thresholds": {k: exp_thresholds.get(k) for k in META_THRESHOLD_KEYS + NUMERIC_THRESHOLD_KEYS},
        "expected_membership_sizes": {k: len(v) for k, v in expected_memberships.items()},
        "actual_membership_sizes": {k: len(v) for k, v in actual_memberships.items()},
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }

    if report["ok"]:
        print("PASS: loader-side and experiment-side ChaosNLI slice logic agree on this FULL_REPRO split.")
    else:
        print("FAIL: loader-side and experiment-side ChaosNLI slice logic disagree on this FULL_REPRO split.")
        for message in errors:
            print(f"  - {message}")

    if warnings:
        print("Notes:")
        for message in warnings:
            print(f"  - {message}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
