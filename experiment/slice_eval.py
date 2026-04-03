#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import save_json, save_rows_csv, to_jsonable
from nlpbench.chaosnli.slices import compute_slice_stats_for_split, slice_masks_from_stats
from experiment.metrics import brier_score as _brier_score_ref
from experiment.metrics import ece_score as _ece_score_ref
from nlpbench.chaosnli import add_chaosnli_data_args, load_chaosnli_splits

OVERRIDABLE_ARG_NAMES = {
    "alpha",
    "pi_eps",
    "tie_tol",
    "eps_cap",
    "max_train_samples",
    "train_subset_seed",
    "train_section",
    "selection_split",
    "embedding_model",
    "embedding_batch_size",
    "embedding_max_length",
    "embedding_storage_dtype",
    "dataset_url",
    "data_root",
    "emb_dir",
    "source_subsets",
    "split_seed",
    "train_frac",
    "val_frac",
    "eval_apply_keep_filter",
}

METRICS = [
    "acc",
    "nll",
    "brier",
    "ece",
    "mass_plausible",
    "top1_in_plausible",
    "conf_mean",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifacts(run_json: Path) -> Dict[str, Any]:
    stem = run_json.with_suffix("")
    out: Dict[str, Any] = {
        "ids": np.load(str(stem) + "_test_full_ids.npy", allow_pickle=True),
        "y": np.load(str(stem) + "_test_full_y.npy", allow_pickle=False),
    }
    for mode in ("A", "B", "C"):
        path = Path(str(stem) + f"_test_full_probs_{mode}.npy")
        out[f"probs_{mode}"] = np.load(path, allow_pickle=False) if path.exists() else None
    return out


def _apply_run_config(args: argparse.Namespace, run_payload: Dict[str, Any]) -> None:
    top_hparams = run_payload.get("hyperparams", {})
    result0 = run_payload.get("results", [{}])[0] if run_payload.get("results") else {}
    data_provenance = result0.get("data_provenance", {})

    merged: Dict[str, Any] = {}
    if isinstance(top_hparams, dict):
        merged.update(top_hparams)
    if isinstance(data_provenance, dict):
        merged.update(data_provenance)

    for key, value in merged.items():
        if key in OVERRIDABLE_ARG_NAMES and hasattr(args, key):
            setattr(args, key, value)


def _align_probs(sample_ids: List[str], artifact_ids: Iterable[Any], probs: np.ndarray | None) -> np.ndarray | None:
    if probs is None:
        return None

    probs = np.asarray(probs, dtype=np.float64)
    artifact_ids_list = [str(x) for x in list(artifact_ids)]
    index = {sid: i for i, sid in enumerate(artifact_ids_list)}

    rows: List[np.ndarray] = []
    for sid in sample_ids:
        if sid not in index:
            raise KeyError(f"Missing saved probabilities for sample_id={sid!r}")
        rows.append(np.asarray(probs[index[sid]], dtype=np.float64))
    return np.stack(rows, axis=0)


def _metrics_for_probs(samples: List[Any], probs: np.ndarray) -> Dict[str, float]:
    if len(samples) == 0:
        raise ValueError("Cannot evaluate an empty slice.")

    y = np.asarray([int(sample.y) for sample in samples], dtype=np.int64)
    plausible = np.stack([np.asarray(sample.plausible_mask, dtype=np.float64) for sample in samples], axis=0)
    probs = np.asarray(probs, dtype=np.float64)

    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    nll = -np.log(np.maximum(probs[np.arange(y.shape[0]), y], 1e-15))

    return {
        "acc": float(np.mean(pred == y)),
        "nll": float(np.mean(nll)),
        "brier": float(_brier_score_ref(probs, y)),
        "ece": float(_ece_score_ref(probs, y)),
        "mass_plausible": float(np.mean(np.sum(probs * plausible, axis=1))),
        "top1_in_plausible": float(np.mean(plausible[np.arange(y.shape[0]), pred])),
        "conf_mean": float(np.mean(conf)),
    }


def _delta_pairs(modes_present: List[str]) -> List[Tuple[str, str]]:
    ordered = [mode for mode in ["A", "B", "C"] if mode in modes_present]
    return [(left, right) for i, left in enumerate(ordered) for right in ordered[i + 1 :]]


def cmd_eval(args: argparse.Namespace) -> None:
    run_json_path = Path(args.run_json)
    thresholds_path = Path(args.thresholds_json)
    run_payload = _load_json(run_json_path)

    if str(run_payload.get("dataset", "")).lower() != "chaosnli":
        raise ValueError(f"Unsupported dataset in run JSON: {run_payload.get('dataset')!r}")
    if str(run_payload.get("cmd", "")).lower() != "run":
        raise ValueError(f"Expected cmd='run', got {run_payload.get('cmd')!r}")

    _apply_run_config(args, run_payload)
    data = load_chaosnli_splits(args)
    test_samples = list(data["test_full"])
    sample_ids = [str(getattr(sample, "sample_id", i)) for i, sample in enumerate(test_samples)]

    thresholds = _load_json(thresholds_path)
    artifacts = _load_artifacts(run_json_path)
    artifact_ids = [str(x) for x in np.asarray(artifacts["ids"], dtype=object).tolist()]

    if sorted(artifact_ids) != sorted(sample_ids):
        missing_from_artifacts = sorted(set(sample_ids) - set(artifact_ids))
        missing_from_reload = sorted(set(artifact_ids) - set(sample_ids))
        raise ValueError(
            "Saved run artifacts do not match the reloaded test split. "
            f"missing_from_artifacts={missing_from_artifacts[:5]} missing_from_reload={missing_from_reload[:5]}"
        )

    y_saved = np.asarray(artifacts["y"], dtype=np.int64)
    y_true_by_artifact = {sid: int(y_saved[i]) for i, sid in enumerate(artifact_ids)}
    y_reloaded = np.asarray([int(sample.y) for sample in test_samples], dtype=np.int64)
    y_saved_aligned = np.asarray([y_true_by_artifact[sid] for sid in sample_ids], dtype=np.int64)
    if y_saved_aligned.shape != y_reloaded.shape or np.any(y_saved_aligned != y_reloaded):
        raise ValueError("Saved y labels do not match the reloaded test split.")

    probs_by_mode: Dict[str, np.ndarray | None] = {}
    modes_present: List[str] = []
    for mode in ("A", "B", "C"):
        aligned = _align_probs(sample_ids, artifact_ids, artifacts.get(f"probs_{mode}"))
        probs_by_mode[mode] = aligned
        if aligned is not None:
            modes_present.append(mode)

    test_stats = compute_slice_stats_for_split(test_samples, n_classes=int(data["C"]))
    masks = slice_masks_from_stats(test_stats, thresholds)

    rows: List[Dict[str, Any]] = []
    result0 = run_payload.get("results", [{}])[0]
    provenance = result0.get("data_provenance", {}) if isinstance(result0, dict) else {}
    hyperparams = run_payload.get("hyperparams", {}) if isinstance(run_payload.get("hyperparams"), dict) else {}

    slice_order = [
        ("S_full", masks["S_full"]),
        ("S_amb", masks["S_amb"]),
        ("S_easy", masks["S_easy"]),
        ("anchor_clean", masks["anchor_clean"]),
        ("low_peak_high_H", masks["low_peak_high_H"]),
        ("low_peak_low_H", masks["low_peak_low_H"]),
        ("high_peak_high_H", masks["high_peak_high_H"]),
        ("high_peak_low_H", masks["high_peak_low_H"]),
    ]

    for slice_name, mask in slice_order:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        subset = [test_samples[int(i)] for i in idx]
        row: Dict[str, Any] = {
            "dataset": "chaosnli",
            "run_json": str(run_json_path),
            "run_file": run_json_path.name,
            "slice": slice_name,
            "n": int(idx.size),
            "selection_split": str(result0.get("selection_split", "val_full")),
            "head": str(hyperparams.get("head", "")),
            "train_section": str(provenance.get("train_section", getattr(args, "train_section", "train_full"))),
        }
        for key in ("source_subsets", "split_seed", "train_frac", "val_frac"):
            if key in provenance:
                value = provenance.get(key, "")
                row[key] = ",".join(str(v) for v in value) if isinstance(value, list) else value

        for mode in modes_present:
            metrics = _metrics_for_probs(subset, probs_by_mode[mode][idx])
            for key, value in metrics.items():
                row[f"{key}_{mode}"] = float(value)

        for left, right in _delta_pairs(modes_present):
            for metric in METRICS:
                left_key = f"{metric}_{left}"
                right_key = f"{metric}_{right}"
                if left_key in row and right_key in row:
                    row[f"delta_{metric}_{left}_minus_{right}"] = float(row[left_key] - row[right_key])

        rows.append(row)

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".json":
        save_json(out_path, rows)
    else:
        save_rows_csv(out_path, rows)
    print(f"wrote: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc slice evaluation for saved ChaosNLI run probabilities.")
    add_chaosnli_data_args(parser)
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--out", required=True)
    parser.set_defaults(fn=cmd_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
