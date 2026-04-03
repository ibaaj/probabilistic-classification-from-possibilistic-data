#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import save_json, save_rows_csv, to_jsonable

SPLITS = [
    "train",
    "val_full",
    "val_S_amb",
    "val_S_easy",
    "test_full",
    "test_S_amb",
    "test_S_easy",
]
MODES = ["A", "B", "C"]
METRICS = ["acc", "nll", "brier", "ece", "mass_plausible", "top1_in_plausible"]
PAIR_ORDER = [("A", "B"), ("A", "C"), ("B", "C")]
TRAIN_SECTION_ORDER = ("train_full", "train_S_amb", "train_S_easy")

GROUP_KEYS = [
    "dataset",
    "selection_split",
    "selection_test_split",
    "source_subsets",
    "split_seed",
    "train_frac",
    "val_frac",
    "train_section",
    "requested_train_size",
    "train_section_size_before_subsample",
    "full_train_size_before_subsample",
    "effective_train_size",
    "train_subset_seed",
    "head",
    "mlp_hidden_dim",
    "mlp_dropout",
    "epochs",
    "batch_size",
    "weight_decay",
    "proj_tau",
    "proj_kmax",
    "log_clip_eps",
    "proj_engine",
    "proj_n_threads",
    "pi_eps",
    "tie_tol",
    "eps_cap",
    "active_modes",
    "embedding_model",
    "embedding_batch_size",
    "embedding_max_length",
    "embedding_storage_dtype",
    "eval_apply_keep_filter",
    "lr_A",
    "lr_B",
    "lr_C",
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _safe_opt_float(value: Any) -> Any:
    x = _safe_float(value)
    return None if not math.isfinite(x) else float(x)


def _safe_opt_int(value: Any) -> Any:
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _safe_csv_list(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v).strip() for v in value if str(v).strip())
    return _safe_text(value)


def _safe_mode_list(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v).strip().upper() for v in value if str(v).strip())
    return _safe_text(value).upper()


def _canonical_group_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return None if not math.isfinite(x) else x
    text = str(value).strip()
    return None if text == "" else text


def _sort_atom(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, np.integer)):
        return (2, int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return (4, "nan")
        return (3, x)
    return (5, str(value))


def _bucket_sort_key(key: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
    return tuple(_sort_atom(value) for value in key)


def _iter_json_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            for match in glob.glob(raw, recursive=True):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() == ".json":
                    paths.append(mp)
            continue
        if p.is_dir():
            paths.extend(sorted(q for q in p.rglob("*.json") if q.is_file()))
        elif p.is_file() and p.suffix.lower() == ".json":
            paths.append(p)

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(rp)
    return deduped


def _get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _infer_train_section(
    result: Dict[str, Any],
    provenance: Dict[str, Any],
    hyperparams: Dict[str, Any],
    run_json: Path,
) -> str:
    for source in (result, provenance, hyperparams):
        value = _safe_text(source.get("train_section", ""))
        if value:
            return value

    path_parts = set(run_json.parts)
    for candidate in TRAIN_SECTION_ORDER:
        if candidate in path_parts:
            return candidate

    return "train_full"


def _flatten_run_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _safe_text(payload.get("dataset", "")).lower() != "chaosnli":
        return []
    if _safe_text(payload.get("cmd", "")).lower() != "run":
        return []

    top_hparams = payload.get("hyperparams", {}) if isinstance(payload.get("hyperparams"), dict) else {}
    top_seeds = payload.get("seeds", {}) if isinstance(payload.get("seeds"), dict) else {}

    rows: List[Dict[str, Any]] = []
    for idx, result in enumerate(payload.get("results", [])):
        if not isinstance(result, dict):
            continue
        provenance = result.get("data_provenance", {}) if isinstance(result.get("data_provenance"), dict) else {}
        train_section = _infer_train_section(result, provenance, top_hparams, path)

        row: Dict[str, Any] = {
            "run_json": str(path),
            "run_file": path.name,
            "timestamp": _safe_text(payload.get("timestamp", "")),
            "result_index": int(idx),
            "dataset": "chaosnli",
            "selection_split": _safe_text(result.get("selection_split", "val_full")) or "val_full",
            "selection_test_split": _safe_text(result.get("selection_test_split", "test_full")) or "test_full",
            "active_modes": _safe_mode_list(result.get("active_modes", [])),
            "source_subsets": _safe_csv_list(provenance.get("source_subsets", "")),
            "split_seed": _safe_opt_int(provenance.get("split_seed", None)),
            "train_frac": _safe_opt_float(provenance.get("train_frac", None)),
            "val_frac": _safe_opt_float(provenance.get("val_frac", None)),
            "emb_dir": _safe_text(provenance.get("emb_dir", "")),
            "train_section": train_section,
            "requested_train_size": _safe_opt_int(result.get("requested_train_size", provenance.get("requested_train_size", None))),
            "train_section_size_before_subsample": _safe_opt_int(
                result.get("train_section_size_before_subsample", provenance.get("train_section_size_before_subsample", None))
            ),
            "full_train_size_before_subsample": _safe_opt_int(
                result.get("full_train_size_before_subsample", provenance.get("full_train_size_before_subsample", None))
            ),
            "effective_train_size": _safe_opt_int(result.get("effective_train_size", provenance.get("effective_train_size", None))),
            "train_subset_seed": _safe_opt_int(result.get("train_subset_seed", provenance.get("train_subset_seed", None))),
            "head": _safe_text(top_hparams.get("head", "")).lower(),
            "mlp_hidden_dim": _safe_opt_int(top_hparams.get("mlp_hidden_dim", None)),
            "mlp_dropout": _safe_opt_float(top_hparams.get("mlp_dropout", None)),
            "epochs": _safe_opt_int(top_hparams.get("epochs", None)),
            "batch_size": _safe_opt_int(top_hparams.get("batch_size", None)),
            "weight_decay": _safe_opt_float(top_hparams.get("weight_decay", None)),
            "proj_tau": _safe_opt_float(top_hparams.get("proj_tau", None)),
            "proj_kmax": _safe_opt_int(top_hparams.get("proj_kmax", None)),
            "log_clip_eps": _safe_opt_float(top_hparams.get("log_clip_eps", None)),
            "proj_engine": _safe_text(top_hparams.get("proj_engine", "")),
            "proj_n_threads": _safe_opt_int(top_hparams.get("proj_n_threads", None)),
            "pi_eps": _safe_opt_float(top_hparams.get("pi_eps", None)),
            "tie_tol": _safe_opt_float(top_hparams.get("tie_tol", None)),
            "eps_cap": _safe_opt_float(top_hparams.get("eps_cap", None)),
            "embedding_model": _safe_text(provenance.get("embedding_model", "")),
            "embedding_batch_size": _safe_opt_int(provenance.get("embedding_batch_size", None)),
            "embedding_max_length": _safe_opt_int(provenance.get("embedding_max_length", None)),
            "embedding_storage_dtype": _safe_text(provenance.get("embedding_storage_dtype", "")),
            "eval_apply_keep_filter": _safe_bool(provenance.get("eval_apply_keep_filter", None)),
            "lr_A": _safe_opt_float(result.get("lr_A", top_hparams.get("lr_A", None))),
            "lr_B": _safe_opt_float(result.get("lr_B", top_hparams.get("lr_B", None))),
            "lr_C": _safe_opt_float(result.get("lr_C", top_hparams.get("lr_C", None))),
            "seed_init_A": _safe_opt_int(result.get("seed_A", top_seeds.get("seed_init_A", None))),
            "seed_init_B": _safe_opt_int(result.get("seed_B", top_seeds.get("seed_init_B", None))),
            "seed_init_C": _safe_opt_int(result.get("seed_C", top_seeds.get("seed_init_C", None))),
            "raw_count_train": _safe_opt_int(_get_nested(provenance, "raw_counts", "train")),
            "raw_count_validation": _safe_opt_int(_get_nested(provenance, "raw_counts", "validation")),
            "raw_count_test": _safe_opt_int(_get_nested(provenance, "raw_counts", "test")),
            "kept_count_train": _safe_opt_int(_get_nested(provenance, "kept_counts", "train")),
            "kept_count_validation": _safe_opt_int(_get_nested(provenance, "kept_counts", "validation")),
            "kept_count_test": _safe_opt_int(_get_nested(provenance, "kept_counts", "test")),
        }

        for split in SPLITS:
            split_metrics = result.get(split, {}) if isinstance(result.get(split), dict) else {}
            for mode in MODES:
                mode_metrics = split_metrics.get(mode, {}) if isinstance(split_metrics.get(mode), dict) else {}
                for metric in METRICS:
                    row[f"{split}_{metric}_{mode}"] = _safe_float(mode_metrics.get(metric, None))

        for split in SPLITS:
            for metric in METRICS:
                for left, right in PAIR_ORDER:
                    left_val = _safe_float(row.get(f"{split}_{metric}_{left}", float("nan")))
                    right_val = _safe_float(row.get(f"{split}_{metric}_{right}", float("nan")))
                    row[f"{split}_{metric}_{left}_minus_{right}"] = (
                        float(left_val - right_val)
                        if math.isfinite(left_val) and math.isfinite(right_val)
                        else float("nan")
                    )

        rows.append(row)

    return rows


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_canonical_group_value(row.get(k, None)) for k in GROUP_KEYS)
        buckets[key].append(row)

    metrics_to_summarize: List[str] = []
    for split in SPLITS:
        for mode in MODES:
            for metric in METRICS:
                metrics_to_summarize.append(f"{split}_{metric}_{mode}")
        for metric in METRICS:
            for left, right in PAIR_ORDER:
                metrics_to_summarize.append(f"{split}_{metric}_{left}_minus_{right}")

    summary_rows: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: _bucket_sort_key(kv[0])):
        out = {GROUP_KEYS[i]: key[i] for i in range(len(GROUP_KEYS))}
        out["n_runs"] = int(len(bucket))
        out["run_files"] = json.dumps(sorted({str(row.get("run_file", "")) for row in bucket}), sort_keys=True)

        for col in metrics_to_summarize:
            stats = _mean_std([_safe_float(row.get(col, float("nan"))) for row in bucket])
            out[f"{col}_mean"] = stats["mean"]
            out[f"{col}_std"] = stats["std"]
            out[f"{col}_min"] = stats["min"]
            out[f"{col}_max"] = stats["max"]

        summary_rows.append(out)

    return summary_rows


def cmd_main(args: argparse.Namespace) -> None:
    json_paths = _iter_json_paths(args.inputs)
    if not json_paths:
        raise FileNotFoundError("No JSON files found from --inputs.")

    run_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in json_paths:
        rows = _flatten_run_json(path)
        if rows:
            run_rows.extend(rows)
        else:
            skipped.append(str(path))

    if not run_rows:
        raise ValueError("No ChaosNLI run JSONs found among the provided inputs.")

    summary_rows = _group_rows(run_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out_dir / "chaosnli_run_rows.csv", run_rows)
    save_rows_csv(out_dir / "chaosnli_run_group_summary.csv", summary_rows)
    save_json(
        out_dir / "chaosnli_run_group_summary.json",
        {
            "n_input_jsons": int(len(json_paths)),
            "n_run_rows": int(len(run_rows)),
            "n_groups": int(len(summary_rows)),
            "skipped_non_run_or_non_chaosnli": skipped,
            "group_keys": GROUP_KEYS,
            "rows": summary_rows,
        },
    )

    print(
        json.dumps(
            to_jsonable(
                {
                    "n_input_jsons": int(len(json_paths)),
                    "n_run_rows": int(len(run_rows)),
                    "n_groups": int(len(summary_rows)),
                    "out_dir": str(out_dir),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate saved ChaosNLI run JSONs into CSV/JSON tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON files, directories, or glob patterns.")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd_main(args)


if __name__ == "__main__":
    main()