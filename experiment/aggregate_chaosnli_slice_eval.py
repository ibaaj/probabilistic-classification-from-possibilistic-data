#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

METRICS = ["acc", "nll", "brier", "ece", "mass_plausible", "top1_in_plausible", "conf_mean"]
MODES = ["A", "B", "C"]
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
    "effective_train_size",
    "train_section_size_before_subsample",
    "full_train_size_before_subsample",
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
    "slice",
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float:
    if value in (None, "", "nan", "NaN"):
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


def _iter_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            for match in glob.glob(raw, recursive=True):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() in {".csv", ".json"}:
                    paths.append(mp)
            continue
        if p.is_dir():
            paths.extend(sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in {".csv", ".json"}))
        elif p.is_file() and p.suffix.lower() in {".csv", ".json"}:
            paths.append(p)

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(rp)
    return deduped


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [dict(row) for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _infer_train_section(
    result0: Dict[str, Any],
    provenance: Dict[str, Any],
    hyperparams: Dict[str, Any],
    run_json: Path,
) -> str:
    for source in (result0, provenance, hyperparams):
        value = _safe_text(source.get("train_section", ""))
        if value:
            return value

    path_parts = set(run_json.parts)
    for candidate in TRAIN_SECTION_ORDER:
        if candidate in path_parts:
            return candidate

    return "train_full"


def _load_run_metadata(run_json_path: str, cache: Dict[Path, Dict[str, Any]]) -> Dict[str, Any]:
    path = Path(run_json_path).resolve()
    if path in cache:
        return cache[path]

    payload = json.loads(path.read_text(encoding="utf-8"))
    result0 = payload.get("results", [{}])[0] if isinstance(payload.get("results"), list) and payload.get("results") else {}
    provenance = result0.get("data_provenance", {}) if isinstance(result0, dict) else {}
    hyperparams = payload.get("hyperparams", {}) if isinstance(payload.get("hyperparams"), dict) else {}

    meta = {
        "dataset": "chaosnli",
        "selection_split": _safe_text(result0.get("selection_split", "val_full")) or "val_full",
        "selection_test_split": _safe_text(result0.get("selection_test_split", "test_full")) or "test_full",
        "source_subsets": _safe_csv_list(provenance.get("source_subsets", "")),
        "split_seed": _safe_opt_int(provenance.get("split_seed", None)),
        "train_frac": _safe_opt_float(provenance.get("train_frac", None)),
        "val_frac": _safe_opt_float(provenance.get("val_frac", None)),
        "train_section": _infer_train_section(result0, provenance, hyperparams, path),
        "requested_train_size": _safe_opt_int(result0.get("requested_train_size", provenance.get("requested_train_size", None))),
        "effective_train_size": _safe_opt_int(result0.get("effective_train_size", provenance.get("effective_train_size", None))),
        "train_section_size_before_subsample": _safe_opt_int(
            result0.get("train_section_size_before_subsample", provenance.get("train_section_size_before_subsample", None))
        ),
        "full_train_size_before_subsample": _safe_opt_int(
            result0.get("full_train_size_before_subsample", provenance.get("full_train_size_before_subsample", None))
        ),
        "train_subset_seed": _safe_opt_int(result0.get("train_subset_seed", provenance.get("train_subset_seed", None))),
        "head": _safe_text(hyperparams.get("head", "")).lower(),
        "mlp_hidden_dim": _safe_opt_int(hyperparams.get("mlp_hidden_dim", None)),
        "mlp_dropout": _safe_opt_float(hyperparams.get("mlp_dropout", None)),
        "epochs": _safe_opt_int(hyperparams.get("epochs", None)),
        "batch_size": _safe_opt_int(hyperparams.get("batch_size", None)),
        "weight_decay": _safe_opt_float(hyperparams.get("weight_decay", None)),
        "proj_tau": _safe_opt_float(hyperparams.get("proj_tau", None)),
        "proj_kmax": _safe_opt_int(hyperparams.get("proj_kmax", None)),
        "log_clip_eps": _safe_opt_float(hyperparams.get("log_clip_eps", None)),
        "proj_engine": _safe_text(hyperparams.get("proj_engine", "")),
        "proj_n_threads": _safe_opt_int(hyperparams.get("proj_n_threads", None)),
        "pi_eps": _safe_opt_float(hyperparams.get("pi_eps", None)),
        "tie_tol": _safe_opt_float(hyperparams.get("tie_tol", None)),
        "eps_cap": _safe_opt_float(hyperparams.get("eps_cap", None)),
        "active_modes": _safe_mode_list(result0.get("active_modes", hyperparams.get("active_modes", []))),
        "embedding_model": _safe_text(provenance.get("embedding_model", "")),
        "embedding_batch_size": _safe_opt_int(provenance.get("embedding_batch_size", None)),
        "embedding_max_length": _safe_opt_int(provenance.get("embedding_max_length", None)),
        "embedding_storage_dtype": _safe_text(provenance.get("embedding_storage_dtype", "")),
        "eval_apply_keep_filter": _safe_bool(provenance.get("eval_apply_keep_filter", None)),
        "lr_A": _safe_opt_float(result0.get("lr_A", hyperparams.get("lr_A", None))),
        "lr_B": _safe_opt_float(result0.get("lr_B", hyperparams.get("lr_B", None))),
        "lr_C": _safe_opt_float(result0.get("lr_C", hyperparams.get("lr_C", None))),
        "timestamp": _safe_text(payload.get("timestamp", "")),
        "run_file": path.name,
    }
    cache[path] = meta
    return meta


def _augment_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache: Dict[Path, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []

    metric_like_prefixes = tuple(
        [f"{metric}_{mode}" for metric in METRICS for mode in MODES]
        + [f"delta_{metric}_{left}_minus_{right}" for metric in METRICS for left, right in PAIR_ORDER]
    )

    for row in rows:
        dataset = _safe_text(row.get("dataset", "")).lower()
        if dataset and dataset != "chaosnli":
            continue

        enriched = dict(row)
        run_json = _safe_text(row.get("run_json", ""))
        if run_json:
            meta = _load_run_metadata(run_json, cache)
            for key, value in meta.items():
                if key not in enriched or enriched.get(key, "") in ("", None):
                    enriched[key] = value

        enriched["dataset"] = "chaosnli"
        enriched["selection_split"] = _safe_text(enriched.get("selection_split", "val_full")) or "val_full"
        enriched["selection_test_split"] = _safe_text(enriched.get("selection_test_split", "test_full")) or "test_full"
        enriched["source_subsets"] = _safe_csv_list(enriched.get("source_subsets", ""))
        enriched["train_section"] = _safe_text(enriched.get("train_section", "train_full")) or "train_full"
        enriched["slice"] = _safe_text(enriched.get("slice", ""))

        int_keys = {
            "split_seed",
            "requested_train_size",
            "effective_train_size",
            "train_section_size_before_subsample",
            "full_train_size_before_subsample",
            "train_subset_seed",
            "mlp_hidden_dim",
            "epochs",
            "batch_size",
            "proj_kmax",
            "proj_n_threads",
            "embedding_batch_size",
            "embedding_max_length",
            "n",
        }
        float_keys = {
            "train_frac",
            "val_frac",
            "mlp_dropout",
            "weight_decay",
            "proj_tau",
            "log_clip_eps",
            "pi_eps",
            "tie_tol",
            "eps_cap",
            "lr_A",
            "lr_B",
            "lr_C",
        }
        bool_keys = {"eval_apply_keep_filter"}
        text_keys = {
            "head",
            "proj_engine",
            "active_modes",
            "embedding_model",
            "embedding_storage_dtype",
        }

        for key in int_keys:
            enriched[key] = _safe_opt_int(enriched.get(key, None))
        for key in float_keys:
            enriched[key] = _safe_opt_float(enriched.get(key, None))
        for key in bool_keys:
            enriched[key] = _safe_bool(enriched.get(key, None))
        for key in text_keys:
            enriched[key] = _safe_text(enriched.get(key, ""))

        for key, value in list(enriched.items()):
            if key.startswith(metric_like_prefixes):
                enriched[key] = _safe_float(value)

        out.append(enriched)

    return out


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

    metric_cols: List[str] = ["n"]
    for mode in MODES:
        for metric in METRICS:
            metric_cols.append(f"{metric}_{mode}")
    for left, right in PAIR_ORDER:
        for metric in METRICS:
            metric_cols.append(f"delta_{metric}_{left}_minus_{right}")

    out_rows: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: _bucket_sort_key(kv[0])):
        out = {GROUP_KEYS[i]: key[i] for i in range(len(GROUP_KEYS))}
        out["n_runs"] = int(len(bucket))
        out["run_files"] = json.dumps(sorted({str(row.get("run_file", "")) for row in bucket}), sort_keys=True)

        for col in metric_cols:
            stats = _mean_std([_safe_float(row.get(col, float("nan"))) for row in bucket])
            out[f"{col}_mean"] = stats["mean"]
            out[f"{col}_std"] = stats["std"]
            out[f"{col}_min"] = stats["min"]
            out[f"{col}_max"] = stats["max"]

        out_rows.append(out)

    return out_rows


def cmd_main(args: argparse.Namespace) -> None:
    paths = _iter_paths(args.inputs)
    if not paths:
        raise FileNotFoundError("No CSV/JSON files found from --inputs.")

    raw_rows: List[Dict[str, Any]] = []
    used_paths: List[str] = []
    for path in paths:
        loaded = _load_rows(path)
        if loaded:
            raw_rows.extend(loaded)
            used_paths.append(str(path))

    rows = _augment_rows(raw_rows)
    if not rows:
        raise ValueError("No ChaosNLI slice-eval rows found among the provided inputs.")

    summary_rows = _group_rows(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out_dir / "chaosnli_slice_rows.csv", rows)
    save_rows_csv(out_dir / "chaosnli_slice_group_summary.csv", summary_rows)
    save_json(
        out_dir / "chaosnli_slice_group_summary.json",
        {
            "n_input_files_scanned": int(len(paths)),
            "n_input_files_used": int(len(used_paths)),
            "n_slice_rows": int(len(rows)),
            "n_groups": int(len(summary_rows)),
            "group_keys": GROUP_KEYS,
            "rows": summary_rows,
        },
    )

    print(
        json.dumps(
            to_jsonable(
                {
                    "n_input_files_scanned": int(len(paths)),
                    "n_input_files_used": int(len(used_paths)),
                    "n_slice_rows": int(len(rows)),
                    "n_groups": int(len(summary_rows)),
                    "out_dir": str(out_dir),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate ChaosNLI slice-evaluation CSV/JSON outputs into tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Slice-eval CSV/JSON files, directories, or glob patterns.")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd_main(args)


if __name__ == "__main__":
    main()