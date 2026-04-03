#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import save_json, save_rows_csv, to_jsonable

TRAIN_SECTION_ORDER = ("train_full", "train_S_amb", "train_S_easy")
MODES = ["A", "B", "C"]

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
    "hp_epochs",
    "hp_seeds",
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
    "proxy_used_A",
    "proxy_subset_size_A",
    "proxy_subset_frac_A",
    "proxy_subset_seed_A",
    "confirm_topk_A",
    "lr_grid_A",
    "lr_grid_B",
    "lr_grid_C",
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float:
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


def _best_val_acc(table: Any) -> float:
    if not isinstance(table, list):
        return float("nan")

    values: List[float] = []
    for row in table:
        if not isinstance(row, dict):
            continue
        value = _safe_float(row.get("val_acc_mean", float("nan")))
        if math.isfinite(value):
            values.append(value)

    return float(max(values)) if values else float("nan")


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


def _infer_train_section(
    result: Dict[str, Any],
    provenance: Dict[str, Any],
    hyperparams: Dict[str, Any],
    path: Path,
) -> str:
    for source in (result, provenance, hyperparams):
        value = _safe_text(source.get("train_section", ""))
        if value:
            return value

    path_parts = set(path.parts)
    for candidate in TRAIN_SECTION_ORDER:
        if candidate in path_parts:
            return candidate

    return "train_full"


def _flatten_hp_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _safe_text(payload.get("dataset", "")).lower() != "chaosnli":
        return []
    if _safe_text(payload.get("cmd", "")).lower() != "hp-search":
        return []

    hparams = payload.get("hyperparams", {}) if isinstance(payload.get("hyperparams"), dict) else {}
    seeds = payload.get("seeds", {}) if isinstance(payload.get("seeds"), dict) else {}

    rows: List[Dict[str, Any]] = []
    for idx, result in enumerate(payload.get("results", [])):
        if not isinstance(result, dict):
            continue

        provenance = result.get("data_provenance", {}) if isinstance(result.get("data_provenance"), dict) else {}
        protocol_a = result.get("protocol_A", None)
        if not isinstance(protocol_a, dict):
            protocol_a = result.get("hp_protocol_A", {})
        if not isinstance(protocol_a, dict):
            protocol_a = {}

        train_section = _infer_train_section(result, provenance, hparams, path)

        row: Dict[str, Any] = {
            "hp_json": str(path),
            "hp_file": path.name,
            "timestamp": _safe_text(payload.get("timestamp", "")),
            "result_index": int(idx),
            "dataset": "chaosnli",
            "selection_split": _safe_text(result.get("selection_split", "val_full")) or "val_full",
            "selection_test_split": _safe_text(result.get("selection_test_split", "")),
            "source_subsets": _safe_csv_list(provenance.get("source_subsets", "")),
            "split_seed": _safe_opt_int(provenance.get("split_seed", None)),
            "train_frac": _safe_opt_float(provenance.get("train_frac", None)),
            "val_frac": _safe_opt_float(provenance.get("val_frac", None)),
            "train_section": train_section,
            "requested_train_size": _safe_opt_int(result.get("requested_train_size", provenance.get("requested_train_size", None))),
            "effective_train_size": _safe_opt_int(result.get("effective_train_size", provenance.get("effective_train_size", None))),
            "train_section_size_before_subsample": _safe_opt_int(
                result.get("train_section_size_before_subsample", provenance.get("train_section_size_before_subsample", None))
            ),
            "full_train_size_before_subsample": _safe_opt_int(
                result.get("full_train_size_before_subsample", provenance.get("full_train_size_before_subsample", None))
            ),
            "train_subset_seed": _safe_opt_int(provenance.get("train_subset_seed", None)),
            "head": _safe_text(hparams.get("head", "")).lower(),
            "mlp_hidden_dim": _safe_opt_int(hparams.get("mlp_hidden_dim", None)),
            "mlp_dropout": _safe_opt_float(hparams.get("mlp_dropout", None)),
            "hp_epochs": _safe_opt_int(hparams.get("hp_epochs", None)),
            "hp_seeds": json.dumps(seeds.get("hp_seeds", []), sort_keys=True),
            "batch_size": _safe_opt_int(hparams.get("batch_size", None)),
            "weight_decay": _safe_opt_float(hparams.get("weight_decay", None)),
            "proj_tau": _safe_opt_float(hparams.get("proj_tau", None)),
            "proj_kmax": _safe_opt_int(hparams.get("proj_kmax", None)),
            "log_clip_eps": _safe_opt_float(hparams.get("log_clip_eps", None)),
            "proj_engine": _safe_text(hparams.get("proj_engine", "")),
            "proj_n_threads": _safe_opt_int(hparams.get("proj_n_threads", None)),
            "pi_eps": _safe_opt_float(hparams.get("pi_eps", None)),
            "tie_tol": _safe_opt_float(hparams.get("tie_tol", None)),
            "eps_cap": _safe_opt_float(hparams.get("eps_cap", None)),
            "active_modes": _safe_mode_list(hparams.get("active_modes", [])),
            "embedding_model": _safe_text(provenance.get("embedding_model", "")),
            "embedding_batch_size": _safe_opt_int(provenance.get("embedding_batch_size", None)),
            "embedding_max_length": _safe_opt_int(provenance.get("embedding_max_length", None)),
            "embedding_storage_dtype": _safe_text(provenance.get("embedding_storage_dtype", "")),
            "eval_apply_keep_filter": _safe_bool(provenance.get("eval_apply_keep_filter", None)),
            "best_lr_A": _safe_opt_float(result.get("best_lr_A", None)),
            "best_lr_B": _safe_opt_float(result.get("best_lr_B", None)),
            "best_lr_C": _safe_opt_float(result.get("best_lr_C", None)),
            "best_lr_A_proxy": _safe_opt_float(result.get("best_lr_A_proxy", None)),
            "proxy_used_A": _safe_bool(protocol_a.get("proxy_used", None)),
            "proxy_subset_size_A": _safe_opt_int(protocol_a.get("proxy_subset_size", None)),
            "full_train_size_A": _safe_opt_int(protocol_a.get("full_train_size", None)),
            "val_size_A": _safe_opt_int(protocol_a.get("val_size", None)),
            "proxy_subset_seed_A": _safe_opt_int(protocol_a.get("proxy_subset_seed", None)),
            "proxy_subset_frac_A": _safe_opt_float(protocol_a.get("proxy_subset_frac", None)),
            "proxy_subset_size_arg_A": _safe_opt_int(protocol_a.get("proxy_subset_size_arg", None)),
            "confirm_topk_A": _safe_opt_int(protocol_a.get("confirm_topk", None)),
            "confirm_lrs_A": json.dumps(protocol_a.get("confirm_lrs", []), sort_keys=True),
            "lr_grid_A": json.dumps(hparams.get("lr_grid_A", []), sort_keys=True),
            "lr_grid_B": json.dumps(hparams.get("lr_grid_B", []), sort_keys=True),
            "lr_grid_C": json.dumps(hparams.get("lr_grid_C", []), sort_keys=True),
            "seed_init_base_A": _safe_opt_int(seeds.get("seed_init_base_A", None)),
            "seed_init_base_B": _safe_opt_int(seeds.get("seed_init_base_B", None)),
            "seed_init_base_C": _safe_opt_int(seeds.get("seed_init_base_C", None)),
        }

        for mode in MODES:
            table = result.get(f"table_{mode}", [])
            row[f"n_lr_candidates_{mode}"] = int(len(table)) if isinstance(table, list) else 0
            row[f"top_val_acc_{mode}"] = _best_val_acc(table)

        rows.append(row)

    return rows


def _lr_counts(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        value = row.get(field, None)
        if value is not None and math.isfinite(float(value)):
            counter[str(value)] += 1
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def _group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_canonical_group_value(row.get(k, None)) for k in GROUP_KEYS)
        buckets[key].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: _bucket_sort_key(kv[0])):
        out = {GROUP_KEYS[i]: key[i] for i in range(len(GROUP_KEYS))}
        out["n_hp_searches"] = int(len(bucket))
        out["hp_files"] = json.dumps(sorted({str(row.get("hp_file", "")) for row in bucket}), sort_keys=True)

        for mode in MODES:
            counts = _lr_counts(bucket, f"best_lr_{mode}")
            out[f"best_lr_{mode}_counts_json"] = json.dumps(counts, sort_keys=True)
            out[f"best_lr_{mode}_unique_n"] = int(len(counts))
            out[f"best_lr_{mode}_mode"] = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if counts else ""

            vals = [
                float(row[f"top_val_acc_{mode}"])
                for row in bucket
                if math.isfinite(_safe_float(row.get(f"top_val_acc_{mode}", float("nan"))))
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[f"top_val_acc_{mode}_mean"] = float(np.mean(arr))
                out[f"top_val_acc_{mode}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            else:
                out[f"top_val_acc_{mode}_mean"] = float("nan")
                out[f"top_val_acc_{mode}_std"] = float("nan")

        out_rows.append(out)

    return out_rows


def cmd_main(args: argparse.Namespace) -> None:
    json_paths = _iter_json_paths(args.inputs)
    if not json_paths:
        raise FileNotFoundError("No JSON files found from --inputs.")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in json_paths:
        extracted = _flatten_hp_json(path)
        if extracted:
            rows.extend(extracted)
        else:
            skipped.append(str(path))

    if not rows:
        raise ValueError("No ChaosNLI hp-search JSONs found among the provided inputs.")

    summary_rows = _group_rows(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(out_dir / "chaosnli_hp_rows.csv", rows)
    save_rows_csv(out_dir / "chaosnli_hp_group_summary.csv", summary_rows)
    save_json(
        out_dir / "chaosnli_hp_group_summary.json",
        {
            "n_input_jsons": int(len(json_paths)),
            "n_hp_rows": int(len(rows)),
            "n_groups": int(len(summary_rows)),
            "skipped_non_hp_or_non_chaosnli": skipped,
            "group_keys": GROUP_KEYS,
            "rows": summary_rows,
        },
    )

    print(
        json.dumps(
            to_jsonable(
                {
                    "n_input_jsons": int(len(json_paths)),
                    "n_hp_rows": int(len(rows)),
                    "n_groups": int(len(summary_rows)),
                    "out_dir": str(out_dir),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate ChaosNLI hp-search JSONs into CSV/JSON tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON files, directories, or glob patterns.")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd_main(args)


if __name__ == "__main__":
    main()