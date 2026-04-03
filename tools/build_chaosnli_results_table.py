#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MODES: Tuple[str, ...] = ("A", "B", "C")
DEFAULT_SECTION_ORDER: Tuple[Tuple[str, str], ...] = (
    ("test_full", "Full test set"),
    ("test_S_amb", r"Ambiguous subset $\\mathcal S_{\\mathrm{amb}}$"),
    ("test_S_easy", r"Easy subset $\\mathcal S_{\\mathrm{easy}}$"),
)
DEFAULT_TRAIN_SECTION_ORDER: Tuple[str, ...] = (
    "train_full",
    "train_S_amb",
    "train_S_easy",
)

CONFIG_FIELDS: Tuple[str, ...] = (
    "head",
    "variant",
    "source_subsets",
    "split_seed",
    "train_frac",
    "val_frac",
    "selection_split",
    "selection_test_split",
    "train_section",
    "requested_train_size",
    "effective_train_size",
    "train_section_size_before_subsample",
    "full_train_size_before_subsample",
    "train_subset_seed",
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
    "mlp_hidden_dim",
    "mlp_dropout",
    "lr_A",
    "lr_B",
    "lr_C",
)


@dataclass(frozen=True)
class RunRecord:
    run_json: Path
    config: Dict[str, Any]
    section_rows: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class SectionSummary:
    config: Dict[str, Any]
    config_label: str
    section_key: str
    section_title: str
    n_runs: int
    n_items: Optional[int]
    means: Dict[str, float]
    stds: Dict[str, float]
    selected_lr_A: str
    selected_lr_B: str
    selected_lr_C: str


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


def _safe_opt_float(value: Any) -> Optional[float]:
    x = _safe_float(value)
    return None if not math.isfinite(x) else float(x)


def _safe_opt_int(value: Any) -> Optional[int]:
    try:
        x = int(float(value))
    except Exception:
        return None
    return int(x)


def _safe_bool(value: Any) -> Optional[bool]:
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


def _format_lr_value(value: Any) -> str:
    x = _safe_opt_float(value)
    if x is None:
        return "--"
    text = f"{x:.3g}"
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e")
        return f"{mantissa}e{int(exponent)}"
    return text


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _iter_run_jsons(run_root: Path) -> List[Path]:
    return sorted(p for p in run_root.rglob("*.json") if p.is_file())


def _infer_variant(path: Path) -> str:
    text = str(path).lower()
    if "support" in text:
        return "support"
    if "plain" in text:
        return "plain"
    return ""


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
    for candidate in DEFAULT_TRAIN_SECTION_ORDER:
        if candidate in path_parts:
            return candidate

    return "train_full"


def _section_block_from_run(result0: Dict[str, Any], section_key: str) -> Dict[str, Any]:
    block = result0.get(section_key, {})
    return block if isinstance(block, dict) else {}


def _extract_sections_from_run(result0: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sizes = result0.get("sizes", {}) if isinstance(result0.get("sizes"), dict) else {}
    section_rows: Dict[str, Dict[str, Any]] = {}

    for section_key, _ in DEFAULT_SECTION_ORDER:
        block = _section_block_from_run(result0, section_key)
        parsed: Dict[str, Any] = {"n": _safe_opt_int(sizes.get(section_key, None))}
        for mode in MODES:
            mode_block = block.get(mode, {})
            value = _safe_float(mode_block.get("acc", float("nan"))) if isinstance(mode_block, dict) else float("nan")
            if math.isfinite(value):
                parsed[mode] = value
        if any(mode in parsed for mode in MODES):
            section_rows[section_key] = parsed

    return section_rows


def _extract_sections_from_slice_csv(slice_csv: Path, sizes: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    section_rows: Dict[str, Dict[str, Any]] = {}
    if not slice_csv.exists():
        return section_rows

    name_map = {
        "S_full": "test_full",
        "S_amb": "test_S_amb",
        "S_easy": "test_S_easy",
    }

    for row in _read_csv_rows(slice_csv):
        slice_name = _safe_text(row.get("slice", ""))
        section_key = name_map.get(slice_name, "")
        if not section_key:
            continue

        parsed: Dict[str, Any] = {"n": _safe_opt_int(row.get("n", sizes.get(section_key, None)))}
        for mode in MODES:
            value = _safe_float(row.get(f"acc_{mode}", float("nan")))
            if math.isfinite(value):
                parsed[mode] = value
        section_rows[section_key] = parsed

    return section_rows


def _find_slice_eval_csv(run_json: Path, run_root: Path, slice_root: Path) -> Optional[Path]:
    relative = run_json.relative_to(run_root)
    candidates = [
        slice_root / relative.parent / f"{run_json.stem}.slice_eval.csv",
        slice_root / f"{run_json.stem}.slice_eval.csv",
    ]

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def _extract_run_record(
    run_json: Path,
    run_root: Path,
    slice_root: Optional[Path],
) -> Optional[RunRecord]:
    payload = _json_load(run_json)
    if _safe_text(payload.get("dataset", "")).lower() != "chaosnli":
        return None
    if _safe_text(payload.get("cmd", "")).lower() != "run":
        return None

    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        return None
    result0 = results[0]
    if not isinstance(result0, dict):
        return None

    provenance = result0.get("data_provenance", {}) if isinstance(result0.get("data_provenance"), dict) else {}
    hyperparams = payload.get("hyperparams", {}) if isinstance(payload.get("hyperparams"), dict) else {}

    section_rows = _extract_sections_from_run(result0)

    if slice_root is not None:
        fallback_csv = _find_slice_eval_csv(run_json, run_root, slice_root)
        if fallback_csv is not None:
            sizes = result0.get("sizes", {}) if isinstance(result0.get("sizes"), dict) else {}
            fallback_rows = _extract_sections_from_slice_csv(fallback_csv, sizes=sizes)
            for section_key, parsed in fallback_rows.items():
                section_rows.setdefault(section_key, parsed)

    if not section_rows:
        return None

    config: Dict[str, Any] = {
        "head": _safe_text(hyperparams.get("head", "")).lower() or "unknown",
        "variant": _infer_variant(run_json),
        "source_subsets": _safe_csv_list(provenance.get("source_subsets", "")),
        "split_seed": _safe_opt_int(provenance.get("split_seed", "")),
        "train_frac": _safe_opt_float(provenance.get("train_frac", "")),
        "val_frac": _safe_opt_float(provenance.get("val_frac", "")),
        "selection_split": _safe_text(result0.get("selection_split", "val_full")) or "val_full",
        "selection_test_split": _safe_text(result0.get("selection_test_split", "test_full")) or "test_full",
        "train_section": _infer_train_section(result0, provenance, hyperparams, run_json),
        "requested_train_size": _safe_opt_int(result0.get("requested_train_size", provenance.get("requested_train_size", None))),
        "effective_train_size": _safe_opt_int(result0.get("effective_train_size", provenance.get("effective_train_size", None))),
        "train_section_size_before_subsample": _safe_opt_int(
            result0.get("train_section_size_before_subsample", provenance.get("train_section_size_before_subsample", None))
        ),
        "full_train_size_before_subsample": _safe_opt_int(
            result0.get("full_train_size_before_subsample", provenance.get("full_train_size_before_subsample", None))
        ),
        "train_subset_seed": _safe_opt_int(result0.get("train_subset_seed", provenance.get("train_subset_seed", None))),
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
        "mlp_hidden_dim": _safe_opt_int(hyperparams.get("mlp_hidden_dim", None)),
        "mlp_dropout": _safe_opt_float(hyperparams.get("mlp_dropout", None)),
        "lr_A": _safe_opt_float(result0.get("lr_A", hyperparams.get("lr_A", None))),
        "lr_B": _safe_opt_float(result0.get("lr_B", hyperparams.get("lr_B", None))),
        "lr_C": _safe_opt_float(result0.get("lr_C", hyperparams.get("lr_C", None))),
    }

    return RunRecord(
        run_json=run_json,
        config=config,
        section_rows=section_rows,
    )


def _build_config_labels(
    keys: Sequence[Tuple[Any, ...]],
    *,
    keep_train_size: bool,
    keep_train_section: bool,
) -> Dict[Tuple[Any, ...], str]:
    records = [{field: key[i] for i, field in enumerate(CONFIG_FIELDS)} for key in keys]
    subset_values = {cfg.get("source_subsets", "") for cfg in records if cfg.get("source_subsets", "")}
    split_seed_values = {cfg.get("split_seed", None) for cfg in records if cfg.get("split_seed", None) is not None}
    selection_values = {cfg.get("selection_split", "") for cfg in records if cfg.get("selection_split", "")}
    train_section_values = {cfg.get("train_section", "") for cfg in records if cfg.get("train_section", "")}
    train_size_values = {cfg.get("effective_train_size", None) for cfg in records if cfg.get("effective_train_size", None) is not None}

    labels: Dict[Tuple[Any, ...], str] = {}
    for key in keys:
        cfg = {field: key[i] for i, field in enumerate(CONFIG_FIELDS)}
        parts: List[str] = []

        parts.append(_safe_text(cfg.get("head", "")) or "unknown")

        variant = _safe_text(cfg.get("variant", ""))
        if variant:
            parts.append(variant)

        if len(train_section_values) > 1 or keep_train_section:
            parts.append(_safe_text(cfg.get("train_section", "")) or "train_full")

        if len(selection_values) > 1 and cfg.get("selection_split", ""):
            parts.append(_safe_text(cfg.get("selection_split", "")))

        if len(subset_values) > 1 and cfg.get("source_subsets", ""):
            parts.append(_safe_text(cfg.get("source_subsets", "")).replace(",", "+"))

        if len(split_seed_values) > 1 and cfg.get("split_seed", None) is not None:
            parts.append(f"seed{int(cfg['split_seed'])}")

        if keep_train_size or len(train_size_values) > 1:
            if cfg.get("effective_train_size", None) is not None:
                parts.append(f"N={int(cfg['effective_train_size'])}")

        labels[key] = "|".join(parts)

    return labels
    

def _resolve_train_size(records: Sequence[RunRecord], requested_train_size: Optional[int]) -> Optional[int]:
    sizes = sorted({int(r.config["effective_train_size"]) for r in records if r.config.get("effective_train_size") is not None})
    if not sizes:
        return None
    if requested_train_size is None:
        return sizes[-1]
    if requested_train_size not in sizes:
        raise ValueError(f"Requested --train-size={requested_train_size} is not available. Found sizes: {sizes}")
    return int(requested_train_size)


def _filter_records(
    records: Sequence[RunRecord],
    *,
    train_size: Optional[int],
    heads: Optional[Sequence[str]],
    source_subsets: Optional[Sequence[str]],
    selection_splits: Optional[Sequence[str]],
    train_sections: Optional[Sequence[str]],
) -> List[RunRecord]:
    out: List[RunRecord] = []
    head_set = {str(x).strip().lower() for x in heads} if heads else None
    subset_set = {str(x).strip() for x in source_subsets} if source_subsets else None
    selection_set = {str(x).strip() for x in selection_splits} if selection_splits else None
    train_section_set = {str(x).strip() for x in train_sections} if train_sections else None

    for record in records:
        cfg = record.config
        if train_size is not None and cfg.get("effective_train_size") != train_size:
            continue
        if head_set is not None and cfg.get("head", "") not in head_set:
            continue
        if subset_set is not None and cfg.get("source_subsets", "") not in subset_set:
            continue
        if selection_set is not None and cfg.get("selection_split", "") not in selection_set:
            continue
        if train_section_set is not None and cfg.get("train_section", "") not in train_section_set:
            continue
        out.append(record)
    return out


def _group_key(record: RunRecord) -> Tuple[Any, ...]:
    return tuple(record.config.get(field, None) for field in CONFIG_FIELDS)


def _train_section_sort_key(name: str) -> Tuple[int, str]:
    if name in DEFAULT_TRAIN_SECTION_ORDER:
        return (DEFAULT_TRAIN_SECTION_ORDER.index(name), name)
    return (len(DEFAULT_TRAIN_SECTION_ORDER), name)


def _bucket_sort_key(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    cfg = {field: key[i] for i, field in enumerate(CONFIG_FIELDS)}
    return (
        cfg.get("selection_split", ""),
        _train_section_sort_key(_safe_text(cfg.get("train_section", "")))[0],
        cfg.get("head", ""),
        cfg.get("variant", ""),
        cfg.get("source_subsets", ""),
        -1 if cfg.get("effective_train_size") is None else int(cfg["effective_train_size"]),
        _safe_text(cfg.get("lr_A", "")),
        _safe_text(cfg.get("lr_B", "")),
        _safe_text(cfg.get("lr_C", "")),
    )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _mode_value_from_record(record: RunRecord, section_key: str, mode: str) -> float:
    section = record.section_rows.get(section_key, {})
    if not isinstance(section, dict):
        return float("nan")
    return _safe_float(section.get(mode, float("nan")))


def _section_n_from_record(record: RunRecord, section_key: str) -> Optional[int]:
    section = record.section_rows.get(section_key, {})
    if not isinstance(section, dict):
        return None
    return _safe_opt_int(section.get("n", None))



def _aggregate_sections(
    records: Sequence[RunRecord],
    *,
    keep_train_size: bool,
    keep_train_section: bool,
    section_order: Sequence[Tuple[str, str]],
) -> List[SectionSummary]:
    buckets: Dict[Tuple[Any, ...], List[RunRecord]] = defaultdict(list)
    for record in records:
        buckets[_group_key(record)].append(record)

    ordered_keys = sorted(buckets.keys(), key=_bucket_sort_key)
    labels = _build_config_labels(
        ordered_keys,
        keep_train_size=keep_train_size,
        keep_train_section=keep_train_section,
    )

    summaries: List[SectionSummary] = []
    for key in ordered_keys:
        cfg = {field: key[i] for i, field in enumerate(CONFIG_FIELDS)}
        bucket = buckets[key]

        for section_key, section_title in section_order:
            means: Dict[str, float] = {}
            stds: Dict[str, float] = {}
            present_any = False

            for mode in MODES:
                values = [_mode_value_from_record(record, section_key, mode) for record in bucket]
                mean, std = _mean_std(values)
                means[mode] = mean
                stds[mode] = std
                if math.isfinite(mean):
                    present_any = True

            if not present_any:
                continue

            n_candidates = [_section_n_from_record(record, section_key) for record in bucket]
            n_candidates = [n for n in n_candidates if n is not None]
            n_items = Counter(n_candidates).most_common(1)[0][0] if n_candidates else None

            summaries.append(
                SectionSummary(
                    config=cfg,
                    config_label=labels[key],
                    section_key=section_key,
                    section_title=section_title,
                    n_runs=len(bucket),
                    n_items=n_items,
                    means=means,
                    stds=stds,
                    selected_lr_A=_format_lr_value(cfg.get("lr_A", None)),
                    selected_lr_B=_format_lr_value(cfg.get("lr_B", None)),
                    selected_lr_C=_format_lr_value(cfg.get("lr_C", None)),
                )
            )

    return summaries


def _fmt_acc(value: float) -> str:
    return "--" if not math.isfinite(value) else f"{value:.3f}"


def _fmt_delta(value: float) -> str:
    return "--" if not math.isfinite(value) else f"{value:+.3f}"


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash ")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _best_modes(summary: SectionSummary) -> List[str]:
    present = [mode for mode in MODES if math.isfinite(summary.means.get(mode, float("nan")))]
    if not present:
        return []
    best = max(summary.means[mode] for mode in present)
    return [mode for mode in present if abs(summary.means[mode] - best) <= 1e-12]


def _build_latex_table(
    summaries: Sequence[SectionSummary],
    *,
    caption: str,
    label: str,
    delta_base: str,
    compare_modes: Sequence[str],
) -> str:
    delta_base = str(delta_base).upper()
    compare_modes = [str(mode).upper() for mode in compare_modes if str(mode).upper() in MODES and str(mode).upper() != delta_base]
    if delta_base not in MODES:
        raise ValueError("delta_base must be one of A, B, C.")
    if not compare_modes:
        raise ValueError("compare_modes must contain at least one mode different from delta_base.")

    section_map: Dict[str, List[SectionSummary]] = defaultdict(list)
    for summary in summaries:
        section_map[summary.section_key].append(summary)

    rendered_sections = [entry for entry in DEFAULT_SECTION_ORDER if section_map.get(entry[0])]
    n_cols = 2 + 3 + 3 + len(compare_modes)

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{ll lll rrr " + "r" * len(compare_modes) + r"}")
    lines.append(r"\toprule")

    delta_headers = [rf"$\Delta_{{{delta_base}-{mode}}}$" for mode in compare_modes]
    header_cells = [
        "",
        "Config",
        r"$\mathrm{lr}_A$",
        r"$\mathrm{lr}_B$",
        r"$\mathrm{lr}_C$",
        r"$\mathrm{Acc}_A$",
        r"$\mathrm{Acc}_B$",
        r"$\mathrm{Acc}_C$",
    ] + delta_headers
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for i, (section_key, section_title) in enumerate(rendered_sections):
        section_rows = section_map.get(section_key, [])
        n_values = [row.n_items for row in section_rows if row.n_items is not None]
        n_suffix = f" (${Counter(n_values).most_common(1)[0][0]}$ items)" if n_values else ""
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\emph{{{section_title}}}{n_suffix}}} \\")
        lines.append(r"\addlinespace[0.3em]")

        for row in section_rows:
            best_modes = set(_best_modes(row))
            acc_cells: List[str] = []
            for mode in MODES:
                value = row.means.get(mode, float("nan"))
                cell = _fmt_acc(value)
                if mode in best_modes and cell != "--":
                    cell = rf"\mathbf{{{cell}}}"
                acc_cells.append(rf"${cell}$" if cell != "--" else "--")

            delta_cells: List[str] = []
            base_value = row.means.get(delta_base, float("nan"))
            for mode in compare_modes:
                other_value = row.means.get(mode, float("nan"))
                delta = base_value - other_value if math.isfinite(base_value) and math.isfinite(other_value) else float("nan")
                cell = _fmt_delta(delta)
                delta_cells.append(rf"${cell}$" if cell != "--" else "--")

            lines.append(
                " & ".join(
                    [
                        "",
                        _latex_escape(row.config_label),
                        _latex_escape(row.selected_lr_A),
                        _latex_escape(row.selected_lr_B),
                        _latex_escape(row.selected_lr_C),
                    ]
                    + acc_cells
                    + delta_cells
                )
                + r" \\"
            )

        if i != len(rendered_sections) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def _write_summary_csv(path: Path, summaries: Sequence[SectionSummary], delta_base: str, compare_modes: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delta_base = str(delta_base).upper()
    compare_modes = [str(mode).upper() for mode in compare_modes]

    fieldnames = [
        "section_key",
        "section_title",
        "config_label",
    ] + list(CONFIG_FIELDS) + [
        "n_runs",
        "n_items",
        "selected_lr_A",
        "selected_lr_B",
        "selected_lr_C",
    ]
    for mode in MODES:
        fieldnames += [f"acc_{mode}_mean", f"acc_{mode}_std"]
    for mode in compare_modes:
        fieldnames += [f"delta_{delta_base}_minus_{mode}_mean"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row: Dict[str, Any] = {
                "section_key": summary.section_key,
                "section_title": summary.section_title,
                "config_label": summary.config_label,
                "n_runs": summary.n_runs,
                "n_items": summary.n_items,
                "selected_lr_A": summary.selected_lr_A,
                "selected_lr_B": summary.selected_lr_B,
                "selected_lr_C": summary.selected_lr_C,
            }
            for field in CONFIG_FIELDS:
                row[field] = summary.config.get(field, None)
            for mode in MODES:
                row[f"acc_{mode}_mean"] = summary.means.get(mode, float("nan"))
                row[f"acc_{mode}_std"] = summary.stds.get(mode, float("nan"))
            base_value = summary.means.get(delta_base, float("nan"))
            for mode in compare_modes:
                other_value = summary.means.get(mode, float("nan"))
                row[f"delta_{delta_base}_minus_{mode}_mean"] = (
                    base_value - other_value if math.isfinite(base_value) and math.isfinite(other_value) else float("nan")
                )
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact ChaosNLI LaTeX results table directly from final run JSONs. "
            "When older run JSONs do not contain the sliced test sections, --slice-root "
            "can be provided as a backward-compatible fallback."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing final ChaosNLI run JSON files.")
    parser.add_argument("--slice-root", type=Path, default=None, help="Optional fallback directory containing per-run *.slice_eval.csv files.")
    parser.add_argument("--out-tex", type=Path, required=True, help="Output LaTeX table path.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional tidy CSV summary path.")
    parser.add_argument("--train-size", type=int, default=None, help="Optional exact filter on effective train size.")
    parser.add_argument("--all-train-sizes", action="store_true", help="Keep all effective train sizes.")
    parser.add_argument("--keep-train-size-in-config", action="store_true", help="Append N=<train size> to config labels.")
    parser.add_argument("--heads", nargs="*", default=None, help="Optional subset of heads to keep, e.g. linear mlp.")
    parser.add_argument("--source-subsets", nargs="*", default=None, help="Optional subset selectors matching source_subsets strings.")
    parser.add_argument("--selection-splits", nargs="*", default=None, help="Optional subset of selection splits.")
    parser.add_argument("--train-sections", nargs="*", default=None, help="Optional subset of train sections.")
    parser.add_argument("--keep-train-section-in-config", action="store_true", help="Always include train_section in config labels.")
    parser.add_argument("--delta-base", choices=["A", "B", "C"], default="A", help="Mode used as the reference in delta columns.")
    parser.add_argument("--compare-modes", nargs="*", default=["B", "C"], help="Modes to subtract from --delta-base.")
    parser.add_argument(
        "--caption",
        default=(
            "Top-1 accuracy on ChaosNLI for Models~A (projection target), B (antipignistic target), and C (vote proportions), "
            "together with the selected learning rates for each target. Accuracies are averaged over paired runs."
        ),
    )
    parser.add_argument("--label", default="tab:chaosnli-results")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_paths = _iter_run_jsons(run_root)
    if not run_paths:
        raise SystemExit(f"No run JSON files found under {args.run_root}")

    records: List[RunRecord] = []
    for run_json in run_paths:
        record = _extract_run_record(run_json, run_root=run_root, slice_root=args.slice_root)
        if record is not None:
            records.append(record)

    if not records:
        raise SystemExit("No ChaosNLI run records could be extracted.")

    pre_filtered = _filter_records(
        records,
        train_size=None,
        heads=args.heads,
        source_subsets=args.source_subsets,
        selection_splits=args.selection_splits,
        train_sections=args.train_sections,
    )
    if not pre_filtered:
        raise SystemExit("No records left after filtering.")

    size_values = {record.config.get("effective_train_size", None) for record in pre_filtered if record.config.get("effective_train_size", None) is not None}
    if bool(args.all_train_sizes):
        train_size = None
    elif args.train_size is not None:
        train_size = _resolve_train_size(pre_filtered, int(args.train_size))
    elif len(size_values) <= 1:
        train_size = _resolve_train_size(pre_filtered, None)
    else:
        train_size = None

    filtered = (
        list(pre_filtered)
        if train_size is None
        else [record for record in pre_filtered if record.config.get("effective_train_size", None) == train_size]
    )
    if not filtered:
        raise SystemExit("No records left after train-size filtering.")

    keep_train_size_in_config = bool(args.keep_train_size_in_config)
    if train_size is None and len(size_values) > 1:
        keep_train_size_in_config = True

    summaries = _aggregate_sections(
        filtered,
        keep_train_size=keep_train_size_in_config,
        keep_train_section=bool(args.keep_train_section_in_config),
        section_order=DEFAULT_SECTION_ORDER,
    )
    if not summaries:
        raise SystemExit("No section summaries could be built.")

    latex = _build_latex_table(
        summaries,
        caption=str(args.caption),
        label=str(args.label),
        delta_base=str(args.delta_base),
        compare_modes=list(args.compare_modes),
    )

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(latex + "\n", encoding="utf-8")

    if args.out_csv is not None:
        _write_summary_csv(
            Path(args.out_csv),
            summaries,
            delta_base=str(args.delta_base),
            compare_modes=list(args.compare_modes),
        )

    summary_payload = {
        "n_run_jsons_scanned": len(run_paths),
        "n_records_used": len(filtered),
        "train_size_used": train_size,
        "all_train_sizes": train_size is None,
        "train_sections": sorted({record.config.get("train_section", "") for record in filtered}),
        "out_tex": str(args.out_tex),
        "out_csv": str(args.out_csv) if args.out_csv is not None else "",
        "configs": [summary.config_label for summary in summaries if summary.section_key == "test_full"],
    }
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()