# ===== tools/agg_common.py =====
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.io_utils import load_json


# Direction: +1 means "higher is better", -1 means "lower is better".
DIRECTION: Dict[str, int] = {
    "acc": +1,
    "mass_plausible": +1,
    "top1_in_plausible": +1,
    "V_le_tau": +1,
    "nll": -1,
    "brier": -1,
    "ece": -1,
    "entropy": -1,
    "V_mean": -1,
    "V_median": -1,
    "V_p90": -1,
    "V_max": -1,
    "violated_mean": -1,
    "viol_pref_mean": -1,
    "viol_low_mean": -1,
    "viol_up_mean": -1,
    "Vpref_mean": -1,
    "Vlow_mean": -1,
    "Vup_mean": -1,
    "calls": -1,
    "cycles_mean": -1,
    "cycles_p90": -1,
    "time_mean_s": -1,
    "finalV_mean": -1,
    "finalV_max": -1,
}

MODES: Tuple[str, ...] = ("A", "B", "C")


@dataclass(frozen=True)
class RunRecord:
    path: Path
    alpha: Optional[float]
    metrics: Dict[str, Dict[str, Dict[str, float]]]
    projection_stats: Dict[str, Dict[str, float]]
    hyperparams: Dict[str, Any]


@dataclass(frozen=True)
class AggRow:
    split: str
    metric: str
    n: int
    A_mean: float
    A_std: float
    B_mean: float
    B_std: float
    C_mean: float
    C_std: float
    BA_mean: float
    BA_std: float
    CA_mean: float
    CA_std: float
    CB_mean: float
    CB_std: float
    win_rate_B: float
    win_rate_C_vs_A: float
    win_rate_C_vs_B: float


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):.12g}"
    return text.replace(".", "p").replace("-", "m")


def fmt(x: float) -> str:
    ax = abs(float(x))
    if ax == 0.0:
        return "0"
    if ax < 1e-3 or ax >= 1e3:
        return f"{x:.2e}"
    if ax < 1.0:
        return f"{x:.4f}"
    if ax < 10.0:
        return f"{x:.3f}"
    return f"{x:.2f}"


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), 0.0)
    return (float(np.mean(arr)), float(np.std(arr, ddof=1)))


def _result_blocks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = doc.get("result")
    if isinstance(result, dict):
        return [result]

    results = doc.get("results", [])
    if not isinstance(results, list):
        return []

    return [row for row in results if isinstance(row, dict)]


def _extract_one_result(
    doc: Dict[str, Any],
    alpha_target: Optional[float],
    tol: float,
) -> Optional[Dict[str, Any]]:
    results = _result_blocks(doc)
    if not results:
        return None

    if alpha_target is None:
        if len(results) == 1:
            return results[0]
        no_alpha = [row for row in results if not is_number(row.get("alpha", None))]
        if len(no_alpha) == 1:
            return no_alpha[0]
        return None

    for row in results:
        alpha_value = row.get("alpha", None)
        if is_number(alpha_value) and abs(float(alpha_value) - float(alpha_target)) <= tol:
            return row

    return None


def _extract_numeric_dict(block: Any) -> Dict[str, float]:
    if not isinstance(block, dict):
        return {}

    out: Dict[str, float] = {}
    for key, value in block.items():
        if is_number(value):
            out[str(key)] = float(value)
    return out


def load_runs(
    input_dir: Path,
    alpha: Optional[float],
    tol: float,
    *,
    projection_keys: Sequence[str],
    allowed_cmds: Optional[Sequence[str]] = None,
) -> List[RunRecord]:
    runs: List[RunRecord] = []
    allowed_cmds_set = None if allowed_cmds is None else {str(x) for x in allowed_cmds}

    for path in sorted(input_dir.glob("*.json")):
        try:
            doc = load_json(path)
        except Exception:
            continue

        if allowed_cmds_set is not None:
            cmd = str(doc.get("cmd", ""))
            if cmd not in allowed_cmds_set:
                continue

        result = _extract_one_result(doc, alpha_target=alpha, tol=tol)
        if result is None:
            continue

        metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split in ("train", "test", "val_full", "val_amb", "test_full", "test_amb"):
            split_block = result.get(split, {})
            if not isinstance(split_block, dict):
                continue

            metrics[split] = {}
            for mode in MODES:
                mode_block = split_block.get(mode, {})
                if isinstance(mode_block, dict):
                    metrics[split][mode] = _extract_numeric_dict(mode_block)

        projection_stats: Dict[str, Dict[str, float]] = {}
        for key in projection_keys:
            proj_block = result.get(key, None)
            numeric_block = _extract_numeric_dict(proj_block)
            if numeric_block:
                projection_stats[str(key)] = numeric_block

        hyperparams = doc.get("hyperparams", {})
        if not isinstance(hyperparams, dict):
            hyperparams = {}

        alpha_loaded = result.get("alpha", None)
        alpha_value = float(alpha_loaded) if is_number(alpha_loaded) else None

        runs.append(
            RunRecord(
                path=path,
                alpha=alpha_value,
                metrics=metrics,
                projection_stats=projection_stats,
                hyperparams=hyperparams,
            )
        )

    return runs


def available_metric_names(runs: List[RunRecord], split: str) -> List[str]:
    seen = set()
    for run in runs:
        for mode in MODES:
            seen.update(run.metrics.get(split, {}).get(mode, {}).keys())
    return sorted(seen)


def available_projection_keys(
    runs: List[RunRecord],
    preferred: Optional[Sequence[str]] = None,
) -> List[str]:
    ordered: List[str] = []
    seen = set()

    if preferred is not None:
        for key in preferred:
            key_text = str(key)
            if any(key_text in run.projection_stats for run in runs):
                ordered.append(key_text)
                seen.add(key_text)

    for run in runs:
        for key in run.projection_stats.keys():
            key_text = str(key)
            if key_text not in seen:
                ordered.append(key_text)
                seen.add(key_text)

    return ordered


def available_projection_metric_names(
    runs: List[RunRecord],
    projection_key: Optional[str] = None,
) -> List[str]:
    seen = set()
    for run in runs:
        if projection_key is None:
            for block in run.projection_stats.values():
                seen.update(block.keys())
        else:
            seen.update(run.projection_stats.get(projection_key, {}).keys())
    return sorted(seen)


def prefer_metric_order(available: Sequence[str], preferred: Sequence[str]) -> List[str]:
    available_set = set(available)
    ordered: List[str] = []

    for metric in preferred:
        if metric in available_set and metric not in ordered:
            ordered.append(metric)

    extras = sorted(available_set - set(ordered))
    ordered.extend(extras)
    return ordered


def _mean_std_or_nan(values: List[float]) -> Tuple[float, float]:
    return mean_std(values) if values else (float("nan"), float("nan"))


def aggregate_split(runs: List[RunRecord], split: str, metrics: List[str]) -> List[AggRow]:
    rows: List[AggRow] = []

    for metric in metrics:
        values_by_mode: Dict[str, List[float]] = {mode: [] for mode in MODES}
        diffs_ba: List[float] = []
        diffs_ca: List[float] = []
        diffs_cb: List[float] = []
        wins_b: List[int] = []
        wins_ca: List[int] = []
        wins_cb: List[int] = []

        direction = int(DIRECTION.get(metric, -1))
        n_any = 0

        for run in runs:
            present: Dict[str, float] = {}

            for mode in MODES:
                value = run.metrics.get(split, {}).get(mode, {}).get(metric, None)
                if value is not None:
                    value_f = float(value)
                    present[mode] = value_f
                    values_by_mode[mode].append(value_f)

            if present:
                n_any += 1

            if "A" in present and "B" in present:
                diffs_ba.append(present["B"] - present["A"])
                wins_b.append(1 if direction * present["B"] > direction * present["A"] else 0)

            if "A" in present and "C" in present:
                diffs_ca.append(present["C"] - present["A"])
                wins_ca.append(1 if direction * present["C"] > direction * present["A"] else 0)

            if "B" in present and "C" in present:
                diffs_cb.append(present["C"] - present["B"])
                wins_cb.append(1 if direction * present["C"] > direction * present["B"] else 0)

        if n_any == 0:
            continue

        A_mean, A_std = _mean_std_or_nan(values_by_mode["A"])
        B_mean, B_std = _mean_std_or_nan(values_by_mode["B"])
        C_mean, C_std = _mean_std_or_nan(values_by_mode["C"])
        BA_mean, BA_std = _mean_std_or_nan(diffs_ba)
        CA_mean, CA_std = _mean_std_or_nan(diffs_ca)
        CB_mean, CB_std = _mean_std_or_nan(diffs_cb)

        rows.append(
            AggRow(
                split=split,
                metric=metric,
                n=n_any,
                A_mean=A_mean,
                A_std=A_std,
                B_mean=B_mean,
                B_std=B_std,
                C_mean=C_mean,
                C_std=C_std,
                BA_mean=BA_mean,
                BA_std=BA_std,
                CA_mean=CA_mean,
                CA_std=CA_std,
                CB_mean=CB_mean,
                CB_std=CB_std,
                win_rate_B=float(np.mean(wins_b)) if wins_b else float("nan"),
                win_rate_C_vs_A=float(np.mean(wins_ca)) if wins_ca else float("nan"),
                win_rate_C_vs_B=float(np.mean(wins_cb)) if wins_cb else float("nan"),
            )
        )

    return rows


def aggregate_projection(
    runs: List[RunRecord],
    metrics: List[str],
    *,
    split_name: str,
    projection_key: str,
    mode: str = "A",
) -> List[AggRow]:
    mode_text = str(mode).upper()
    if mode_text != "A":
        raise ValueError("projection aggregation mode must be 'A'.")

    rows: List[AggRow] = []

    for metric in metrics:
        values: List[float] = []
        for run in runs:
            value = run.projection_stats.get(projection_key, {}).get(metric, None)
            if value is not None:
                values.append(float(value))

        if not values:
            continue

        mean_value, std_value = mean_std(values)

        rows.append(
            AggRow(
                split=split_name,
                metric=metric,
                n=len(values),
                A_mean=mean_value,
                A_std=std_value,
                B_mean=float("nan"),
                B_std=float("nan"),
                C_mean=float("nan"),
                C_std=float("nan"),
                BA_mean=float("nan"),
                BA_std=float("nan"),
                CA_mean=float("nan"),
                CA_std=float("nan"),
                CB_mean=float("nan"),
                CB_std=float("nan"),
                win_rate_B=float("nan"),
                win_rate_C_vs_A=float("nan"),
                win_rate_C_vs_B=float("nan"),
            )
        )

    return rows


def write_csv(path: Path, rows: List[AggRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "metric",
                "n",
                "A_mean",
                "A_std",
                "B_mean",
                "B_std",
                "C_mean",
                "C_std",
                "B_minus_A_mean",
                "B_minus_A_std",
                "C_minus_A_mean",
                "C_minus_A_std",
                "C_minus_B_mean",
                "C_minus_B_std",
                "win_rate_B",
                "win_rate_C_vs_A",
                "win_rate_C_vs_B",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row.split,
                    row.metric,
                    row.n,
                    row.A_mean,
                    row.A_std,
                    row.B_mean,
                    row.B_std,
                    row.C_mean,
                    row.C_std,
                    row.BA_mean,
                    row.BA_std,
                    row.CA_mean,
                    row.CA_std,
                    row.CB_mean,
                    row.CB_std,
                    row.win_rate_B,
                    row.win_rate_C_vs_A,
                    row.win_rate_C_vs_B,
                ]
            )


def _projection_only_mode(rows: List[AggRow]) -> Optional[str]:
    mode_seen: Optional[str] = None

    for row in rows:
        finite_modes = [
            mode
            for mode, value in (("A", row.A_mean), ("B", row.B_mean), ("C", row.C_mean))
            if math.isfinite(value)
        ]
        if len(finite_modes) != 1:
            return None

        current_mode = finite_modes[0]
        if mode_seen is None:
            mode_seen = current_mode
        elif mode_seen != current_mode:
            return None

    return mode_seen


def _cell_mean_pm(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return r"--"
    return f"{fmt(mean)}\\pm{fmt(std)}"


def _cell_scalar(value: float) -> str:
    if not math.isfinite(value):
        return r"--"
    return fmt(value)


def _latex_row(cells: Sequence[Any]) -> str:
    return " & ".join(str(cell) for cell in cells) + r"\\"


def latex_table(rows: List[AggRow], *, caption: str, label: str) -> str:
    has_A = any(math.isfinite(row.A_mean) for row in rows)
    has_B = any(math.isfinite(row.B_mean) for row in rows)
    has_C = any(math.isfinite(row.C_mean) for row in rows)
    projection_mode = _projection_only_mode(rows) if rows else None

    lines: List[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")

    if projection_mode is not None:
        lines.append(r"\begin{tabular}{lrr}")
        lines.append(r"\toprule")
        lines.append(_latex_row([r"Metric", rf"${projection_mode}$ (mean$\pm$sd)", r"$n$"]))
        lines.append(r"\midrule")

        for row in rows:
            if projection_mode == "A":
                mean_value, std_value = row.A_mean, row.A_std
            elif projection_mode == "B":
                mean_value, std_value = row.B_mean, row.B_std
            else:
                mean_value, std_value = row.C_mean, row.C_std

            lines.append(
                _latex_row(
                    [
                        row.metric,
                        _cell_mean_pm(mean_value, std_value),
                        row.n,
                    ]
                )
            )

    elif has_A and has_B and has_C:
        lines.append(r"\begin{tabular}{lrrrrrr}")
        lines.append(r"\toprule")
        lines.append(
            _latex_row(
                [
                    r"Metric",
                    r"$A$ (mean$\pm$sd)",
                    r"$B$ (mean$\pm$sd)",
                    r"$C$ (mean$\pm$sd)",
                    r"$\Delta_{B-A}$",
                    r"$\Delta_{C-A}$",
                    r"$n$",
                ]
            )
        )
        lines.append(r"\midrule")

        for row in rows:
            lines.append(
                _latex_row(
                    [
                        row.metric,
                        _cell_mean_pm(row.A_mean, row.A_std),
                        _cell_mean_pm(row.B_mean, row.B_std),
                        _cell_mean_pm(row.C_mean, row.C_std),
                        _cell_mean_pm(row.BA_mean, row.BA_std),
                        _cell_mean_pm(row.CA_mean, row.CA_std),
                        row.n,
                    ]
                )
            )

    elif has_A and has_B:
        lines.append(r"\begin{tabular}{lrrrrr}")
        lines.append(r"\toprule")
        lines.append(
            _latex_row(
                [
                    r"Metric",
                    r"$A$ (mean$\pm$sd)",
                    r"$B$ (mean$\pm$sd)",
                    r"$\Delta$ (mean$\pm$sd)",
                    r"win$_B$",
                    r"$n$",
                ]
            )
        )
        lines.append(r"\midrule")

        for row in rows:
            lines.append(
                _latex_row(
                    [
                        row.metric,
                        _cell_mean_pm(row.A_mean, row.A_std),
                        _cell_mean_pm(row.B_mean, row.B_std),
                        _cell_mean_pm(row.BA_mean, row.BA_std),
                        _cell_scalar(row.win_rate_B),
                        row.n,
                    ]
                )
            )

    elif has_B and has_C:
        lines.append(r"\begin{tabular}{lrrrrr}")
        lines.append(r"\toprule")
        lines.append(
            _latex_row(
                [
                    r"Metric",
                    r"$B$ (mean$\pm$sd)",
                    r"$C$ (mean$\pm$sd)",
                    r"$\Delta$ (mean$\pm$sd)",
                    r"win$_C$",
                    r"$n$",
                ]
            )
        )
        lines.append(r"\midrule")

        for row in rows:
            lines.append(
                _latex_row(
                    [
                        row.metric,
                        _cell_mean_pm(row.B_mean, row.B_std),
                        _cell_mean_pm(row.C_mean, row.C_std),
                        _cell_mean_pm(row.CB_mean, row.CB_std),
                        _cell_scalar(row.win_rate_C_vs_B),
                        row.n,
                    ]
                )
            )

    elif has_B:
        lines.append(r"\begin{tabular}{lrr}")
        lines.append(r"\toprule")
        lines.append(_latex_row([r"Metric", r"$B$ (mean$\pm$sd)", r"$n$"]))
        lines.append(r"\midrule")

        for row in rows:
            lines.append(_latex_row([row.metric, _cell_mean_pm(row.B_mean, row.B_std), row.n]))

    elif has_A:
        lines.append(r"\begin{tabular}{lrr}")
        lines.append(r"\toprule")
        lines.append(_latex_row([r"Metric", r"$A$ (mean$\pm$sd)", r"$n$"]))
        lines.append(r"\midrule")

        for row in rows:
            lines.append(_latex_row([row.metric, _cell_mean_pm(row.A_mean, row.A_std), row.n]))

    elif has_C:
        lines.append(r"\begin{tabular}{lrr}")
        lines.append(r"\toprule")
        lines.append(_latex_row([r"Metric", r"$C$ (mean$\pm$sd)", r"$n$"]))
        lines.append(r"\midrule")

        for row in rows:
            lines.append(_latex_row([row.metric, _cell_mean_pm(row.C_mean, row.C_std), row.n]))

    else:
        lines.append(r"\begin{tabular}{lr}")
        lines.append(r"\toprule")
        lines.append(_latex_row([r"Metric", r"$n$"]))
        lines.append(r"\midrule")

        for row in rows:
            lines.append(_latex_row([row.metric, row.n]))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def warn_if_hyperparams_mismatch(runs: List[RunRecord]) -> None:
    if not runs:
        return

    reference = runs[0].hyperparams
    reference_name = runs[0].path.name

    for run in runs[1:]:
        if run.hyperparams != reference:
            print(f"[WARN] hyperparams mismatch: {run.path.name} differs from {reference_name}")
            return