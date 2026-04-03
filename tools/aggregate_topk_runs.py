#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.agg_common import (
    aggregate_projection,
    aggregate_split,
    alpha_tag,
    available_metric_names,
    available_projection_keys,
    available_projection_metric_names,
    latex_table,
    load_runs,
    prefer_metric_order,
    warn_if_hyperparams_mismatch,
    write_csv,
)

BASE_METRICS: List[str] = [
    "acc",
    "nll",
    "ece",
    "brier",
    "mass_plausible",
    "top1_in_plausible",
]

OPTIONAL_METRICS: List[str] = [
    "entropy",
    "V_mean",
    "V_median",
    "V_p90",
    "V_max",
    "V_le_tau",
    "violated_mean",
    "viol_pref_mean",
    "viol_low_mean",
    "viol_up_mean",
    "Vpref_mean",
    "Vlow_mean",
    "Vup_mean",
]

DEFAULT_PROJ_METRICS: List[str] = [
    "calls",
    "cycles_mean",
    "cycles_p90",
    "time_mean_s",
    "finalV_mean",
    "finalV_max",
]

PROJECTION_KEYS: List[str] = [
    "projection_stats_train_A",
]

EXCLUDED_METRICS = {"n"}


def projection_descriptor(projection_key: str) -> tuple[str, str]:
    mapping = {
        "projection_stats_train_A": (
            "proj_train_A",
            "Aggregation over repeated experiment runs (projection stats on train)",
        ),
    }
    return mapping.get(
        projection_key,
        (
            projection_key.replace("projection_stats_", "proj_"),
            f"Aggregation over repeated experiment runs ({projection_key})",
        ),
    )


def resolve_metrics(runs, user_metrics: Optional[List[str]]) -> List[str]:
    if user_metrics:
        return [m for m in user_metrics if m not in EXCLUDED_METRICS]

    available = set(available_metric_names(runs, "train")) | set(available_metric_names(runs, "test"))
    available -= EXCLUDED_METRICS
    return prefer_metric_order(sorted(available), BASE_METRICS + OPTIONAL_METRICS)


def resolve_proj_metrics(runs, user_metrics: Optional[List[str]]) -> List[str]:
    if user_metrics:
        return list(user_metrics)

    available = available_projection_metric_names(runs)
    return prefer_metric_order(available, DEFAULT_PROJ_METRICS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate exp-run JSON logs over repeated runs.")
    p.add_argument("--input-dir", type=str, default="out", help="Directory containing *.json run logs.")
    p.add_argument("--alpha", type=float, default=None, help="Optional alpha selector for multi-alpha logs.")
    p.add_argument("--alpha-tol", type=float, default=1e-12)
    p.add_argument("--out-dir", type=str, default="out")
    p.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit metric list. If omitted, infer a sensible exp metric set from the logs.",
    )
    p.add_argument(
        "--proj-metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit projection metric list. If omitted, infer from the logs.",
    )
    return p.parse_args()


def _name_prefix(alpha: Optional[float]) -> str:
    return f"agg_exp_alpha{alpha_tag(alpha)}" if alpha is not None else "agg_exp"


def _caption(title: str, alpha: Optional[float]) -> str:
    if alpha is None:
        return f"{title}."
    return f"{title}, $\\alpha={alpha}$."


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    runs = load_runs(
        input_dir=input_dir,
        alpha=args.alpha,
        tol=float(args.alpha_tol),
        projection_keys=PROJECTION_KEYS,
        allowed_cmds=("topk-exp", "run"),
    )
    if not runs:
        selector = "unqualified result" if args.alpha is None else f"alpha={args.alpha}"
        raise SystemExit(f"No runs found in {input_dir} for {selector} (tol={args.alpha_tol}).")

    warn_if_hyperparams_mismatch(runs)

    metrics = resolve_metrics(runs, args.metrics)
    proj_metrics = resolve_proj_metrics(runs, args.proj_metrics)

    rows_train = aggregate_split(runs, split="train", metrics=metrics)
    rows_test = aggregate_split(runs, split="test", metrics=metrics)

    proj_keys = available_projection_keys(runs, PROJECTION_KEYS)
    rows_proj_all = []
    proj_tables = []

    for projection_key in proj_keys:
        split_name, caption_title = projection_descriptor(projection_key)
        rows_proj = aggregate_projection(
            runs,
            metrics=proj_metrics,
            split_name=split_name,
            projection_key=projection_key,
            mode="A",
        )
        if not rows_proj:
            continue

        rows_proj_all.extend(rows_proj)
        tex_path = out_dir / f"{_name_prefix(args.alpha)}_{split_name}.tex"
        proj_tables.append(
            (
                tex_path,
                rows_proj,
                _caption(caption_title, args.alpha),
                f"tab:{_name_prefix(args.alpha)}_{split_name}",
            )
        )

    all_rows = rows_train + rows_test + rows_proj_all
    prefix = _name_prefix(args.alpha)

    csv_path = out_dir / f"{prefix}.csv"
    tex_train = out_dir / f"{prefix}_train.tex"
    tex_test = out_dir / f"{prefix}_test.tex"

    write_csv(csv_path, all_rows)

    tex_train.write_text(
        latex_table(
            rows_train,
            caption=_caption("Aggregation over repeated experiment runs (train split)", args.alpha),
            label=f"tab:{prefix}_train",
        ),
        encoding="utf-8",
    )
    tex_test.write_text(
        latex_table(
            rows_test,
            caption=_caption("Aggregation over repeated experiment runs (test split)", args.alpha),
            label=f"tab:{prefix}_test",
        ),
        encoding="utf-8",
    )

    wrote_tables = [str(tex_train), str(tex_test)]

    for tex_path, rows_proj, caption_text, label in proj_tables:
        tex_path.write_text(
            latex_table(
                rows_proj,
                caption=caption_text,
                label=label,
            ),
            encoding="utf-8",
        )
        wrote_tables.append(str(tex_path))

    if args.alpha is None:
        print(f"runs={len(runs)}")
    else:
        print(f"alpha={args.alpha}  runs={len(runs)}")
    print(f"wrote: {csv_path}")
    print("wrote tables: " + " ".join(wrote_tables))


if __name__ == "__main__":
    main()