#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROTOCOL_ORDER: Tuple[str, ...] = ("val_full", "val_S_amb", "val_S_easy")
SECTION_ORDER: Tuple[str, ...] = ("test_full", "test_S_amb", "test_S_easy")
TRAIN_SECTION_ORDER: Tuple[str, ...] = ("train_full", "train_S_amb", "train_S_easy")

PROTOCOL_LABEL: Dict[str, str] = {
    "val_full": r"\texttt{val\_full}",
    "val_S_amb": r"\texttt{val\_S\_amb}",
    "val_S_easy": r"\texttt{val\_S\_easy}",
}

SECTION_LABEL: Dict[str, str] = {
    "test_full": r"\texttt{test\_full}",
    "test_S_amb": r"\texttt{test\_S\_amb}",
    "test_S_easy": r"\texttt{test\_S\_easy}",
}

TRAIN_SECTION_LABEL: Dict[str, str] = {
    "train_full": r"\texttt{train\_full}",
    "train_S_amb": r"\texttt{train\_S\_amb}",
    "train_S_easy": r"\texttt{train\_S\_easy}",
}

REQUIRED_COLUMNS = {
    "selection_split",
    "section_key",
    "section_title",
    "config_label",
    "head",
    "variant",
    "source_subsets",
    "split_seed",
    "train_frac",
    "val_frac",
    "train_section",
    "n_runs",
    "n_items",
    "selected_lr_A",
    "selected_lr_B",
    "selected_lr_C",
    "acc_A_mean",
    "acc_A_std",
    "acc_B_mean",
    "acc_B_std",
    "acc_C_mean",
    "acc_C_std",
}


@dataclass(frozen=True)
class RowRec:
    protocol: str
    section_key: str
    section_title: str
    config_label: str
    head: str
    variant: str
    source_subsets: str
    split_seed: str
    train_frac: str
    val_frac: str
    train_section: str
    n_runs: int
    n_items: Optional[int]
    selected_lr_A: str
    selected_lr_B: str
    selected_lr_C: str
    acc_A_mean: float
    acc_A_std: float
    acc_B_mean: float
    acc_B_std: float
    acc_C_mean: float
    acc_C_std: float
    delta_A_minus_B_mean: float
    delta_A_minus_C_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one merged article-facing ChaosNLI train-section LaTeX table "
            "from the per-protocol CSV summaries."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out/real_nlp/chaosnli/train_section_sweep/article"),
        help="Directory under which CSV files are auto-discovered.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        help="Specific CSV file(s) to read. Can be passed multiple times.",
    )
    parser.add_argument("--head", default=None, help="Optional exact head filter, e.g. linear.")
    parser.add_argument("--variant", default=None, help="Optional exact variant filter.")
    parser.add_argument("--source-subsets", default=None, help="Optional exact source_subsets filter.")
    parser.add_argument("--split-seed", default=None, help="Optional exact split_seed filter.")
    parser.add_argument("--out-tex", type=Path, default=None, help="Optional path to write the merged LaTeX table.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional path to write the merged tidy CSV.")
    parser.add_argument(
        "--label",
        default="tab:chaosnli-train-section-merged",
        help="LaTeX label for the merged table.",
    )
    parser.add_argument(
        "--caption",
        default=(
        r"Top-1 accuracy on ChaosNLI when the training set is restricted to "
        r"\texttt{train\_full}, \texttt{train\_S\_amb}, or \texttt{train\_S\_easy}, "
        r"for Models~A (projection target), B (antipignistic target), and C (vote-proportion target). "
        r"The selected learning rates for each target are reported alongside the accuracies. "
        r"Results are organized by training section, validation section, and test section. "
        r"Accuracies are mean $\pm$ standard deviation over paired runs, and the best mean accuracy in each row is shown in bold."
        ),
        help="Caption text for the merged LaTeX table.",
    )
    return parser.parse_args()


def discover_csvs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def _safe_float(value: str) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def _read_candidate_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            return []
        return [dict(row) for row in reader]


def _row_from_dict(row: Dict[str, str]) -> RowRec:
    acc_A_mean = _safe_float(row.get("acc_A_mean", ""))
    acc_A_std = _safe_float(row.get("acc_A_std", ""))
    acc_B_mean = _safe_float(row.get("acc_B_mean", ""))
    acc_B_std = _safe_float(row.get("acc_B_std", ""))
    acc_C_mean = _safe_float(row.get("acc_C_mean", ""))
    acc_C_std = _safe_float(row.get("acc_C_std", ""))

    delta_A_minus_B_mean = _safe_float(row.get("delta_A_minus_B_mean", ""))
    if not math.isfinite(delta_A_minus_B_mean) and math.isfinite(acc_A_mean) and math.isfinite(acc_B_mean):
        delta_A_minus_B_mean = acc_A_mean - acc_B_mean

    delta_A_minus_C_mean = _safe_float(row.get("delta_A_minus_C_mean", ""))
    if not math.isfinite(delta_A_minus_C_mean) and math.isfinite(acc_A_mean) and math.isfinite(acc_C_mean):
        delta_A_minus_C_mean = acc_A_mean - acc_C_mean

    return RowRec(
        protocol=str(row.get("selection_split", "")).strip(),
        section_key=str(row.get("section_key", "")).strip(),
        section_title=str(row.get("section_title", "")).strip(),
        config_label=str(row.get("config_label", "")).strip(),
        head=str(row.get("head", "")).strip(),
        variant=str(row.get("variant", "")).strip(),
        source_subsets=str(row.get("source_subsets", "")).strip(),
        split_seed=str(row.get("split_seed", "")).strip(),
        train_frac=str(row.get("train_frac", "")).strip(),
        val_frac=str(row.get("val_frac", "")).strip(),
        train_section=str(row.get("train_section", "")).strip() or "train_full",
        n_runs=int(float(row.get("n_runs", "0") or 0)),
        n_items=_safe_int(row.get("n_items", "")),
        selected_lr_A=str(row.get("selected_lr_A", "")).strip(),
        selected_lr_B=str(row.get("selected_lr_B", "")).strip(),
        selected_lr_C=str(row.get("selected_lr_C", "")).strip(),
        acc_A_mean=acc_A_mean,
        acc_A_std=acc_A_std,
        acc_B_mean=acc_B_mean,
        acc_B_std=acc_B_std,
        acc_C_mean=acc_C_mean,
        acc_C_std=acc_C_std,
        delta_A_minus_B_mean=delta_A_minus_B_mean,
        delta_A_minus_C_mean=delta_A_minus_C_mean,
    )


def read_rows(csv_paths: Iterable[Path]) -> List[RowRec]:
    rows: List[RowRec] = []
    seen = set()

    for path in csv_paths:
        for raw_row in _read_candidate_csv(path):
            rec = _row_from_dict(raw_row)
            if rec.protocol not in PROTOCOL_ORDER or rec.section_key not in SECTION_ORDER:
                continue

            key = (
                rec.protocol,
                rec.section_key,
                rec.train_section,
                rec.head,
                rec.variant,
                rec.source_subsets,
                rec.split_seed,
            )
            if key in seen:
                raise ValueError(
                    "Duplicate merged row detected for "
                    f"protocol={rec.protocol!r}, section={rec.section_key!r}, train_section={rec.train_section!r}."
                )
            seen.add(key)
            rows.append(rec)

    return rows


def keep_row(row: RowRec, args: argparse.Namespace) -> bool:
    if args.head is not None and row.head != args.head:
        return False
    if args.variant is not None and row.variant != args.variant:
        return False
    if args.source_subsets is not None and row.source_subsets != args.source_subsets:
        return False
    if args.split_seed is not None and row.split_seed != args.split_seed:
        return False
    return True


def _train_section_rank(name: str) -> Tuple[int, str]:
    if name in TRAIN_SECTION_ORDER:
        return (TRAIN_SECTION_ORDER.index(name), name)
    return (len(TRAIN_SECTION_ORDER), name)


def _protocol_rank(name: str) -> Tuple[int, str]:
    if name in PROTOCOL_ORDER:
        return (PROTOCOL_ORDER.index(name), name)
    return (len(PROTOCOL_ORDER), name)


def _section_rank(name: str) -> Tuple[int, str]:
    if name in SECTION_ORDER:
        return (SECTION_ORDER.index(name), name)
    return (len(SECTION_ORDER), name)


def _sort_key(row: RowRec) -> Tuple[int, int, int, str, str, str]:
    return (
        _train_section_rank(row.train_section)[0],
        _protocol_rank(row.protocol)[0],
        _section_rank(row.section_key)[0],
        row.head,
        row.variant,
        row.source_subsets,
    )


def format_mean_std(mean: float, std: float, *, bold: bool = False) -> str:
    if not math.isfinite(mean):
        return "--"
    body = f"{mean:.3f}\\pm{std:.3f}" if math.isfinite(std) else f"{mean:.3f}"
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def format_delta(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return rf"${value:+.3f}$"


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash ")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _train_cell(train_section: str, span: int) -> str:
    body = rf"\makecell[l]{{{TRAIN_SECTION_LABEL.get(train_section, latex_escape(train_section))}}}"
    return rf"\multirow[t]{{{span}}}{{*}}[-0.35em]{{{body}}}"


def _val_cell(protocol: str, span: int) -> str:
    body = rf"\makecell[l]{{{PROTOCOL_LABEL.get(protocol, latex_escape(protocol))}}}"
    return rf"\multirow[t]{{{span}}}{{*}}[-0.35em]{{{body}}}"


def _test_cell(section: str, n_items: Optional[int]) -> str:
    del n_items
    label = SECTION_LABEL.get(section, latex_escape(section))
    return rf"\makecell[l]{{{label}}}"


def render_latex(rows: Sequence[RowRec], *, caption: str, label: str) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{0.9pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.}")
    lines.append(r"\begin{tabular}{lll lll rrr rr}")
    lines.append(r"\toprule")
    lines.append(
        r"Train section & Val section & Test section & $\mathrm{lr}_A$ & $\mathrm{lr}_B$ & $\mathrm{lr}_C$ & "
        r"$\mathrm{Acc}_A$ & $\mathrm{Acc}_B$ & $\mathrm{Acc}_C$ & $\Delta_{A-B}$ & $\Delta_{A-C}$ \\"
    )
    lines.append(r"\midrule")

    ordered = sorted(rows, key=_sort_key)

    by_train: Dict[str, List[RowRec]] = {}
    for row in ordered:
        by_train.setdefault(row.train_section, []).append(row)

    train_keys = [k for k in TRAIN_SECTION_ORDER if k in by_train]
    train_keys.extend(sorted(k for k in by_train if k not in TRAIN_SECTION_ORDER))

    for train_index, train_section in enumerate(train_keys):
        train_rows = sorted(by_train[train_section], key=_sort_key)
        train_span = len(train_rows)
        train_cell = _train_cell(train_section, train_span)

        by_val: Dict[str, List[RowRec]] = {}
        for row in train_rows:
            by_val.setdefault(row.protocol, []).append(row)

        val_keys = [k for k in PROTOCOL_ORDER if k in by_val]
        val_keys.extend(sorted(k for k in by_val if k not in PROTOCOL_ORDER))

        first_row_for_train = True

        for val_index, protocol in enumerate(val_keys):
            val_rows = sorted(by_val[protocol], key=lambda row: _section_rank(row.section_key))
            val_span = len(val_rows)
            val_cell = _val_cell(protocol, val_span)

            for row_index, row in enumerate(val_rows):
                finite_scores = [v for v in (row.acc_A_mean, row.acc_B_mean, row.acc_C_mean) if math.isfinite(v)]
                best = max(finite_scores) if finite_scores else float("nan")

                acc_A = format_mean_std(
                    row.acc_A_mean,
                    row.acc_A_std,
                    bold=math.isfinite(row.acc_A_mean) and abs(row.acc_A_mean - best) <= 1e-12,
                )
                acc_B = format_mean_std(
                    row.acc_B_mean,
                    row.acc_B_std,
                    bold=math.isfinite(row.acc_B_mean) and abs(row.acc_B_mean - best) <= 1e-12,
                )
                acc_C = format_mean_std(
                    row.acc_C_mean,
                    row.acc_C_std,
                    bold=math.isfinite(row.acc_C_mean) and abs(row.acc_C_mean - best) <= 1e-12,
                )

                first_col = train_cell if first_row_for_train and row_index == 0 else ""
                second_col = val_cell if row_index == 0 else ""
                third_col = _test_cell(row.section_key, row.n_items)

                lines.append(
                    " & ".join(
                        [
                            first_col,
                            second_col,
                            third_col,
                            latex_escape(row.selected_lr_A or "--"),
                            latex_escape(row.selected_lr_B or "--"),
                            latex_escape(row.selected_lr_C or "--"),
                            acc_A,
                            acc_B,
                            acc_C,
                            format_delta(row.delta_A_minus_B_mean),
                            format_delta(row.delta_A_minus_C_mean),
                        ]
                    )
                    + r" \\"
                )

            first_row_for_train = False

            if val_index != len(val_keys) - 1:
                lines.append(r"\cmidrule(lr){2-11}")

        if train_index != len(train_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[RowRec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selection_split",
        "section_key",
        "section_title",
        "train_section",
        "config_label",
        "head",
        "variant",
        "source_subsets",
        "split_seed",
        "train_frac",
        "val_frac",
        "n_runs",
        "n_items",
        "selected_lr_A",
        "selected_lr_B",
        "selected_lr_C",
        "acc_A_mean",
        "acc_A_std",
        "acc_B_mean",
        "acc_B_std",
        "acc_C_mean",
        "acc_C_std",
        "delta_A_minus_B_mean",
        "delta_A_minus_C_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "selection_split": row.protocol,
                    "section_key": row.section_key,
                    "section_title": row.section_title,
                    "train_section": row.train_section,
                    "config_label": row.config_label,
                    "head": row.head,
                    "variant": row.variant,
                    "source_subsets": row.source_subsets,
                    "split_seed": row.split_seed,
                    "train_frac": row.train_frac,
                    "val_frac": row.val_frac,
                    "n_runs": row.n_runs,
                    "n_items": row.n_items,
                    "selected_lr_A": row.selected_lr_A,
                    "selected_lr_B": row.selected_lr_B,
                    "selected_lr_C": row.selected_lr_C,
                    "acc_A_mean": row.acc_A_mean,
                    "acc_A_std": row.acc_A_std,
                    "acc_B_mean": row.acc_B_mean,
                    "acc_B_std": row.acc_B_std,
                    "acc_C_mean": row.acc_C_mean,
                    "acc_C_std": row.acc_C_std,
                    "delta_A_minus_B_mean": row.delta_A_minus_B_mean,
                    "delta_A_minus_C_mean": row.delta_A_minus_C_mean,
                }
            )


def main() -> int:
    args = parse_args()

    csv_paths = list(args.csv)
    if not csv_paths:
        csv_paths = discover_csvs(args.root)

    csv_paths = [Path(p).resolve() for p in csv_paths]

    if not csv_paths:
        print("No CSV files found.", file=sys.stderr, flush=True)
        return 1

    missing_inputs = [str(p) for p in csv_paths if not p.exists()]
    if missing_inputs:
        print("Missing input CSV files:", file=sys.stderr, flush=True)
        for p in missing_inputs:
            print(f"  {p}", file=sys.stderr, flush=True)
        return 1

    rows = read_rows(csv_paths)
    rows = [row for row in rows if keep_row(row, args)]

    if not rows:
        print("No matching rows found after filtering.", file=sys.stderr, flush=True)
        print("Input CSVs checked:", file=sys.stderr, flush=True)
        for p in csv_paths:
            print(f"  {p}", file=sys.stderr, flush=True)
        return 1

    latex = render_latex(rows, caption=str(args.caption), label=str(args.label))

    if args.out_tex is not None:
        out_tex = Path(args.out_tex).resolve()
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(latex + "\n", encoding="utf-8")
        print(f"WROTE_TEX={out_tex}", flush=True)

    if args.out_csv is not None:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out_csv, rows)
        print(f"WROTE_CSV={out_csv}", flush=True)

    print(f"N_ROWS={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())