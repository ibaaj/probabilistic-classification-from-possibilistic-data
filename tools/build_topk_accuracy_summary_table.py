#!/usr/bin/env python3
"""
Build one global LaTeX table from aggregated top-k CSV files.

For each configuration and Ntr, read:
  - train acc_A / acc_B
  - test  acc_A / acc_B

Preferred filenames:
  agg_exp_alpha<...>.csv
Fallback filenames:
  agg_alpha<...>.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

OUTPUT_FILENAME = "big_all_test_acc_topk.tex"


@dataclass
class RowRec:
    d: str
    beta: str
    alpha: str
    ntr: int
    lrA: str
    lrB: str
    trA_m: Optional[float]
    trA_s: Optional[float]
    trB_m: Optional[float]
    trB_s: Optional[float]
    teA_m: Optional[float]
    teA_s: Optional[float]
    teB_m: Optional[float]
    teB_s: Optional[float]


def safe_float(text: str) -> Optional[float]:
    try:
        value = float(text)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def token_to_float_str(token: str) -> str:
    return token.replace("p", ".")


def parse_dcs_folder(name: str) -> Tuple[str, str]:
    """
    Parse names like:
        d30_cs1p5
        d80_cs0p9
        d150_cs0p6
    Returns:
        (d, beta)
    """
    match = re.fullmatch(r"d(\d+)_cs([0-9]+(?:p[0-9]+)?)", name)
    if not match:
        return ("?", "?")
    return (match.group(1), token_to_float_str(match.group(2)))


def parse_alpha_folder(name: str) -> str:
    """
    Parse names like:
        alpha0p4
        alpha0p95
    Returns:
        alpha as a human-readable decimal string.
    """
    match = re.fullmatch(r"alpha([0-9]+(?:p[0-9]+)?)", name)
    if not match:
        return "?"
    return token_to_float_str(match.group(1))


def read_single_value(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def pick_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def get_acc_from_csv(
    csv_path: Path,
    split: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Return:
        (A_mean, A_std, B_mean, B_std)
    for metric='acc' and the requested split.
    """
    if not csv_path.exists():
        return (None, None, None, None)

    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") == split and row.get("metric") == "acc":
                return (
                    safe_float(row.get("A_mean", "")),
                    safe_float(row.get("A_std", "")),
                    safe_float(row.get("B_mean", "")),
                    safe_float(row.get("B_std", "")),
                )

    return (None, None, None, None)


def latex_escape_text(text: str) -> str:
    text = text.replace("\\", r"\textbackslash ")
    text = text.replace("_", r"\_")
    return text


def format_pm(
    mean: Optional[float],
    std: Optional[float],
    *,
    bold: bool = False,
    decimals: int = 4,
) -> str:
    if mean is None:
        return "--"

    if std is None:
        body = f"{mean:.{decimals}f}"
    else:
        body = f"{mean:.{decimals}f}\\pm{std:.{decimals}f}"

    if bold:
        return rf"$\bm{{{body}}}$"
    return rf"${body}$"


def numeric_or_inf(text: str) -> float:
    value = safe_float(text)
    return float(value) if value is not None else math.inf


def row_sort_key(row: RowRec) -> Tuple[float, float, float, int]:
    return (
        numeric_or_inf(row.d),
        numeric_or_inf(row.beta),
        numeric_or_inf(row.alpha),
        int(row.ntr),
    )


def csv_candidates(ntr_dir: Path, alpha_folder_name: str) -> List[Path]:
    return [
        ntr_dir / f"agg_exp_{alpha_folder_name}.csv",
        ntr_dir / f"agg_{alpha_folder_name}.csv",
    ]


def read_lr_value(path: Path) -> str:
    return read_single_value(path) if path.exists() else "--"


def build_row(
    *,
    d: str,
    beta: str,
    alpha: str,
    ntr: int,
    ntr_dir: Path,
    alpha_folder_name: str,
) -> Optional[RowRec]:
    csv_path = pick_existing(csv_candidates(ntr_dir, alpha_folder_name))
    if csv_path is None:
        return None

    lrA = read_lr_value(ntr_dir / "lrA_selected.txt")
    lrB = read_lr_value(ntr_dir / "lrB_selected.txt")

    trA_m, trA_s, trB_m, trB_s = get_acc_from_csv(csv_path, "train")
    teA_m, teA_s, teB_m, teB_s = get_acc_from_csv(csv_path, "test")

    return RowRec(
        d=d,
        beta=beta,
        alpha=alpha,
        ntr=ntr,
        lrA=lrA,
        lrB=lrB,
        trA_m=trA_m,
        trA_s=trA_s,
        trB_m=trB_m,
        trB_s=trB_s,
        teA_m=teA_m,
        teA_s=teA_s,
        teB_m=teB_m,
        teB_s=teB_s,
    )


def collect_rows(
    root: Path,
    dcs_list: List[str],
    alpha_list: List[str],
    ntr_order: List[int],
) -> List[RowRec]:
    rows: List[RowRec] = []

    for dcs_name in dcs_list:
        d_str, beta_str = parse_dcs_folder(dcs_name)
        dcs_dir = root / dcs_name
        if not dcs_dir.exists():
            continue

        for alpha_name in alpha_list:
            alpha_value = parse_alpha_folder(alpha_name)
            alpha_dir = dcs_dir / alpha_name
            if not alpha_dir.exists():
                continue

            for ntr in ntr_order:
                ntr_dir = alpha_dir / f"ntr{ntr}"
                row = build_row(
                    d=d_str,
                    beta=beta_str,
                    alpha=alpha_value,
                    ntr=ntr,
                    ntr_dir=ntr_dir,
                    alpha_folder_name=alpha_name,
                )
                if row is not None:
                    rows.append(row)

    return sorted(rows, key=row_sort_key)


def render_big_table(rows: List[RowRec], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(
        r"\caption{Each row corresponds to $(d,\beta,\alpha,N_{\mathrm{tr}})$; "
        r"we also report the selected learning rates and the training accuracies. "
        r"The best test accuracy between models A and B is shown in bold.}"
    )
    lines.append(r"\label{tab:all_test_acc}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lll r rr rr rr}")
    lines.append(r"\toprule")
    lines.append(
        r"$d$ & $\beta$ & $\alpha$ & $N_{\mathrm{tr}}$ & "
        r"$\mathrm{lr}_A$ & $\mathrm{lr}_B$ & "
        r"$\mathrm{Acc}_A^{\mathrm{tr}}$ & $\mathrm{Acc}_B^{\mathrm{tr}}$ & "
        r"$\mathrm{Acc}_A^{\mathrm{te}}$ & $\mathrm{Acc}_B^{\mathrm{te}}$ \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        lrA = latex_escape_text(row.lrA)
        lrB = latex_escape_text(row.lrB)

        trA = format_pm(row.trA_m, row.trA_s, bold=False)
        trB = format_pm(row.trB_m, row.trB_s, bold=False)

        eps = 1e-12
        finite_test = {
            "A": row.teA_m,
            "B": row.teB_m,
        }
        finite_test = {name: value for name, value in finite_test.items() if value is not None}

        bold_A = False
        bold_B = False
        if finite_test:
            best_value = max(finite_test.values())
            bold_A = row.teA_m is not None and abs(float(row.teA_m) - float(best_value)) <= eps
            bold_B = row.teB_m is not None and abs(float(row.teB_m) - float(best_value)) <= eps

        teA = format_pm(row.teA_m, row.teA_s, bold=bold_A)
        teB = format_pm(row.teB_m, row.teB_s, bold=bold_B)

        lines.append(
            rf"{row.d} & {row.beta} & {row.alpha} & {row.ntr} & "
            rf"{lrA} & {lrB} & "
            rf"{trA} & {trB} & {teA} & {teB} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("./out/agg_gapwide_stair"))
    parser.add_argument("--outdir", type=Path, default=Path("./out/big_tables"))
    parser.add_argument("--dcs", type=str, default="d30_cs1p5,d80_cs0p9,d150_cs0p6")
    parser.add_argument("--alphas", type=str, default="alpha0p4,alpha0p6,alpha0p8,alpha0p95")
    parser.add_argument("--ntr-order", type=str, default="200,500,1000")
    return parser.parse_args()


def parse_csv_arg(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int_csv_arg(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    args = parse_args()

    dcs_list = parse_csv_arg(args.dcs)
    alpha_list = parse_csv_arg(args.alphas)
    ntr_order = parse_int_csv_arg(args.ntr_order)

    rows = collect_rows(args.root, dcs_list, alpha_list, ntr_order)

    out_path = args.outdir / OUTPUT_FILENAME
    render_big_table(rows, out_path)

    print(f"Wrote: {out_path}  (rows={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())