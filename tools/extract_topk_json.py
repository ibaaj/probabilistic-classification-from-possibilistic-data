#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import load_json


def is_finite_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def get_results_list(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = doc.get("results", None)
    if not isinstance(results, list) or not results:
        raise SystemExit("JSON has no non-empty 'results' list.")

    out = [item for item in results if isinstance(item, dict)]
    if not out:
        raise SystemExit("JSON 'results' exists, but contains no object records.")
    return out


def select_result_block(
    doc: Dict[str, Any],
    *,
    alpha: Optional[float],
    alpha_tol: float,
) -> Dict[str, Any]:
    results = get_results_list(doc)

    if alpha is None:
        if len(results) == 1:
            return results[0]
        raise SystemExit(
            "Multiple results[*] entries found; please provide --alpha to select one."
        )

    alpha_target = float(alpha)
    tol = float(alpha_tol)

    for rec in results:
        value = rec.get("alpha", None)
        if is_finite_number(value) and abs(float(value) - alpha_target) <= tol:
            return rec

    raise SystemExit(
        f"No results[*] entry matched alpha={alpha_target} within tol={tol}."
    )


def get_path(doc: Any, path: List[Any]) -> Any:
    cur = doc
    for key in path:
        if isinstance(cur, dict) and isinstance(key, str):
            cur = cur.get(key, None)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return None
    return cur


def require_finite_number(value: Any, name: str) -> float:
    if not is_finite_number(value):
        raise SystemExit(f"Missing or non-finite value for '{name}'.")
    return float(value)


def read_result_record(
    json_path: str,
    *,
    alpha: Optional[float],
    alpha_tol: float,
) -> Dict[str, Any]:
    doc = load_json(Path(json_path))
    return select_result_block(doc, alpha=alpha, alpha_tol=alpha_tol)


def cmd_acc_ab(args: argparse.Namespace) -> None:
    rec = read_result_record(
        args.json,
        alpha=args.alpha,
        alpha_tol=float(args.alpha_tol),
    )

    acc_A = require_finite_number(
        get_path(rec, ["test", "A", "acc"]),
        "results[*].test.A.acc",
    )
    acc_B = require_finite_number(
        get_path(rec, ["test", "B", "acc"]),
        "results[*].test.B.acc",
    )

    print(f"{acc_A:.6f} {acc_B:.6f}")


def cmd_acc_abc(args: argparse.Namespace) -> None:
    rec = read_result_record(
        args.json,
        alpha=args.alpha,
        alpha_tol=float(args.alpha_tol),
    )

    acc_A = require_finite_number(
        get_path(rec, ["test", "A", "acc"]),
        "results[*].test.A.acc",
    )
    acc_B = require_finite_number(
        get_path(rec, ["test", "B", "acc"]),
        "results[*].test.B.acc",
    )
    acc_C = require_finite_number(
        get_path(rec, ["test", "C", "acc"]),
        "results[*].test.C.acc",
    )

    print(f"{acc_A:.6f} {acc_B:.6f} {acc_C:.6f}")


def cmd_best_lr_A(args: argparse.Namespace) -> None:
    rec = read_result_record(
        args.json,
        alpha=args.alpha,
        alpha_tol=float(args.alpha_tol),
    )
    best_lr_A = require_finite_number(rec.get("best_lr_A", None), "results[*].best_lr_A")
    print(f"{best_lr_A:.12g}")


def cmd_best_lr_B(args: argparse.Namespace) -> None:
    rec = read_result_record(
        args.json,
        alpha=args.alpha,
        alpha_tol=float(args.alpha_tol),
    )
    best_lr_B = require_finite_number(rec.get("best_lr_B", None), "results[*].best_lr_B")
    print(f"{best_lr_B:.12g}")


def cmd_best_lrs_abc(args: argparse.Namespace) -> None:
    rec = read_result_record(
        args.json,
        alpha=args.alpha,
        alpha_tol=float(args.alpha_tol),
    )

    best_lr_A = require_finite_number(rec.get("best_lr_A", None), "results[*].best_lr_A")
    best_lr_B = require_finite_number(rec.get("best_lr_B", None), "results[*].best_lr_B")
    best_lr_C = require_finite_number(rec.get("best_lr_C", None), "results[*].best_lr_C")

    print(f"{best_lr_A:.12g} {best_lr_B:.12g} {best_lr_C:.12g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract selected values from synthetic experiment JSON logs."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_acc_ab = sub.add_parser(
        "acc-ab",
        help="Print test accuracies: acc_A acc_B from a topk-exp JSON.",
    )
    p_acc_ab.add_argument("--json", required=True)
    p_acc_ab.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="If provided, select results[*] by alpha.",
    )
    p_acc_ab.add_argument("--alpha-tol", type=float, default=1e-12)
    p_acc_ab.set_defaults(fn=cmd_acc_ab)

    p_acc_abc = sub.add_parser(
        "acc-abc",
        help="Print test accuracies: acc_A acc_B acc_C from an experiment JSON.",
    )
    p_acc_abc.add_argument("--json", required=True)
    p_acc_abc.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="If provided, select results[*] by alpha.",
    )
    p_acc_abc.add_argument("--alpha-tol", type=float, default=1e-12)
    p_acc_abc.set_defaults(fn=cmd_acc_abc)

    p_best_A = sub.add_parser(
        "best-lr-A",
        help="Print best_lr_A from an hp-search JSON.",
    )
    p_best_A.add_argument("--json", required=True)
    p_best_A.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="If provided, select results[*] by alpha. Required if the log contains multiple alphas.",
    )
    p_best_A.add_argument("--alpha-tol", type=float, default=1e-12)
    p_best_A.set_defaults(fn=cmd_best_lr_A)

    p_best_B = sub.add_parser(
        "best-lr-B",
        help="Print best_lr_B from an hp-search JSON.",
    )
    p_best_B.add_argument("--json", required=True)
    p_best_B.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="If provided, select results[*] by alpha. Required if the log contains multiple alphas.",
    )
    p_best_B.add_argument("--alpha-tol", type=float, default=1e-12)
    p_best_B.set_defaults(fn=cmd_best_lr_B)

    p_best_abc = sub.add_parser(
        "best-lrs-abc",
        help="Print best_lr_A best_lr_B best_lr_C from a general experiment hp-search JSON.",
    )
    p_best_abc.add_argument("--json", required=True)
    p_best_abc.add_argument("--alpha", type=float, required=True)
    p_best_abc.add_argument("--alpha-tol", type=float, default=1e-12)
    p_best_abc.set_defaults(fn=cmd_best_lrs_abc)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()