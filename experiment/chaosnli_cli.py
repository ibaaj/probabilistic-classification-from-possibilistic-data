#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from common.io_utils import save_json, to_jsonable
from common.sampling import resolve_subset_size
from nlpbench.chaosnli.slices import (
    SELECTION_SPLITS,
    VALIDATION_SECTION_ORDER,
    TEST_SECTION_ORDER,
    build_protocol_sections,
    compute_slice_stats_for_split,
    normalize_selection_split,
    test_section_for_selection_split,
)
from experiment.metrics import evaluate_metrics, topk_constraint_violations
from experiment.target_protocol import ProjectionStats
from experiment.targets import ProjectionTarget, target_B, target_C
from experiment.train import train_model
from nlpbench.chaosnli import add_chaosnli_data_args, load_chaosnli_splits
from nlpbench.sampling import stratified_subset_by_label
from topk.model import build_head


SEED_ARG_NAMES = {
    "seed_init_A",
    "seed_init_B",
    "seed_init_C",
    "seed_init_base_A",
    "seed_init_base_B",
    "seed_init_base_C",
    "hp_seeds",
    "train_subset_seed",
}

NON_HPARAM_ARG_NAMES = {
    "cmd",
    "fn",
    "no_progress",
    "out",
    "out_dir",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _save_run_artifacts(base_json_path: str, artifacts: Dict[str, Any]) -> None:
    base = Path(base_json_path)
    stem = base.with_suffix("")

    np.save(
        str(stem) + "_test_full_ids.npy",
        np.asarray(artifacts.get("sample_ids_test_full", []), dtype=object),
        allow_pickle=True,
    )
    np.save(
        str(stem) + "_test_full_y.npy",
        np.asarray(artifacts.get("y_test_full", []), dtype=np.int64),
    )
    for mode in ("A", "B", "C"):
        key = f"probs_{mode}_test_full"
        if key in artifacts and artifacts[key] is not None:
            np.save(
                str(stem) + f"_test_full_probs_{mode}.npy",
                np.asarray(artifacts[key], dtype=np.float64),
            )


def _configure_torch() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    np.seterr(over="raise", invalid="raise", divide="raise")


def _normalize_active_modes(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        mode = str(value).upper()
        if mode not in {"A", "B", "C"}:
            raise ValueError(f"Unsupported mode {value!r}.")
        if mode not in out:
            out.append(mode)
    if not out:
        raise ValueError("active modes cannot be empty.")
    return out


def _resolve_hp_subset_size_A(args: argparse.Namespace, total_train: int) -> int:
    return resolve_subset_size(
        total_size=int(total_train),
        explicit_size=int(getattr(args, "hp_train_subset_size_A", 0)),
        frac=float(getattr(args, "hp_train_subset_frac_A", 1.0)),
    )


def _topk_lr_candidates(records: List[Dict[str, Any]], k: int) -> List[float]:
    if k <= 0 or not records:
        return []
    ranked = sorted(records, key=lambda row: (-float(row["val_acc_mean"]), float(row["lr"])))
    out: List[float] = []
    for row in ranked:
        lr = float(row["lr"])
        if lr not in out:
            out.append(lr)
        if len(out) >= int(k):
            break
    return out


def _config_blocks_from_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seeds: Dict[str, Any] = {}
    hyperparams: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in NON_HPARAM_ARG_NAMES:
            continue
        if key in SEED_ARG_NAMES:
            seeds[key] = to_jsonable(value)
        else:
            hyperparams[key] = to_jsonable(value)
    return seeds, hyperparams


def _extract_data_provenance(data: Dict[str, Any]) -> Dict[str, Any]:
    skip_keys = {"train", "train_full", "val_full", "test_full", "C", "D"}
    return {key: to_jsonable(value) for key, value in data.items() if key not in skip_keys}


def _canonical_slice_thresholds_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Load the canonical slice thresholds from the loader payload."""
    if "slice_thresholds" in data and isinstance(data["slice_thresholds"], dict):
        return dict(data["slice_thresholds"])
    if "train_section_thresholds" in data and isinstance(data["train_section_thresholds"], dict):
        return dict(data["train_section_thresholds"])
    raise KeyError("ChaosNLI data payload is missing slice thresholds.")


def _make_model(args: argparse.Namespace, D: int, C: int, seed: int):
    return build_head(
        head=str(args.head),
        d=int(D),
        C=int(C),
        hidden_dim=int(args.mlp_hidden_dim),
        dropout=float(args.mlp_dropout),
        seed=int(seed),
    )


def _evaluate_split(items: List[Any], model: Any, tau_eval: float) -> Dict[str, Any]:
    if not items:
        return {}
    return evaluate_metrics(items, model, tau_eval=float(tau_eval), violation_fn=topk_constraint_violations)


def _target_fn_for_mode(args: argparse.Namespace, mode: str):
    mode = str(mode).upper()
    if mode == "A":
        return ProjectionTarget(
            tau=float(args.proj_tau),
            K_max=int(args.proj_kmax),
            log_clip_eps=float(args.log_clip_eps),
            engine=str(args.proj_engine),
            n_threads=int(args.proj_n_threads),
            pi_eps=float(getattr(args, "pi_eps", 1e-6)),
            tie_tol=float(getattr(args, "tie_tol", 0.0)),
            eps_cap=float(getattr(args, "eps_cap", 0.05)),
        )
    if mode == "B":
        return target_B
    if mode == "C":
        return target_C
    raise ValueError("mode must be one of 'A', 'B', 'C'.")


def _train_one(
    *,
    mode: str,
    train_items: List[Any],
    val_items: List[Any],
    D: int,
    C: int,
    lr: float,
    epochs: int,
    args: argparse.Namespace,
    seed: int,
):
    model = _make_model(args, D=D, C=C, seed=int(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(args.weight_decay))
    target_fn = _target_fn_for_mode(args, mode)
    proj_stats = train_model(
        samples_train=train_items,
        model=model,
        optimizer=optimizer,
        target_fn=target_fn,
        epochs=int(epochs),
        batch_size=int(args.batch_size),
        seed=int(seed),
        val_items=val_items,
        scheduler="cosine",
        no_progress=bool(args.no_progress),
    )
    proj_diag = None
    if mode == "A" and hasattr(target_fn, "flush_diagnostics"):
        proj_diag = target_fn.flush_diagnostics()

    return model, proj_stats, proj_diag


def _resolve_output_path(args: argparse.Namespace, cmd_name: str) -> str:
    out = str(getattr(args, "out", "") or "").strip()
    out_dir = str(getattr(args, "out_dir", "") or "").strip()
    if out:
        return out
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        filename = "hp-search.json" if cmd_name == "hp-search" else "run.json"
        return str(p / filename)
    return ""


def _slice_protocol_from_data(
    *,
    data: Dict[str, Any],
    selection_split: str,
) -> Dict[str, Any]:
    selection_name = normalize_selection_split(selection_split)
    n_classes = int(data["C"])
    val_full = list(data["val_full"])
    test_full = list(data["test_full"])

    thresholds = _canonical_slice_thresholds_from_data(data)

    missing_embedding_counts = data.get("missing_embedding_counts", {})
    missing_train_full = int(missing_embedding_counts.get("train_full", 0))
    if missing_train_full != 0:
        raise RuntimeError(
            "ChaosNLI protocol requires zero missing train_full embeddings "
            f"when reusing canonical slice thresholds, got train_full missing={missing_train_full}."
        )

    val_sections = build_protocol_sections(
        val_full,
        split_prefix="val",
        thresholds=thresholds,
        n_classes=n_classes,
    )
    test_sections = build_protocol_sections(
        test_full,
        split_prefix="test",
        thresholds=thresholds,
        n_classes=n_classes,
    )

    return {
        "selection_split": selection_name,
        "selection_test_split": test_section_for_selection_split(selection_name),
        "slice_thresholds": thresholds,
        "val_sections": val_sections,
        "test_sections": test_sections,
    }


def _section_sizes(val_sections: Dict[str, List[Any]], test_sections: Dict[str, List[Any]]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for section_name in VALIDATION_SECTION_ORDER:
        sizes[section_name] = int(len(val_sections[section_name]))
    for section_name in TEST_SECTION_ORDER:
        sizes[section_name] = int(len(test_sections[section_name]))
    return sizes


def _evaluate_sections_for_modes(
    *,
    models: Dict[str, Any],
    section_to_items: Dict[str, List[Any]],
    tau_eval: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for section_name, items in section_to_items.items():
        out[section_name] = {
            mode: ({} if mode not in models else _evaluate_split(items, models[mode], tau_eval=float(tau_eval)))
            for mode in ("A", "B", "C")
        }
    return out


def _hp_search_once(args: argparse.Namespace) -> Dict[str, Any]:
    active_modes = _normalize_active_modes(list(args.active_modes))
    selection_split = normalize_selection_split(str(args.selection_split))

    data = load_chaosnli_splits(args)
    train_items = list(data["train"])
    train_full = list(data["train_full"])
    D = int(data["D"])
    C = int(data["C"])

    protocol = _slice_protocol_from_data(data=data, selection_split=selection_split)
    val_sections = protocol["val_sections"]
    val_selection_items = list(val_sections[selection_split])

    def _search(
        mode: str,
        lr_grid: List[float],
        seed_init_base: int,
        train_items_override: List[Any] | None = None,
    ) -> Tuple[float | None, List[Dict[str, Any]]]:
        mode = str(mode).upper()
        if mode not in active_modes:
            return None, []

        fitted_train_items = train_items if train_items_override is None else train_items_override
        records: List[Dict[str, Any]] = []
        for lr in lr_grid:
            vals: List[float] = []
            for hp_seed in list(args.hp_seeds):
                seed_eff = int(seed_init_base) + int(hp_seed)
                model, _, _ = _train_one(
                    mode=mode,
                    train_items=fitted_train_items,
                    val_items=val_selection_items,
                    D=D,
                    C=C,
                    lr=float(lr),
                    epochs=int(args.hp_epochs),
                    args=args,
                    seed=seed_eff,
                )
                vm = _evaluate_split(val_selection_items, model, tau_eval=float(args.proj_tau))
                vals.append(float(vm["acc"]))

            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            records.append(
                {
                    "lr": float(lr),
                    "val_acc_mean": mean_v,
                    "val_acc_std": std_v,
                    "per_seed": [float(v) for v in vals],
                    "train_size": int(len(fitted_train_items)),
                    "selection_split": selection_split,
                    "selection_size": int(len(val_selection_items)),
                    "val_full_size": int(len(val_sections["val_full"])),
                    "val_S_amb_size": int(len(val_sections["val_S_amb"])),
                    "val_S_easy_size": int(len(val_sections["val_S_easy"])),
                }
            )
            print(
                f"[chaosnli][HP {mode}] lr={lr:.2e}  selection={selection_split}  "
                f"val_acc={mean_v:.4f} +/- {std_v:.4f}  "
                f"train_n={len(fitted_train_items)}  selection_n={len(val_selection_items)}",
                flush=True,
            )

        best = max(records, key=lambda r: r["val_acc_mean"])
        return float(best["lr"]), records

    best_lr_A_proxy = None
    table_A_proxy: List[Dict[str, Any]] = []
    best_lr_A = None
    table_A: List[Dict[str, Any]] = []
    table_A_confirm: List[Dict[str, Any]] = []
    confirm_lrs_A: List[float] = []
    train_items_A_proxy = list(train_items)
    proxy_used_A = False

    if "A" in active_modes:
        subset_size_A = _resolve_hp_subset_size_A(args, len(train_items))
        if subset_size_A < len(train_items):
            proxy_used_A = True
            train_items_A_proxy = stratified_subset_by_label(
                list(train_items),
                subset_size=int(subset_size_A),
                seed=int(args.hp_train_subset_seed_A),
                label_attr="y",
            )

        best_lr_A_proxy, table_A_proxy = _search(
            "A",
            list(args.lr_grid_A),
            int(args.seed_init_base_A),
            train_items_override=train_items_A_proxy,
        )
        best_lr_A = best_lr_A_proxy
        table_A = list(table_A_proxy)

        confirm_topk_A = int(args.hp_confirm_topk_A)
        if proxy_used_A and confirm_topk_A > 0 and table_A_proxy:
            confirm_lrs_A = _topk_lr_candidates(table_A_proxy, confirm_topk_A)
            best_lr_A, table_A_confirm = _search(
                "A",
                confirm_lrs_A,
                int(args.seed_init_base_A),
                train_items_override=list(train_items),
            )
            if table_A_confirm:
                table_A = list(table_A_confirm)

    best_lr_B, table_B = _search("B", list(args.lr_grid_B), int(args.seed_init_base_B))
    best_lr_C, table_C = _search("C", list(args.lr_grid_C), int(args.seed_init_base_C))

    sizes = {
        "train": int(len(train_items)),
        "train_full": int(len(train_full)),
    }
    sizes.update(_section_sizes(val_sections, protocol["test_sections"]))

    return {
        "active_modes": active_modes,
        "selection_split": selection_split,
        "selection_test_split": protocol["selection_test_split"],
        "slice_thresholds": protocol["slice_thresholds"],
        "best_lr_A": float(best_lr_A) if best_lr_A is not None else None,
        "best_lr_B": float(best_lr_B) if best_lr_B is not None else None,
        "best_lr_C": float(best_lr_C) if best_lr_C is not None else None,
        "table_A": table_A,
        "table_B": table_B,
        "table_C": table_C,
        "best_lr_A_proxy": float(best_lr_A_proxy) if best_lr_A_proxy is not None else None,
        "table_A_proxy": table_A_proxy,
        "table_A_confirm": table_A_confirm,
        "hp_protocol_A": {
            "proxy_used": bool(proxy_used_A),
            "proxy_subset_size": int(len(train_items_A_proxy)),
            "full_train_size": int(len(train_items)),
            "val_size": int(len(val_selection_items)),
            "proxy_subset_seed": int(args.hp_train_subset_seed_A),
            "proxy_subset_frac": float(args.hp_train_subset_frac_A),
            "proxy_subset_size_arg": int(args.hp_train_subset_size_A),
            "confirm_topk": int(args.hp_confirm_topk_A),
            "confirm_lrs": [float(lr) for lr in confirm_lrs_A],
        },
        "sizes": sizes,
        "requested_train_size": int(data.get("requested_train_size", 0)),
        "effective_train_size": int(data.get("effective_train_size", len(train_items))),
        "full_train_size_before_subsample": int(
            data.get("full_train_size_before_subsample", len(train_full))
        ),
        "data_provenance": _extract_data_provenance(data),
    }


def cmd_hp_search(args: argparse.Namespace) -> None:
    result = _hp_search_once(args=args)
    seeds, hyperparams = _config_blocks_from_args(args)
    out = {
        "timestamp": _timestamp(),
        "cmd": "hp-search",
        "dataset": "chaosnli",
        "seeds": seeds,
        "hyperparams": hyperparams,
        "results": [result],
    }

    out_path = _resolve_output_path(args, "hp-search")
    if out_path:
        save_json(str(out_path), out)
        print(f"wrote: {out_path}", flush=True)
    else:
        print(json.dumps(to_jsonable(out), indent=2, sort_keys=True))


def _run_once(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    active_modes = _normalize_active_modes(list(args.active_modes))
    selection_split = normalize_selection_split(str(args.selection_split))

    data = load_chaosnli_splits(args)
    train_items = list(data["train"])
    train_full = list(data["train_full"])
    D = int(data["D"])
    C = int(data["C"])

    protocol = _slice_protocol_from_data(data=data, selection_split=selection_split)
    val_sections = protocol["val_sections"]
    test_sections = protocol["test_sections"]
    val_selection_items = list(val_sections[selection_split])
    selection_test_split = protocol["selection_test_split"]

    models: Dict[str, Any] = {}
    proj_stats_A: ProjectionStats | None = None
    proj_diag_A = None

    if "A" in active_modes:
        model_A, proj_stats_A, proj_diag_A = _train_one(
            mode="A",
            train_items=train_items,
            val_items=val_selection_items,
            D=D,
            C=C,
            lr=float(args.lr_A),
            epochs=int(args.epochs),
            args=args,
            seed=int(args.seed_init_A),
        )
        models["A"] = model_A

    if "B" in active_modes:
        model_B, _, _ = _train_one(
            mode="B",
            train_items=train_items,
            val_items=val_selection_items,
            D=D,
            C=C,
            lr=float(args.lr_B),
            epochs=int(args.epochs),
            args=args,
            seed=int(args.seed_init_B),
        )
        models["B"] = model_B

    if "C" in active_modes:
        model_C, _, _ = _train_one(
            mode="C",
            train_items=train_items,
            val_items=val_selection_items,
            D=D,
            C=C,
            lr=float(args.lr_C),
            epochs=int(args.epochs),
            args=args,
            seed=int(args.seed_init_C),
        )
        models["C"] = model_C

    train_res = {mode: ({} if mode not in models else _evaluate_split(train_items, models[mode], tau_eval=float(args.proj_tau))) for mode in ("A", "B", "C")}
    val_section_res = _evaluate_sections_for_modes(
        models=models,
        section_to_items=val_sections,
        tau_eval=float(args.proj_tau),
    )
    test_section_res = _evaluate_sections_for_modes(
        models=models,
        section_to_items=test_sections,
        tau_eval=float(args.proj_tau),
    )

    test_full = list(test_sections["test_full"])
    probs_test_full: Dict[str, np.ndarray | None] = {}
    for mode in ("A", "B", "C"):
        if mode not in models:
            probs_test_full[mode] = None
            continue
        X = np.stack([np.asarray(sample.x, dtype=np.float64) for sample in test_full], axis=0)
        probs_test_full[mode] = np.asarray(models[mode].predict_proba(X), dtype=np.float64)

    sample_ids_test_full = [str(getattr(sample, "sample_id", i)) for i, sample in enumerate(test_full)]
    y_test_full = np.asarray([int(sample.y) for sample in test_full], dtype=np.int64)

    selected_test_metrics = test_section_res[selection_test_split]
    acc_A = selected_test_metrics["A"].get("acc", float("nan"))
    acc_B = selected_test_metrics["B"].get("acc", float("nan"))
    acc_C = selected_test_metrics["C"].get("acc", float("nan"))
    print(
        f"[chaosnli] selection={selection_split} selected_test={selection_test_split} "
        f"A={acc_A:.4f} B={acc_B:.4f} C={acc_C:.4f}",
        flush=True,
    )

    result: Dict[str, Any] = {
        "active_modes": active_modes,
        "selection_split": selection_split,
        "selection_test_split": selection_test_split,
        "slice_thresholds": protocol["slice_thresholds"],
        "lr_A": float(args.lr_A),
        "lr_B": float(args.lr_B),
        "lr_C": float(args.lr_C),
        "seed_A": int(args.seed_init_A),
        "seed_B": int(args.seed_init_B),
        "seed_C": int(args.seed_init_C),
        "train": train_res,
        "val_full": val_section_res["val_full"],
        "val_S_amb": val_section_res["val_S_amb"],
        "val_S_easy": val_section_res["val_S_easy"],
        "test": test_section_res["test_full"],
        "test_full": test_section_res["test_full"],
        "test_S_amb": test_section_res["test_S_amb"],
        "test_S_easy": test_section_res["test_S_easy"],
        "sizes": {
            "train": int(len(train_items)),
            "train_full": int(len(train_full)),
            **_section_sizes(val_sections, test_sections),
        },
        "requested_train_size": int(data.get("requested_train_size", 0)),
        "effective_train_size": int(data.get("effective_train_size", len(train_items))),
        "full_train_size_before_subsample": int(
            data.get("full_train_size_before_subsample", len(train_full))
        ),
        "train_subset_seed": int(
            data.get("train_subset_seed", getattr(args, "train_subset_seed", 42))
        ),
        "data_provenance": _extract_data_provenance(data),
    }

    if proj_stats_A is not None:
        result["projection_stats_train_A"] = asdict(proj_stats_A)

    if proj_diag_A is not None:
        result["projection_diagnostics_train_A"] = to_jsonable(proj_diag_A)

    artifacts = {
        "sample_ids_test_full": np.asarray(sample_ids_test_full, dtype=object),
        "y_test_full": y_test_full,
        "probs_A_test_full": probs_test_full["A"],
        "probs_B_test_full": probs_test_full["B"],
        "probs_C_test_full": probs_test_full["C"],
    }
    return result, artifacts


def cmd_run(args: argparse.Namespace) -> None:
    result, artifacts = _run_once(args=args)
    seeds, hyperparams = _config_blocks_from_args(args)
    out = {
        "timestamp": _timestamp(),
        "cmd": "run",
        "dataset": "chaosnli",
        "seeds": seeds,
        "hyperparams": hyperparams,
        "results": [result],
    }

    out_path = _resolve_output_path(args, "run")
    if out_path:
        save_json(str(out_path), out)
        _save_run_artifacts(str(out_path), artifacts)
        print(f"wrote: {out_path}", flush=True)
    else:
        print(json.dumps(to_jsonable(out), indent=2, sort_keys=True))


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("protocol")
    group.add_argument(
        "--selection-split",
        choices=list(SELECTION_SPLITS),
        default="val_full",
        help=(
            "Validation section used both for hyperparameter selection and "
            "checkpoint selection. The paired test section is recorded as "
            "selection_test_split in saved run logs."
        ),
    )


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--batch-size", type=int, default=256)
    group.add_argument("--weight-decay", type=float, default=1e-4)
    group.add_argument("--proj-tau", type=float, default=1e-6)
    group.add_argument("--proj-kmax", type=int, default=500)
    group.add_argument("--log-clip-eps", type=float, default=1e-15)
    group.add_argument("--head", choices=["linear", "mlp"], default="linear")
    group.add_argument("--mlp-hidden-dim", type=int, default=256)
    group.add_argument("--mlp-dropout", type=float, default=0.1)
    group.add_argument(
        "--proj-engine",
        choices=["cpp", "cpp_batch", "python", "auto"],
        default="cpp_batch",
    )
    group.add_argument("--proj-n-threads", type=int, default=0)
    group.add_argument("--active-modes", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    group.add_argument("--no-progress", action="store_true")


def _add_hp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hp-epochs", type=int, default=100)
    parser.add_argument("--hp-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr-grid-A", type=float, nargs="+", required=True)
    parser.add_argument("--lr-grid-B", type=float, nargs="+", required=True)
    parser.add_argument("--lr-grid-C", type=float, nargs="+", required=True)
    parser.add_argument("--seed-init-base-A", type=int, default=1000)
    parser.add_argument("--seed-init-base-B", type=int, default=2000)
    parser.add_argument("--seed-init-base-C", type=int, default=3000)
    parser.add_argument("--hp-train-subset-frac-A", type=float, default=1.0)
    parser.add_argument("--hp-train-subset-size-A", type=int, default=0)
    parser.add_argument("--hp-train-subset-seed-A", type=int, default=0)
    parser.add_argument("--hp-confirm-topk-A", type=int, default=3)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-A", type=float, required=True)
    parser.add_argument("--lr-B", type=float, required=True)
    parser.add_argument("--lr-C", type=float, required=True)
    parser.add_argument("--seed-init-A", type=int, default=1000)
    parser.add_argument("--seed-init-B", type=int, default=2000)
    parser.add_argument("--seed-init-C", type=int, default=3000)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChaosNLI-only CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hp = sub.add_parser("hp-search")
    add_chaosnli_data_args(hp)
    _add_protocol_args(hp)
    _add_common_train_args(hp)
    _add_hp_args(hp)
    hp.set_defaults(fn=cmd_hp_search)

    run = sub.add_parser("run")
    add_chaosnli_data_args(run)
    _add_protocol_args(run)
    _add_common_train_args(run)
    _add_run_args(run)
    run.set_defaults(fn=cmd_run)
    return parser


def main() -> None:
    _configure_torch()
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
