from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from common.io_utils import save_json
from common.sampling import resolve_subset_size, stratified_subset_by_label
from topk.config_utils import build_topk_config
from topk.data import TopKConfig
from topk.data_splits import make_train_val_from_seed
from topk.train import train_topk_model


def _normalize_hp_mode(mode: str) -> str:
    value = str(mode).strip().upper()
    if value not in {"A", "B", "BOTH"}:
        raise ValueError("hp_mode must be 'A', 'B', or 'both'.")
    return value


def _target_kind_from_alias(alias: str) -> str:
    alias_u = str(alias).strip().upper()
    if alias_u == "A":
        return "projection"
    if alias_u == "B":
        return "fixed_target"
    raise ValueError("alias must be 'A' or 'B'.")


def _topk_lr_candidates(records: List[Dict[str, Any]], k: int) -> List[float]:
    if k <= 0 or not records:
        return []

    criterion = str(records[0].get("criterion", "acc")).lower()
    higher_is_better = criterion != "nll"

    ranked = sorted(
        records,
        key=lambda row: (
            -float(row["val_mean"]) if higher_is_better else float(row["val_mean"]),
            float(row["lr"]),
        ),
    )

    output: List[float] = []
    for row in ranked:
        lr = float(row["lr"])
        if lr not in output:
            output.append(lr)
        if len(output) >= int(k):
            break
    return output


def select_lr_by_validation_seeds(
    *,
    mode: str,
    lr_grid: Sequence[float],
    val_seeds: Sequence[int],
    cfg_base: TopKConfig,
    N_train: int,
    N_val: int,
    seed_init_base: int,
    criterion: str = "acc",
    epochs: int = 30,
    batch_size: int = 128,
    weight_decay: float = 1e-3,
    proj_tau_train: float = 1e-10,
    proj_K_train: int = 200,
    log_clip_eps: float = 1e-15,
    proj_engine: str = "cpp_batch",
    proj_n_threads: int = 0,
    train_subset_frac: float = 1.0,
    train_subset_size: int = 0,
    train_subset_seed: int = 0,
    no_progress: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    mode_alias = str(mode).strip().upper()
    if mode_alias not in {"A", "B"}:
        raise ValueError("mode must be 'A' or 'B'.")

    criterion_name = str(criterion).lower()
    if criterion_name not in {"acc", "nll"}:
        raise ValueError("criterion must be 'acc' or 'nll'.")

    subset_size = resolve_subset_size(total_size=int(N_train), explicit_size=int(train_subset_size), frac=float(train_subset_frac))

    records: List[Dict[str, Any]] = []
    lr_candidates = [float(lr) for lr in lr_grid]
    lr_iterator = tqdm(lr_candidates, desc=f"LR grid (mode {mode_alias})", disable=bool(no_progress))

    for lr in lr_iterator:
        values: List[float] = []
        train_size_used = subset_size
        val_size_used = int(N_val)

        for seed_data in tqdm(val_seeds, desc="val seeds", leave=False, disable=bool(no_progress)):
            train_samples, val_samples = make_train_val_from_seed(cfg=cfg_base, N_train=int(N_train), N_val=int(N_val), seed_data=int(seed_data))
            if subset_size < len(train_samples):
                train_samples = stratified_subset_by_label(train_samples, subset_size=int(subset_size), seed=int(train_subset_seed), label_attr="y")

            train_size_used = int(len(train_samples))
            val_size_used = int(len(val_samples))
            seed_init = int(seed_init_base) + int(seed_data)

            _, run_log = train_topk_model(
                samples_train=train_samples,
                samples_test=val_samples,
                target_kind=_target_kind_from_alias(mode_alias),
                lr=float(lr),
                epochs=int(epochs),
                batch_size=int(batch_size),
                weight_decay=float(weight_decay),
                proj_tau=float(proj_tau_train),
                proj_Kmax=int(proj_K_train),
                log_clip_eps=float(log_clip_eps),
                proj_engine=str(proj_engine),
                proj_n_threads=int(proj_n_threads),
                seed=int(seed_init),
                head="linear",
                no_progress=True,
            )
            values.append(float(run_log["test_metrics"][criterion_name]))

        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        records.append({
            "lr": float(lr),
            "val_mean": mean_value,
            "val_std": std_value,
            "criterion": criterion_name,
            "train_size": int(train_size_used),
            "val_size": int(val_size_used),
        })

        print(
            f"  [HP mode={mode_alias}] lr={lr:.1e}  val_{criterion_name}={mean_value:.4f} +/- {std_value:.4f}  train_n={train_size_used}  val_n={val_size_used}",
            flush=True,
        )

        best_so_far = min(records, key=lambda row: row["val_mean"]) if criterion_name == "nll" else max(records, key=lambda row: row["val_mean"])
        lr_iterator.set_postfix(best_lr=best_so_far["lr"], best=best_so_far["val_mean"])

    best_record = min(records, key=lambda row: row["val_mean"]) if criterion_name == "nll" else max(records, key=lambda row: row["val_mean"])
    return float(best_record["lr"]), records


def run_hp_search(
    *,
    n_classes: int,
    d: int,
    alpha_list: Sequence[float],
    N_train: int,
    N_val: int,
    val_seeds: Sequence[int],
    lr_grid: Sequence[float],
    seed_init_base_A: int = 1000,
    seed_init_base_B: int = 2000,
    criterion: str,
    pi_eps: float,
    alpha_noise: float,
    class_sep: float,
    x_noise: float,
    tie_tol: float,
    eps_cap: float,
    pi_stair_step: float,
    pi_stair_m: int,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    proj_tau_train: float,
    proj_K_train: int,
    log_clip_eps: float,
    proj_engine: str = "cpp_batch",
    proj_n_threads: int = 0,
    hp_mode: str = "both",
    hp_train_subset_frac_A: float = 1.0,
    hp_train_subset_size_A: int = 0,
    hp_train_subset_seed_A: int = 0,
    hp_confirm_topk_A: int = 0,
    out: str = "",
    no_progress: bool = False,
) -> Dict[str, Any]:
    criterion_name = str(criterion).lower()
    if criterion_name not in {"acc", "nll"}:
        raise ValueError("criterion must be 'acc' or 'nll'.")

    hp_mode_name = _normalize_hp_mode(hp_mode)
    lr_grid_values = [float(lr) for lr in lr_grid]
    if not lr_grid_values:
        raise ValueError("lr_grid must be non-empty.")

    log: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cmd": "hp-search",
        "hp_mode": hp_mode_name,
        "criterion": criterion_name,
        "model_roles": {"A": "projection", "B": "fixed_target"},
        "val_seeds": [int(seed) for seed in val_seeds],
        "lr_grid": lr_grid_values,
        "hyperparams": {
            "n_classes": int(n_classes), "d": int(d), "k": int(n_classes), "N_train": int(N_train), "N_val": int(N_val),
            "optimizer": "adam", "batch_size": int(batch_size), "epochs": int(epochs), "weight_decay": float(weight_decay),
            "proj_tau_train": float(proj_tau_train), "proj_K_train": int(proj_K_train), "log_clip_eps": float(log_clip_eps),
            "pi_eps": float(pi_eps), "alpha_noise": float(alpha_noise), "class_sep": float(class_sep), "x_noise": float(x_noise),
            "tie_tol": float(tie_tol), "eps_cap": float(eps_cap), "pi_shape": "stair", "pi_stair_step": float(pi_stair_step),
            "pi_stair_m": int(pi_stair_m), "proj_engine": str(proj_engine), "proj_n_threads": int(proj_n_threads),
            "device": "cpu", "dtype": "float64", "torch_threads": 1,
            "hp_train_subset_frac_A": float(hp_train_subset_frac_A), "hp_train_subset_size_A": int(hp_train_subset_size_A),
            "hp_train_subset_seed_A": int(hp_train_subset_seed_A), "hp_confirm_topk_A": int(hp_confirm_topk_A),
        },
        "results": [],
    }

    for alpha in tqdm([float(a) for a in alpha_list], desc="hp-search alpha sweep", disable=bool(no_progress)):
        cfg = build_topk_config(
            n_classes=int(n_classes), d=int(d), alpha=float(alpha), pi_eps=float(pi_eps), alpha_noise=float(alpha_noise),
            class_sep=float(class_sep), x_noise=float(x_noise), tie_tol=float(tie_tol), eps_cap=float(eps_cap),
            pi_stair_step=float(pi_stair_step), pi_stair_m=int(pi_stair_m),
        )

        shared_search_kwargs = dict(
            val_seeds=val_seeds, cfg_base=cfg, N_train=int(N_train), N_val=int(N_val), criterion=criterion_name,
            epochs=int(epochs), batch_size=int(batch_size), weight_decay=float(weight_decay), proj_tau_train=float(proj_tau_train),
            proj_K_train=int(proj_K_train), log_clip_eps=float(log_clip_eps), proj_engine=str(proj_engine), proj_n_threads=int(proj_n_threads),
            no_progress=bool(no_progress),
        )

        best_lr_A_proxy = None
        table_A_proxy: List[Dict[str, Any]] = []
        best_lr_A = None
        table_A: List[Dict[str, Any]] = []
        table_A_confirm: List[Dict[str, Any]] = []
        confirm_lrs_A: List[float] = []

        proxy_subset_size_A = resolve_subset_size(total_size=int(N_train), explicit_size=int(hp_train_subset_size_A), frac=float(hp_train_subset_frac_A))
        proxy_used_A = proxy_subset_size_A < int(N_train)

        if hp_mode_name in {"A", "BOTH"}:
            best_lr_A_proxy, table_A_proxy = select_lr_by_validation_seeds(
                mode="A", lr_grid=lr_grid_values, seed_init_base=int(seed_init_base_A),
                train_subset_frac=float(hp_train_subset_frac_A), train_subset_size=int(hp_train_subset_size_A), train_subset_seed=int(hp_train_subset_seed_A),
                **shared_search_kwargs,
            )
            best_lr_A = best_lr_A_proxy
            table_A = list(table_A_proxy)

            if proxy_used_A and int(hp_confirm_topk_A) > 0 and table_A_proxy:
                confirm_lrs_A = _topk_lr_candidates(table_A_proxy, int(hp_confirm_topk_A))
                best_lr_A, table_A_confirm = select_lr_by_validation_seeds(
                    mode="A", lr_grid=confirm_lrs_A, seed_init_base=int(seed_init_base_A),
                    train_subset_frac=1.0, train_subset_size=0, train_subset_seed=int(hp_train_subset_seed_A),
                    **shared_search_kwargs,
                )
                if table_A_confirm:
                    table_A = list(table_A_confirm)
        else:
            best_lr_A = None
            table_A = []

        if hp_mode_name in {"B", "BOTH"}:
            best_lr_B, table_B = select_lr_by_validation_seeds(mode="B", lr_grid=lr_grid_values, seed_init_base=int(seed_init_base_B), **shared_search_kwargs)
        else:
            best_lr_B, table_B = None, []

        log["results"].append({
            "alpha": float(alpha),
            "best_lr_A": float(best_lr_A) if best_lr_A is not None else None,
            "best_lr_B": float(best_lr_B) if best_lr_B is not None else None,
            "table_A": table_A,
            "table_B": table_B,
            "best_lr_A_proxy": float(best_lr_A_proxy) if best_lr_A_proxy is not None else None,
            "table_A_proxy": table_A_proxy,
            "table_A_confirm": table_A_confirm,
            "protocol_A": {
                "proxy_used": bool(proxy_used_A), "proxy_subset_size": int(proxy_subset_size_A), "full_train_size": int(N_train), "val_size": int(N_val),
                "proxy_subset_seed": int(hp_train_subset_seed_A), "proxy_subset_frac": float(hp_train_subset_frac_A),
                "proxy_subset_size_arg": int(hp_train_subset_size_A), "confirm_topk": int(hp_confirm_topk_A),
                "confirm_lrs": [float(lr) for lr in confirm_lrs_A],
            },
        })

    if out:
        out_path = Path(out)
        if out_path.suffix.lower() == ".json":
            save_json(str(out_path), log)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            save_json(str(out_path / "hp_search.json"), log)
    return log
