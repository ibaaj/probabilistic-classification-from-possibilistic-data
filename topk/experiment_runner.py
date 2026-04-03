from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from tqdm import tqdm

from common.io_utils import save_json
from topk.config_utils import build_topk_config
from topk.data_splits import make_train_test_from_seed
from topk.train import train_topk_model


def _run_single_alpha(
    *,
    alpha: float,
    n_classes: int,
    d: int,
    N_train: int,
    N_test: int,
    seed_data: int,
    seed_init_A: int,
    seed_init_B: int,
    lr_A: float,
    lr_B: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    pi_eps: float,
    alpha_noise: float,
    class_sep: float,
    x_noise: float,
    proj_tau_train: float,
    proj_K_train: int,
    log_clip_eps: float,
    tie_tol: float,
    eps_cap: float,
    pi_stair_step: float,
    pi_stair_m: int,
    proj_engine: str,
    proj_n_threads: int,
    no_progress: bool,
) -> Dict[str, Any]:
    cfg = build_topk_config(
        n_classes=n_classes,
        d=d,
        alpha=alpha,
        pi_eps=pi_eps,
        alpha_noise=alpha_noise,
        class_sep=class_sep,
        x_noise=x_noise,
        tie_tol=tie_tol,
        eps_cap=eps_cap,
        pi_stair_step=pi_stair_step,
        pi_stair_m=pi_stair_m,
    )

    train_samples, test_samples = make_train_test_from_seed(
        cfg=cfg,
        N_train=int(N_train),
        N_test=int(N_test),
        seed_data=int(seed_data),
    )

    projection_model, projection_log = train_topk_model(
        samples_train=train_samples,
        samples_test=test_samples,
        target_kind="projection",
        lr=float(lr_A),
        epochs=int(epochs),
        batch_size=int(batch_size),
        weight_decay=float(weight_decay),
        proj_tau=float(proj_tau_train),
        proj_Kmax=int(proj_K_train),
        log_clip_eps=float(log_clip_eps),
        seed=int(seed_init_A),
        head="linear",
        proj_engine=str(proj_engine),
        proj_n_threads=int(proj_n_threads),
        no_progress=bool(no_progress),
    )
    del projection_model

    fixed_model, fixed_log = train_topk_model(
        samples_train=train_samples,
        samples_test=test_samples,
        target_kind="fixed_target",
        lr=float(lr_B),
        epochs=int(epochs),
        batch_size=int(batch_size),
        weight_decay=float(weight_decay),
        proj_tau=float(proj_tau_train),
        proj_Kmax=int(proj_K_train),
        log_clip_eps=float(log_clip_eps),
        seed=int(seed_init_B),
        head="linear",
        proj_engine=str(proj_engine),
        proj_n_threads=int(proj_n_threads),
        no_progress=bool(no_progress),
    )
    del fixed_model

    return {
        "alpha": float(alpha),
        "model_roles": {"A": "projection", "B": "fixed_target"},
        "train": {"A": projection_log["train_metrics"], "B": fixed_log["train_metrics"]},
        "test": {"A": projection_log["test_metrics"], "B": fixed_log["test_metrics"]},
        "projection_stats_train_A": projection_log.get("projection_stats_train"),
    }


def run_topk_experiment(
    *,
    n_classes: int,
    d: int,
    alpha_list: Sequence[float],
    N_train: int,
    N_test: int,
    seed_data: int,
    seed_init_A: int,
    seed_init_B: int,
    lr_A: float,
    lr_B: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    pi_eps: float,
    alpha_noise: float,
    class_sep: float,
    x_noise: float,
    proj_tau_train: float,
    proj_K_train: int,
    log_clip_eps: float,
    tie_tol: float,
    eps_cap: float,
    pi_stair_step: float,
    pi_stair_m: int,
    proj_engine: str = "cpp_batch",
    proj_n_threads: int = 0,
    out: str = "",
    no_progress: bool = False,
) -> Dict[str, Any]:
    run_log: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cmd": "topk-exp",
        "model_roles": {"A": "projection", "B": "fixed_target"},
        "seeds": {
            "seed_data": int(seed_data),
            "seed_init_A": int(seed_init_A),
            "seed_init_B": int(seed_init_B),
        },
        "hyperparams": {
            "n_classes": int(n_classes),
            "d": int(d),
            "k": int(n_classes),
            "N_train": int(N_train),
            "N_test": int(N_test),
            "optimizer": "adam",
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "weight_decay": float(weight_decay),
            "lr_A": float(lr_A),
            "lr_B": float(lr_B),
            "proj_tau_train": float(proj_tau_train),
            "proj_K_train": int(proj_K_train),
            "log_clip_eps": float(log_clip_eps),
            "pi_eps": float(pi_eps),
            "alpha_noise": float(alpha_noise),
            "class_sep": float(class_sep),
            "x_noise": float(x_noise),
            "tie_tol": float(tie_tol),
            "eps_cap": float(eps_cap),
            "pi_shape": "stair",
            "pi_stair_step": float(pi_stair_step),
            "pi_stair_m": int(pi_stair_m),
            "proj_engine": str(proj_engine),
            "proj_n_threads": int(proj_n_threads),
            "device": "cpu",
            "dtype": "float64",
            "torch_threads": 1,
        },
        "results": [],
    }

    for alpha in tqdm([float(a) for a in alpha_list], desc="alpha sweep", disable=bool(no_progress)):
        alpha_result = _run_single_alpha(
            alpha=float(alpha),
            n_classes=int(n_classes),
            d=int(d),
            N_train=int(N_train),
            N_test=int(N_test),
            seed_data=int(seed_data),
            seed_init_A=int(seed_init_A),
            seed_init_B=int(seed_init_B),
            lr_A=float(lr_A),
            lr_B=float(lr_B),
            epochs=int(epochs),
            batch_size=int(batch_size),
            weight_decay=float(weight_decay),
            pi_eps=float(pi_eps),
            alpha_noise=float(alpha_noise),
            class_sep=float(class_sep),
            x_noise=float(x_noise),
            proj_tau_train=float(proj_tau_train),
            proj_K_train=int(proj_K_train),
            log_clip_eps=float(log_clip_eps),
            tie_tol=float(tie_tol),
            eps_cap=float(eps_cap),
            pi_stair_step=float(pi_stair_step),
            pi_stair_m=int(pi_stair_m),
            proj_engine=str(proj_engine),
            proj_n_threads=int(proj_n_threads),
            no_progress=bool(no_progress),
        )
        run_log["results"].append(alpha_result)

    if out:
        out_path = Path(out)
        if out_path.suffix.lower() == ".json":
            save_json(str(out_path), run_log)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            filename = f"topk-exp_seeddata{seed_data}_initA{seed_init_A}_initB{seed_init_B}.json"
            save_json(str(out_path / filename), run_log)
    return run_log
