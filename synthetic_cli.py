#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from klbox.kl_types import LOG_EPS


def _configure_torch_single_thread() -> None:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def _require_cpp_backend() -> None:
    try:
        from klbox.dykstra_cpp import dykstra_kl_project_cpp as _mono  # noqa: F401
        from klbox.dykstra_cpp import dykstra_kl_project_cpp_batch as _batch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("synthetic_cli.py requires the C++ projection backend (klbox.dykstra_cpp).") from exc


def _add_numeric_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("numerical clipping")
    group.add_argument("--pi-eps", type=float, default=1e-6, help="Background π epsilon (>0).")
    group.add_argument("--log-clip-eps", type=float, default=LOG_EPS, help="Clipping epsilon used in logs/projections.")


def _add_gap_box_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("synthetic gap-box construction")
    group.add_argument("--tie-tol", type=float, default=0.0)
    group.add_argument("--eps-cap", type=float, default=1e-9, help="Cap for gap epsilon in GAP-WIDE.")


def _add_dataset_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dataset / task")
    group.add_argument("--n-classes", type=int, default=20)
    group.add_argument("--d", type=int, default=10)
    group.add_argument("--alpha", type=float, nargs="+", default=[0.2])
    group.add_argument("--train", type=int, default=2000)
    group.add_argument("--test", type=int, default=2000)
    group.add_argument("--alpha-noise", type=float, default=0.15)
    group.add_argument("--class-sep", type=float, default=2.0)
    group.add_argument("--x-noise", type=float, default=1.0)


def _add_pi_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("π (stair only)")
    group.add_argument("--pi-stair-step", type=float, default=1e-3)
    group.add_argument("--pi-stair-m", type=int, default=0, help="0 => grade all neighbors.")


def _add_train_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training (adam only, cpu/float64 fixed)")
    group.add_argument("--epochs", type=int, default=30)
    group.add_argument("--batch", type=int, default=128)
    group.add_argument("--weight-decay", type=float, default=1e-3)
    group.add_argument("--proj-tau-train", "--proj-tau", dest="proj_tau_train", type=float, default=1e-10)
    group.add_argument("--proj-K-train", "--proj-kmax", dest="proj_K_train", type=int, default=200)
    group.add_argument("--proj-engine", choices=["python", "cpp", "cpp_batch", "auto"], default="cpp_batch")
    group.add_argument("--proj-n-threads", type=int, default=0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sweep = sub.add_parser("dykstra-sweep")
    p_sweep.add_argument("--n", type=int, default=100)
    p_sweep.add_argument("--runs", type=int, default=100)
    p_sweep.add_argument("--seed", type=int, default=0)
    p_sweep.add_argument("--Kmax", type=int, nargs="+", default=[1000, 10000, 50000, 100000])
    p_sweep.add_argument("--tau", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-6, 1e-8])
    p_sweep.add_argument("--no-progress", action="store_true")
    _add_numeric_group(p_sweep)
    _add_gap_box_group(p_sweep)

    p_exp = sub.add_parser("topk-exp")
    _add_dataset_group(p_exp)
    _add_pi_group(p_exp)
    _add_train_group(p_exp)
    _add_numeric_group(p_exp)
    _add_gap_box_group(p_exp)
    p_exp.add_argument("--seed-data", type=int, default=0)
    p_exp.add_argument("--seed-init-A", type=int, default=1000)
    p_exp.add_argument("--seed-init-B", type=int, default=2000)
    p_exp.add_argument("--lr-A", type=float, required=True)
    p_exp.add_argument("--lr-B", type=float, required=True)
    p_exp.add_argument("--out", type=str, default="")
    p_exp.add_argument("--no-progress", action="store_true")

    p_hp = sub.add_parser("hp-search")
    _add_dataset_group(p_hp)
    _add_pi_group(p_hp)
    _add_train_group(p_hp)
    _add_numeric_group(p_hp)
    _add_gap_box_group(p_hp)
    p_hp.add_argument("--val", type=int, default=2000)
    p_hp.add_argument("--criterion", type=str, default="acc", choices=["nll", "acc"])
    p_hp.add_argument("--lr-grid", type=float, nargs="+", required=True)
    p_hp.add_argument("--val-seeds", type=int, nargs="+", default=[0, 1, 2])
    p_hp.add_argument("--seed-init-base-A", type=int, default=1000)
    p_hp.add_argument("--seed-init-base-B", type=int, default=2000)
    p_hp.add_argument("--hp-train-subset-frac-A", type=float, default=1.0)
    p_hp.add_argument("--hp-train-subset-size-A", type=int, default=0)
    p_hp.add_argument("--hp-train-subset-seed-A", type=int, default=0)
    p_hp.add_argument("--hp-confirm-topk-A", type=int, default=0)
    p_hp.add_argument("--hp-mode", type=str, default="both", choices=["A", "B", "both"])
    p_hp.add_argument("--out", type=str, default="")
    p_hp.add_argument("--no-progress", action="store_true")
    return parser


def _handle_dykstra_sweep(args: argparse.Namespace) -> None:
    from klbox.protocol import latex_table as latex_table_dykstra
    from klbox.protocol import run_sweep

    for k_max in args.Kmax:
        rows = run_sweep(
            n=int(args.n), runs=int(args.runs), taus=list(args.tau), K_max=int(k_max), seed=int(args.seed),
            tie_tol=float(args.tie_tol), eps_cap=float(args.eps_cap), pi_eps=float(args.pi_eps), log_clip_eps=float(args.log_clip_eps),
            no_progress=bool(args.no_progress),
        )
        print(latex_table_dykstra(rows, n=int(args.n), runs=int(args.runs), K_max=int(k_max), label=f"tab:dykstra-K{int(k_max)}"))
        print()


def _handle_topk_exp(args: argparse.Namespace) -> None:
    from topk.experiments import run_topk_experiment
    run_topk_experiment(
        n_classes=int(args.n_classes), d=int(args.d), alpha_list=list(args.alpha), N_train=int(args.train), N_test=int(args.test),
        seed_data=int(args.seed_data), seed_init_A=int(args.seed_init_A), seed_init_B=int(args.seed_init_B),
        lr_A=float(args.lr_A), lr_B=float(args.lr_B), epochs=int(args.epochs), batch_size=int(args.batch),
        weight_decay=float(args.weight_decay), pi_eps=float(args.pi_eps), alpha_noise=float(args.alpha_noise), class_sep=float(args.class_sep),
        x_noise=float(args.x_noise), proj_tau_train=float(args.proj_tau_train), proj_K_train=int(args.proj_K_train),
        log_clip_eps=float(args.log_clip_eps), tie_tol=float(args.tie_tol), eps_cap=float(args.eps_cap),
        pi_stair_step=float(args.pi_stair_step), pi_stair_m=int(args.pi_stair_m), proj_engine=str(args.proj_engine), proj_n_threads=int(args.proj_n_threads),
        out=str(args.out), no_progress=bool(args.no_progress),
    )


def _handle_hp_search(args: argparse.Namespace) -> None:
    from topk.experiments import run_hp_search
    run_hp_search(
        n_classes=int(args.n_classes), d=int(args.d), alpha_list=list(args.alpha), N_train=int(args.train), N_val=int(args.val),
        val_seeds=list(args.val_seeds), lr_grid=list(args.lr_grid), seed_init_base_A=int(args.seed_init_base_A), seed_init_base_B=int(args.seed_init_base_B),
        criterion=str(args.criterion), pi_eps=float(args.pi_eps), alpha_noise=float(args.alpha_noise), class_sep=float(args.class_sep), x_noise=float(args.x_noise),
        tie_tol=float(args.tie_tol), eps_cap=float(args.eps_cap), pi_stair_step=float(args.pi_stair_step), pi_stair_m=int(args.pi_stair_m),
        epochs=int(args.epochs), batch_size=int(args.batch), weight_decay=float(args.weight_decay), proj_tau_train=float(args.proj_tau_train), proj_K_train=int(args.proj_K_train),
        log_clip_eps=float(args.log_clip_eps), proj_engine=str(args.proj_engine), proj_n_threads=int(args.proj_n_threads), hp_mode=str(args.hp_mode),
        hp_train_subset_frac_A=float(args.hp_train_subset_frac_A), hp_train_subset_size_A=int(args.hp_train_subset_size_A), hp_train_subset_seed_A=int(args.hp_train_subset_seed_A),
        hp_confirm_topk_A=int(args.hp_confirm_topk_A), out=str(args.out), no_progress=bool(args.no_progress),
    )


def main() -> None:
    _configure_torch_single_thread()
    _require_cpp_backend()
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "dykstra-sweep":
        _handle_dykstra_sweep(args)
        return
    if args.cmd == "topk-exp":
        _handle_topk_exp(args)
        return
    if args.cmd == "hp-search":
        _handle_hp_search(args)
        return
    raise RuntimeError(f"Unknown subcommand: {args.cmd!r}")


if __name__ == "__main__":
    np.seterr(over="raise", invalid="raise", divide="raise")
    main()
