from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from klbox.constraints import build_constraint_family
from klbox.dykstra import dykstra_kl_project
from klbox.dykstra_cpp import dykstra_kl_project_cpp, dykstra_kl_project_cpp_batch
from klbox.gaps import choose_gap_parameters
from klbox.linear_system import build_linear_system
from klbox.possibility import antipignistic_reverse_mapping, compute_possibility_order
from klbox.protocol import sample_pi, sample_q


def build_case(n: int, seed: int, tie_tol: float, eps_cap: float, pi_eps: float, log_clip_eps: float, *, include_prefix_constraints: bool = True):
    rng = np.random.default_rng(seed)
    pi = sample_pi(n, rng, eps=pi_eps)
    q = sample_q(n, rng, eps=log_clip_eps)
    order = compute_possibility_order(pi)
    anti = antipignistic_reverse_mapping(order)
    gaps = choose_gap_parameters(tilde_pi=order.tilde_pi, dot_g=anti.dot_g, tie_tol=float(tie_tol), eps_cap=float(eps_cap))
    system = build_linear_system(order, gaps, include_prefix_constraints=bool(include_prefix_constraints))
    constraints = build_constraint_family(order, gaps, include_prefix_constraints=bool(include_prefix_constraints))
    return q, order, gaps, system, constraints


def run_python(q, order, gaps, system, constraints, tau: float, K_max: int, log_clip_eps: float):
    del order, gaps
    return dykstra_kl_project(q=q, constraints=constraints, system=system, tau=float(tau), K_max=int(K_max), log_clip_eps=float(log_clip_eps))


def run_cpp(q, order, gaps, tau: float, K_max: int, log_clip_eps: float, *, include_prefix_constraints: bool = True):
    return dykstra_kl_project_cpp(q=q, order=order, gaps=gaps, tau=float(tau), K_max=int(K_max), log_clip_eps=float(log_clip_eps), include_prefix_constraints=bool(include_prefix_constraints))


def run_cpp_batch(cases, tau: float, K_max: int, log_clip_eps: float, *, include_prefix_constraints: bool = True, n_threads: int = 0):
    q_batch = np.stack([case[0] for case in cases], axis=0)
    orders = [case[1] for case in cases]
    gaps_list = [case[2] for case in cases]
    return dykstra_kl_project_cpp_batch(
        q_batch=q_batch,
        orders=orders,
        gaps_list=gaps_list,
        tau=float(tau),
        K_max=int(K_max),
        log_clip_eps=float(log_clip_eps),
        n_threads=int(n_threads),
        include_prefix_constraints=bool(include_prefix_constraints),
    )


def timed_run(fn: Callable[[], object]) -> tuple[object, float]:
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return result, t1 - t0


def summarize(name: str, timings: list[float]) -> str:
    return f"{name}: mean={statistics.mean(timings):.6f}s, median={statistics.median(timings):.6f}s, min={min(timings):.6f}s, max={max(timings):.6f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare klbox Python Dykstra vs C++ mono vs C++ batch.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tau", type=float, default=1e-10)
    parser.add_argument("--k-max", type=int, default=300)
    parser.add_argument("--tie-tol", type=float, default=1e-12)
    parser.add_argument("--eps-cap", type=float, default=1e-3)
    parser.add_argument("--pi-eps", type=float, default=1e-6)
    parser.add_argument("--log-clip-eps", type=float, default=1e-15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-threads", type=int, default=0)
    parser.add_argument("--no-prefix-constraints", action="store_true")
    args = parser.parse_args()

    include_prefix_constraints = not bool(args.no_prefix_constraints)
    one_case = build_case(args.n, args.seed, args.tie_tol, args.eps_cap, args.pi_eps, args.log_clip_eps, include_prefix_constraints=include_prefix_constraints)
    python_result = run_python(*one_case, tau=args.tau, K_max=args.k_max, log_clip_eps=args.log_clip_eps)
    cpp_result = run_cpp(one_case[0], one_case[1], one_case[2], tau=args.tau, K_max=args.k_max, log_clip_eps=args.log_clip_eps, include_prefix_constraints=include_prefix_constraints)
    batch_result = run_cpp_batch([one_case], tau=args.tau, K_max=args.k_max, log_clip_eps=args.log_clip_eps, include_prefix_constraints=include_prefix_constraints, n_threads=args.batch_threads)[0]

    print("Correctness check")
    print(f"  max |p_python - p_cpp|       = {float(np.max(np.abs(python_result.p_star - cpp_result.p_star))):.3e}")
    print(f"  max |p_python - p_cpp_batch| = {float(np.max(np.abs(python_result.p_star - batch_result.p_star))):.3e}")
    print(f"  |V_python - V_cpp|           = {abs(float(python_result.final_V) - float(cpp_result.final_V)):.3e}")
    print(f"  |V_python - V_cpp_batch|     = {abs(float(python_result.final_V) - float(batch_result.final_V)):.3e}")

    python_times: list[float] = []
    cpp_times: list[float] = []
    cpp_batch_times: list[float] = []
    cases = [build_case(args.n, args.seed + i, args.tie_tol, args.eps_cap, args.pi_eps, args.log_clip_eps, include_prefix_constraints=include_prefix_constraints) for i in range(args.batch_size)]

    for _ in range(args.repeats):
        _, dt_python = timed_run(lambda: [run_python(*case, tau=args.tau, K_max=args.k_max, log_clip_eps=args.log_clip_eps) for case in cases])
        python_times.append(dt_python)
        _, dt_cpp = timed_run(lambda: [run_cpp(case[0], case[1], case[2], tau=args.tau, K_max=args.k_max, log_clip_eps=args.log_clip_eps, include_prefix_constraints=include_prefix_constraints) for case in cases])
        cpp_times.append(dt_cpp)
        _, dt_cpp_batch = timed_run(lambda: run_cpp_batch(cases, tau=args.tau, K_max=args.k_max, log_clip_eps=args.log_clip_eps, include_prefix_constraints=include_prefix_constraints, n_threads=args.batch_threads))
        cpp_batch_times.append(dt_cpp_batch)

    print()
    print(f"benchmark batch_size={args.batch_size} threads={args.batch_threads if args.batch_threads > 0 else 'auto'}")
    print(summarize("python-loop", python_times))
    print(summarize("cpp-loop   ", cpp_times))
    print(summarize("cpp-batch  ", cpp_batch_times))
    print(f"speedup (python/cpp-batch mean): {statistics.mean(python_times) / statistics.mean(cpp_batch_times):.3f}x")
    print(f"speedup (cpp-loop/cpp-batch mean): {statistics.mean(cpp_times) / statistics.mean(cpp_batch_times):.3f}x")


if __name__ == "__main__":
    main()
