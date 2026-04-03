from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from klbox.dykstra import DykstraResult
from klbox.gaps import GapParameters
from klbox.kl_types import FloatArray
from klbox.possibility import PossibilityOrder

try:
    from . import _dykstra_cpp as _backend
except ImportError:
    import _dykstra_cpp as _backend


def _as_float64_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.ascontiguousarray(arr)


def _as_float64_2d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return np.ascontiguousarray(arr)


def _as_int32_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.ascontiguousarray(arr)


def _as_int32_2d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return np.ascontiguousarray(arr)


def _result_from_raw_dict(out: dict[str, Any]) -> DykstraResult:
    return DykstraResult(
        p_star=np.asarray(out["p_star"], dtype=np.float64),
        cycles=int(out["cycles"]),
        final_V=float(out["final_V"]),
        elapsed_s=float(out["elapsed_s"]),
    )


def _results_from_batch_raw_dict(out: dict[str, Any]) -> list[DykstraResult]:
    p_star = np.asarray(out["p_star"], dtype=np.float64)
    cycles = np.asarray(out["cycles"], dtype=np.int64)
    final_v = np.asarray(out["final_V"], dtype=np.float64)
    elapsed = np.asarray(out["elapsed_s"], dtype=np.float64)

    if p_star.ndim != 2:
        raise ValueError(f"Expected p_star.ndim == 2, got {p_star.ndim}.")
    batch_size = int(p_star.shape[0])
    if cycles.shape != (batch_size,) or final_v.shape != (batch_size,) or elapsed.shape != (batch_size,):
        raise ValueError("Batch backend returned inconsistent output shapes.")

    return [
        DykstraResult(
            p_star=np.asarray(p_star[i], dtype=np.float64),
            cycles=int(cycles[i]),
            final_V=float(final_v[i]),
            elapsed_s=float(elapsed[i]),
        )
        for i in range(batch_size)
    ]


def _stack_orders(orders: Sequence[PossibilityOrder]) -> tuple[np.ndarray, np.ndarray]:
    if not orders:
        raise ValueError("orders must be non-empty")
    sigma = np.stack([np.asarray(order.sigma, dtype=np.int32) for order in orders], axis=0)
    tilde_pi = np.stack([np.asarray(order.tilde_pi, dtype=np.float64) for order in orders], axis=0)
    return np.ascontiguousarray(sigma), np.ascontiguousarray(tilde_pi)


def _stack_gaps(gaps_list: Sequence[GapParameters]) -> tuple[np.ndarray, np.ndarray]:
    if not gaps_list:
        raise ValueError("gaps_list must be non-empty")
    underline = np.stack([np.asarray(gaps.underline, dtype=np.float64) for gaps in gaps_list], axis=0)
    overline = np.stack([np.asarray(gaps.overline, dtype=np.float64) for gaps in gaps_list], axis=0)
    return np.ascontiguousarray(underline), np.ascontiguousarray(overline)


def dykstra_kl_project_cpp_raw(
    q: FloatArray,
    sigma: np.ndarray,
    tilde_pi: FloatArray,
    underline: FloatArray,
    overline: FloatArray,
    tau: float,
    K_max: int,
    log_clip_eps: float,
    *,
    include_prefix_constraints: bool = True,
    include_lower_constraints: bool = True,
    include_upper_constraints: bool = True,
) -> DykstraResult:
    out = _backend.dykstra_kl_project_cpp_raw(
        _as_float64_1d(q, "q"),
        _as_int32_1d(sigma, "sigma"),
        _as_float64_1d(tilde_pi, "tilde_pi"),
        _as_float64_1d(underline, "underline"),
        _as_float64_1d(overline, "overline"),
        float(tau),
        int(K_max),
        float(log_clip_eps),
        bool(include_prefix_constraints),
        bool(include_lower_constraints),
        bool(include_upper_constraints),
    )
    return _result_from_raw_dict(out)


def dykstra_kl_project_cpp(
    q: FloatArray,
    order: PossibilityOrder,
    gaps: GapParameters,
    tau: float,
    K_max: int,
    log_clip_eps: float,
    *,
    include_prefix_constraints: bool = True,
    include_lower_constraints: bool = True,
    include_upper_constraints: bool = True,
) -> DykstraResult:
    return dykstra_kl_project_cpp_raw(
        q=q,
        sigma=order.sigma,
        tilde_pi=order.tilde_pi,
        underline=gaps.underline,
        overline=gaps.overline,
        tau=tau,
        K_max=K_max,
        log_clip_eps=log_clip_eps,
        include_prefix_constraints=include_prefix_constraints,
        include_lower_constraints=include_lower_constraints,
        include_upper_constraints=include_upper_constraints,
    )


def dykstra_kl_project_cpp_batch_raw(
    q_batch: FloatArray,
    sigma_batch: np.ndarray,
    tilde_pi_batch: FloatArray,
    underline_batch: FloatArray,
    overline_batch: FloatArray,
    tau: float,
    K_max: int,
    log_clip_eps: float,
    *,
    n_threads: int = 0,
    include_prefix_constraints: bool = True,
    include_lower_constraints: bool = True,
    include_upper_constraints: bool = True,
) -> list[DykstraResult]:
    out = _backend.dykstra_kl_project_cpp_batch_raw(
        _as_float64_2d(q_batch, "q_batch"),
        _as_int32_2d(sigma_batch, "sigma_batch"),
        _as_float64_2d(tilde_pi_batch, "tilde_pi_batch"),
        _as_float64_2d(underline_batch, "underline_batch"),
        _as_float64_2d(overline_batch, "overline_batch"),
        float(tau),
        int(K_max),
        float(log_clip_eps),
        int(n_threads),
        bool(include_prefix_constraints),
        bool(include_lower_constraints),
        bool(include_upper_constraints),
    )
    return _results_from_batch_raw_dict(out)


def dykstra_kl_project_cpp_batch(
    q_batch: FloatArray,
    orders: Sequence[PossibilityOrder],
    gaps_list: Sequence[GapParameters],
    tau: float,
    K_max: int,
    log_clip_eps: float,
    *,
    n_threads: int = 0,
    include_prefix_constraints: bool = True,
    include_lower_constraints: bool = True,
    include_upper_constraints: bool = True,
) -> list[DykstraResult]:
    q_batch = _as_float64_2d(q_batch, "q_batch")
    if q_batch.shape[0] != len(orders) or q_batch.shape[0] != len(gaps_list):
        raise ValueError(
            f"Batch size mismatch: q_batch has {q_batch.shape[0]} rows, "
            f"orders has length {len(orders)}, gaps_list has length {len(gaps_list)}."
        )

    sigma_batch, tilde_pi_batch = _stack_orders(orders)
    underline_batch, overline_batch = _stack_gaps(gaps_list)
    return dykstra_kl_project_cpp_batch_raw(
        q_batch=q_batch,
        sigma_batch=sigma_batch,
        tilde_pi_batch=tilde_pi_batch,
        underline_batch=underline_batch,
        overline_batch=overline_batch,
        tau=tau,
        K_max=K_max,
        log_clip_eps=log_clip_eps,
        n_threads=n_threads,
        include_prefix_constraints=include_prefix_constraints,
        include_lower_constraints=include_lower_constraints,
        include_upper_constraints=include_upper_constraints,
    )
