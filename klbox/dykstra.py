from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time

import numpy as np

from klbox.constraints import Constraint
from klbox.kl_types import FloatArray
from klbox.linear_system import LinearSystem, violation_V
from klbox.np_utils import clip_strictly_positive, normalize_to_simplex


@dataclass(frozen=True)
class DykstraResult:
    """
    Result returned by the implemented Dykstra loop.

    p_star is the iterate returned at termination.
    """
    p_star: FloatArray
    cycles: int
    final_V: float
    elapsed_s: float


def dykstra_kl_project(
    q: FloatArray,
    constraints: Sequence[Constraint],
    system: LinearSystem,
    tau: float,
    K_max: int,
    log_clip_eps: float,
) -> DykstraResult:
    """
    Implement the stabilized Dykstra iteration used in the numerical section.

    The returned p_star is the iterate obtained at termination.
    Stopping is based on the maximal feasibility violation V(p).
    """
    q = normalize_to_simplex(
        clip_strictly_positive(np.asarray(q, dtype=np.float64), log_clip_eps)
    )

    n = int(q.size)
    m = len(constraints)

    z_prev = q.copy()
    d_ring = np.zeros((m, n), dtype=np.float64)

    t0 = time.perf_counter()

    if m == 0:
        t1 = time.perf_counter()
        return DykstraResult(
            p_star=z_prev,
            cycles=0,
            final_V=violation_V(system, z_prev),
            elapsed_s=t1 - t0,
        )

    for cycle in range(1, K_max + 1):
        for h in range(1, m + 1):
            idx = h - 1
            d_t_minus_m = d_ring[idx]

            z_prev_clip = clip_strictly_positive(z_prev, log_clip_eps)
            log_u = np.log(z_prev_clip) + d_t_minus_m
            log_u -= float(np.max(log_u))
            u = np.exp(log_u)

            z_new = constraints[idx].project(u)

            z_new_clip = clip_strictly_positive(z_new, log_clip_eps)
            d_t = d_t_minus_m + np.log(z_prev_clip / z_new_clip)

            if not np.all(np.isfinite(z_new)):
                raise FloatingPointError(
                    f"z_new non-finite at cycle={cycle}, h={h}, constraint={type(constraints[idx]).__name__}"
                )
            if not np.all(np.isfinite(d_t)):
                raise FloatingPointError(
                    f"d_t non-finite at cycle={cycle}, h={h}, constraint={type(constraints[idx]).__name__}"
                )

            d_ring[idx] = d_t
            z_prev = z_new

        V = violation_V(system, z_prev)
        if V <= tau:
            t1 = time.perf_counter()
            return DykstraResult(
                p_star=z_prev,
                cycles=cycle,
                final_V=V,
                elapsed_s=t1 - t0,
            )

    t1 = time.perf_counter()
    V = violation_V(system, z_prev)
    return DykstraResult(
        p_star=z_prev,
        cycles=K_max,
        final_V=V,
        elapsed_s=t1 - t0,
    )