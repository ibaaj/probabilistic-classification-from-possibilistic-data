from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from klbox.gaps import GapParameters, choose_gap_parameters
from klbox.kl_types import FloatArray
from klbox.linear_system import LinearSystem, build_linear_system, violation_V
from klbox.possibility import PossibilityOrder, antipignistic_reverse_mapping, compute_possibility_order


@dataclass(frozen=True)
class TopKSample:
    x: FloatArray
    y: int
    pi: FloatArray
    plausible_mask: np.ndarray

    order: PossibilityOrder
    dot_p: FloatArray
    gaps: GapParameters
    system: LinearSystem


@dataclass(frozen=True)
class TopKConfig:
    n_classes: int = 20
    d: int = 10

    # alpha(x) and its noise
    alpha: float = 0.2
    alpha_noise: float = 0.0

    # data geometry/noise
    class_sep: float = 2.0
    x_noise: float = 1.0

    # π construction (stair)
    pi_eps: float = 1e-6
    pi_stair_step: float = 1e-3
    pi_stair_m: int = 0

    # tie handling for π-order
    tie_tol: float = 0.0

    # gap epsilon cap
    eps_cap: float = 1e-9


def _topk_ordered_excluding_y(dist: FloatArray, y: int) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    C = int(dist.size)
    y = int(y)
    if y < 0 or y >= C:
        raise ValueError("y out of range.")
    idx = np.arange(C, dtype=np.int64)
    mask = idx != y
    idx_other = idx[mask]
    dist_other = dist[mask]
    order = np.lexsort((idx_other, dist_other))
    return idx_other[order]


def _build_pi_topk_stair(
    y: int,
    ordered_neighbors: np.ndarray,
    n_classes: int,
    alpha: float,
    pi_eps: float,
    *,
    stair_step: float,
    stair_m: int,
) -> FloatArray:
    C = int(n_classes)
    y = int(y)
    alpha = float(alpha)
    pi_eps = float(pi_eps)
    stair_step = float(stair_step)
    stair_m = int(stair_m)

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Require 0 <= alpha <= 1.")
    if not (0.0 < pi_eps < 1.0):
        raise ValueError("Require 0 < pi_eps < 1.")
    if stair_step < 0.0:
        raise ValueError("Require stair_step >= 0.")
    if y < 0 or y >= C:
        raise ValueError("y out of range.")

    ordered_neighbors = np.asarray(ordered_neighbors, dtype=np.int64)
    if ordered_neighbors.ndim != 1:
        raise ValueError("ordered_neighbors must be 1D.")
    if np.any((ordered_neighbors < 0) | (ordered_neighbors >= C)):
        raise ValueError("ordered_neighbors contains out-of-range indices.")
    if np.any(ordered_neighbors == y):
        raise ValueError("ordered_neighbors must exclude y.")

    base = np.zeros(C, dtype=np.float64)
    base[y] = 1.0

    k_minus_1 = int(ordered_neighbors.size)
    m = k_minus_1 if stair_m <= 0 else min(k_minus_1, stair_m)

    alpha_floor = 0.0

    if m <= 0:
        last_level = alpha_floor
    else:
        last_level = alpha - float(m - 1) * stair_step
        last_level = max(last_level, alpha_floor)

    for r in range(k_minus_1):
        j = int(ordered_neighbors[r])
        if r < m:
            level = alpha - float(r) * stair_step
            level = max(level, alpha_floor)
        else:
            level = last_level
        base[j] = level

    pi = base + pi_eps
    pi[y] = 1.0

    cap = 1.0 - pi_eps
    pi[ordered_neighbors] = np.minimum(pi[ordered_neighbors], cap)

    return pi


def make_topk_dataset(
    cfg: TopKConfig,
    N: int,
    rng: np.random.Generator,
    mu: np.ndarray,
) -> List[TopKSample]:
    C = int(cfg.n_classes)
    d = int(cfg.d)

    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (C, d):
        raise ValueError(f"mu must have shape {(C, d)}, got {mu.shape}.")

    samples: List[TopKSample] = []

    for _ in range(int(N)):
        y = int(rng.integers(0, C))
        x = mu[y] + float(cfg.x_noise) * rng.normal(size=(d,)).astype(np.float64)

        dist = np.sum((mu - x[None, :]) ** 2, axis=1)
        ordered_neighbors = _topk_ordered_excluding_y(dist, y=y)

        # Compatibility field for the shared metric interface.
        # In the current synthetic top-k setup this is the full class set, so
        # plausibility-mass-style metrics are vacuous and are filtered out upstream.
        plausible_mask = np.ones(C, dtype=bool)

        eta = float(rng.normal())
        alpha_x = float(cfg.alpha) + float(cfg.alpha_noise) * eta
        alpha_x = float(np.clip(alpha_x, 0.0, 1.0 - float(cfg.pi_eps)))

        pi = _build_pi_topk_stair(
            y=y,
            ordered_neighbors=ordered_neighbors,
            n_classes=C,
            alpha=float(alpha_x),
            pi_eps=float(cfg.pi_eps),
            stair_step=float(cfg.pi_stair_step),
            stair_m=int(cfg.pi_stair_m),
        )

        pi_arr = np.asarray(pi, dtype=np.float64)
        if not np.all(np.isfinite(pi_arr)):
            raise RuntimeError("pi contains non-finite values.")
        if not np.all(pi_arr > 0.0):
            raise RuntimeError("pi must be strictly positive everywhere.")
        if abs(float(np.max(pi_arr)) - 1.0) > 1e-12:
            raise RuntimeError(f"max(pi)={float(np.max(pi_arr))} but expected 1.")

        order = compute_possibility_order(pi)
        anti = antipignistic_reverse_mapping(order)

        gaps = choose_gap_parameters(
            tilde_pi=order.tilde_pi,
            dot_g=anti.dot_g,
            tie_tol=float(cfg.tie_tol),
            eps_cap=float(cfg.eps_cap),
        )

        system = build_linear_system(order, gaps)

        if violation_V(system, anti.dot_p) > 1e-10:
            raise RuntimeError("dot_p is not feasible for the constructed F^{box} (gap-wide); check construction.")

        samples.append(
            TopKSample(
                x=x,
                y=y,
                pi=pi_arr,
                plausible_mask=plausible_mask,
                order=order,
                dot_p=anti.dot_p,
                gaps=gaps,
                system=system,
            )
        )

    return samples