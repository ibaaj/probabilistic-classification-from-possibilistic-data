from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from klbox.gaps import GapParameters
from klbox.kl_types import FloatArray
from klbox.possibility import PossibilityOrder


@dataclass(frozen=True)
class LinearSystem:
    """
    Block linear system A p >= b used to monitor feasibility.

    The three blocks are:
      - prefix block
      - lower-gap block
      - upper-gap block
    """

    A: FloatArray
    b: FloatArray

    A_pref: FloatArray
    b_pref: FloatArray

    D: FloatArray

    D_low: FloatArray
    b_low: FloatArray

    D_up: FloatArray
    b_up: FloatArray


def _empty_block(n: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((0, n), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )


def _build_prefix_block(sigma: np.ndarray, tilde_pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(tilde_pi.size)
    n_minus_1 = max(n - 1, 0)

    A_pref = np.zeros((n_minus_1, n), dtype=np.float64)
    b_pref = np.zeros((n_minus_1,), dtype=np.float64)

    for r in range(1, n):
        A_pref[r - 1, sigma[:r]] = 1.0
        b_pref[r - 1] = 1.0 - float(tilde_pi[r])

    return A_pref, b_pref


def _build_D_block(sigma: np.ndarray, n: int) -> np.ndarray:
    n_minus_1 = max(n - 1, 0)
    D = np.zeros((n_minus_1, n), dtype=np.float64)

    for r in range(1, n):
        i = int(sigma[r - 1])
        j = int(sigma[r])
        D[r - 1, i] = +1.0
        D[r - 1, j] = -1.0

    return D


def build_linear_system(
    order: PossibilityOrder,
    gaps: GapParameters,
    *,
    include_prefix_constraints: bool = True,
    include_lower_constraints: bool = True,
    include_upper_constraints: bool = True,
) -> LinearSystem:
    """
    Build the block linear system A p >= b.

    Blocks:
      - prefix:
          sum_{k in {sigma(1),...,sigma(r)}} p_k >= 1 - tilde_pi[r+1]
      - lower:
          p_{sigma(r)} - p_{sigma(r+1)} >= underline[r]
      - upper:
          p_{sigma(r)} - p_{sigma(r+1)} <= overline[r]
        written as
          p_{sigma(r+1)} - p_{sigma(r)} >= -overline[r]
    """
    sigma = np.asarray(order.sigma, dtype=np.int64)
    tilde_pi = np.asarray(order.tilde_pi, dtype=np.float64)
    underline = np.asarray(gaps.underline, dtype=np.float64)
    overline = np.asarray(gaps.overline, dtype=np.float64)

    if sigma.ndim != 1 or tilde_pi.ndim != 1:
        raise ValueError("build_linear_system: sigma and tilde_pi must be 1D.")
    if sigma.size != tilde_pi.size:
        raise ValueError("build_linear_system: sigma and tilde_pi must have the same length.")

    n = int(tilde_pi.size)
    n_minus_1 = max(n - 1, 0)

    if underline.shape != (n_minus_1,):
        raise ValueError(
            f"Expected underline.shape={(n_minus_1,)}, got {underline.shape}."
        )
    if overline.shape != (n_minus_1,):
        raise ValueError(
            f"Expected overline.shape={(n_minus_1,)}, got {overline.shape}."
        )

    if include_prefix_constraints:
        A_pref, b_pref = _build_prefix_block(sigma, tilde_pi)
    else:
        A_pref, b_pref = _empty_block(n)

    D = _build_D_block(sigma, n)

    if include_lower_constraints:
        D_low = D.copy()
        b_low = underline.copy()
    else:
        D_low, b_low = _empty_block(n)

    if include_upper_constraints:
        D_up = -D
        b_up = -overline.copy()
    else:
        D_up, b_up = _empty_block(n)

    blocks_A: list[np.ndarray] = []
    blocks_b: list[np.ndarray] = []

    if A_pref.shape[0] > 0:
        blocks_A.append(A_pref)
        blocks_b.append(b_pref)

    if D_low.shape[0] > 0:
        blocks_A.append(D_low)
        blocks_b.append(b_low)

    if D_up.shape[0] > 0:
        blocks_A.append(D_up)
        blocks_b.append(b_up)

    if blocks_A:
        A = np.vstack(blocks_A)
        b = np.concatenate(blocks_b)
    else:
        A = np.zeros((0, n), dtype=np.float64)
        b = np.zeros((0,), dtype=np.float64)

    return LinearSystem(
        A=A,
        b=b,
        A_pref=A_pref,
        b_pref=b_pref,
        D=D,
        D_low=D_low,
        b_low=b_low,
        D_up=D_up,
        b_up=b_up,
    )


def violation_V(system: LinearSystem, p: FloatArray) -> float:
    """
    V(p) = max_i (b_i - a_i^T p)_+ for the system A p >= b.
    """
    p = np.asarray(p, dtype=np.float64)
    if system.A.shape[0] == 0:
        return 0.0
    return float(np.maximum(system.b - system.A @ p, 0.0).max(initial=0.0))