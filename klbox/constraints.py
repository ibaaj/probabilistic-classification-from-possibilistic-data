from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

import numpy as np

from klbox.gaps import GapParameters
from klbox.kl_types import FloatArray, TINY_POS
from klbox.np_utils import clip_strictly_positive, normalize_to_simplex
from klbox.possibility import PossibilityOrder


def _normalize_strictly_positive(z: FloatArray, eps: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = clip_strictly_positive(z, eps)
    return normalize_to_simplex(z)


class Constraint:
    """
    Base class for the constraint sets C_i used by Dykstra's algorithm.
    """

    def project(self, z: FloatArray) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class PrefConstraint(Constraint):
    """
    C^{pref}_s = {p in Δ_n : sum_{k in A_s} p_k >= b}.

    This is the projector of Proposition 7.
    """

    idx: np.ndarray
    b: float

    def project(self, z: FloatArray) -> FloatArray:
        z_sharp = _normalize_strictly_positive(z, TINY_POS)

        rho = float(np.sum(z_sharp[self.idx]))
        b = float(self.b)

        if rho >= b:
            return z_sharp

        denom_in = max(rho, TINY_POS)
        denom_out = max(1.0 - rho, TINY_POS)

        scale_in = b / denom_in
        scale_out = (1.0 - b) / denom_out

        out = z_sharp * scale_out
        out[self.idx] = z_sharp[self.idx] * scale_in
        return out


@dataclass(frozen=True)
class GapConstraint(Constraint):
    """
    C = {p in Δ_n : p_i - p_j >= delta}, with -1 < delta < 1.

    This is the projector of Proposition 8.
    """

    i: int
    j: int
    delta: float
    feasibility_tol: float = 0.0

    def project(self, z: FloatArray) -> FloatArray:
        z_sharp = _normalize_strictly_positive(z, TINY_POS)

        i = int(self.i)
        j = int(self.j)
        delta = float(self.delta)

        s = float(z_sharp[i] - z_sharp[j])
        if s >= delta - float(self.feasibility_tol):
            return z_sharp

        zi = float(z_sharp[i])
        zj = float(z_sharp[j])
        u = 1.0 - zi - zj

        A = zi * (1.0 - delta)
        B = -delta * u
        C = -zj * (1.0 + delta)

        if A <= 0.0:
            return z_sharp

        disc = B * B - 4.0 * A * C
        disc = max(disc, 0.0)

        E = (-B + math.sqrt(disc)) / (2.0 * A)
        if not math.isfinite(E) or E <= 0.0:
            return z_sharp

        D = zi * E + zj / E + u
        if not math.isfinite(D) or D <= 0.0:
            return z_sharp

        out = z_sharp / D
        out[i] = (E / D) * zi
        out[j] = (1.0 / (E * D)) * zj
        return out


def build_constraint_family(
    order: PossibilityOrder,
    gaps: GapParameters,
    *,
    include_prefix_constraints: bool = True,
    include_lower_constraints: bool = True,
    include_upper_constraints: bool = True,
) -> List[Constraint]:
    """
    Build the family (C_i)_{i=1}^m used by Dykstra's algorithm.

    Order of constraints:
      1. prefix constraints
      2. lower-gap constraints
      3. upper-gap constraints
    """
    sigma = np.asarray(order.sigma, dtype=np.int64)
    tilde_pi = np.asarray(order.tilde_pi, dtype=np.float64)
    underline = np.asarray(gaps.underline, dtype=np.float64)
    overline = np.asarray(gaps.overline, dtype=np.float64)

    n = int(tilde_pi.size)
    n_minus_1 = max(n - 1, 0)

    if underline.shape != (n_minus_1,):
        raise ValueError(
            f"build_constraint_family: expected underline.shape={(n_minus_1,)}, got {underline.shape}."
        )
    if overline.shape != (n_minus_1,):
        raise ValueError(
            f"build_constraint_family: expected overline.shape={(n_minus_1,)}, got {overline.shape}."
        )

    constraints: List[Constraint] = []

    if include_prefix_constraints:
        for s in range(1, n):
            A_s = sigma[:s]
            b_s = 1.0 - float(tilde_pi[s])
            constraints.append(PrefConstraint(idx=A_s, b=b_s))

    if include_lower_constraints:
        for s in range(1, n):
            i = int(sigma[s - 1])
            j = int(sigma[s])
            constraints.append(GapConstraint(i=i, j=j, delta=float(underline[s - 1])))

    if include_upper_constraints:
        for s in range(1, n):
            i = int(sigma[s])
            j = int(sigma[s - 1])
            constraints.append(GapConstraint(i=i, j=j, delta=-float(overline[s - 1])))

    return constraints