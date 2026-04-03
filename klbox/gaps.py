from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from klbox.kl_types import FloatArray


@dataclass(frozen=True)
class GapParameters:
    """
    Gap parameters used in the shape constraints.

    underline[r0] = \\underline\\delta_{r0+1}
    overline[r0]  = \\overline\\delta_{r0+1}
    """

    underline: FloatArray
    overline: FloatArray


def choose_gap_parameters(
    tilde_pi: FloatArray,
    dot_g: FloatArray,
    tie_tol: float,
    eps_cap: float,
) -> GapParameters:
    """
    Choose the lower and upper gap parameters from the ordered possibility
    vector \\tilde\\pi and the antipignistic adjacent gaps \\dot g.

    Equal ranks receive:
      underline[r] = overline[r] = 0.

    Strict ranks receive a wide box:
      underline[r] = epsilon,
      overline[r]  = 1 - epsilon,
    with epsilon chosen so that \\dot p stays feasible.
    """
    tilde_pi = np.asarray(tilde_pi, dtype=np.float64)
    dot_g = np.asarray(dot_g, dtype=np.float64)

    if tilde_pi.ndim != 1 or dot_g.ndim != 1:
        raise ValueError("choose_gap_parameters: tilde_pi and dot_g must be 1D arrays.")
    if tilde_pi.size != dot_g.size + 1:
        raise ValueError("choose_gap_parameters: expected len(dot_g)=len(tilde_pi)-1.")

    tie_tol = float(tie_tol)
    eps_cap = float(eps_cap)

    if tie_tol < 0.0:
        raise ValueError("choose_gap_parameters: tie_tol must be >= 0.")
    if eps_cap < 0.0:
        raise ValueError("choose_gap_parameters: eps_cap must be >= 0.")

    diffs = tilde_pi[:-1] - tilde_pi[1:]
    is_equal = np.abs(diffs) <= tie_tol
    is_strict = diffs > tie_tol

    n_minus_1 = int(tilde_pi.size - 1)
    underline = np.zeros(n_minus_1, dtype=np.float64)
    overline = np.zeros(n_minus_1, dtype=np.float64)

    underline[is_equal] = 0.0
    overline[is_equal] = 0.0

    strict_idx = np.where(is_strict)[0]
    if strict_idx.size == 0:
        return GapParameters(underline=underline, overline=overline)

    dotg_strict = dot_g[strict_idx].astype(np.float64)
    if np.any(~np.isfinite(dotg_strict)) or np.any(dotg_strict < 0.0):
        raise ValueError("choose_gap_parameters: dot_g must be finite and >= 0 on strict ranks.")
    if np.any(dotg_strict >= 1.0):
        raise ValueError("choose_gap_parameters: found dot_g >= 1 on strict ranks; cannot build valid box.")

    g_min = float(np.min(dotg_strict))
    g_max = float(np.max(dotg_strict))

    epsilon_raw = min(eps_cap, g_min)
    epsilon_raw = max(0.0, epsilon_raw)

    # Feasibility requires epsilon <= 1 - g_max.
    epsilon = min(epsilon_raw, max(0.0, 1.0 - g_max))
    epsilon = max(0.0, float(epsilon))

    one_minus = np.nextafter(1.0, 0.0)
    wide_upper = min(one_minus, 1.0 - epsilon)

    underline[strict_idx] = epsilon
    overline[strict_idx] = wide_upper

    if np.any(underline[strict_idx] > dotg_strict + 1e-15):
        raise RuntimeError("choose_gap_parameters: underline exceeds dot_g on strict ranks (box empty).")
    if np.any(overline[strict_idx] + 1e-15 < dotg_strict):
        raise RuntimeError("choose_gap_parameters: overline below dot_g on strict ranks (box empty).")

    return GapParameters(underline=underline, overline=overline)