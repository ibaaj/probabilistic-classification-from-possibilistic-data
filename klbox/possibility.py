from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from klbox.kl_types import FloatArray
from klbox.np_utils import normalize_to_simplex


@dataclass(frozen=True)
class PossibilityOrder:
    """
    Possibility ordering used throughout the paper.

    Attributes
    ----------
    sigma:
        0-based permutation such that
        pi[sigma[0]] >= pi[sigma[1]] >= ... >= pi[sigma[n-1]].
    tilde_pi:
        Sorted possibility values:
        tilde_pi[r0] = \\tilde\\pi_{r0+1}, with tilde_pi[0] = 1 after normalization.
    """

    sigma: np.ndarray
    tilde_pi: FloatArray


def compute_possibility_order(pi: FloatArray) -> PossibilityOrder:
    """
    Given a possibility vector pi, compute the permutation sigma and the
    sorted vector tilde_pi used in the paper.

    The returned tilde_pi is normalized so that tilde_pi[0] = 1.
    """
    pi = np.asarray(pi, dtype=np.float64)
    if pi.ndim != 1:
        raise ValueError("compute_possibility_order: pi must be a 1D array.")
    if pi.size == 0:
        raise ValueError("compute_possibility_order: pi must be non-empty.")

    sigma = np.argsort(-pi, kind="stable")
    tilde_pi = pi[sigma].astype(np.float64)

    if float(tilde_pi[0]) <= 0.0:
        raise ValueError("compute_possibility_order: pi must have a positive maximum.")

    tilde_pi = tilde_pi / float(tilde_pi[0])
    return PossibilityOrder(sigma=sigma, tilde_pi=tilde_pi)


@dataclass(frozen=True)
class AntipignisticReverse:
    """
    Antipignistic reverse mapping.

    Attributes
    ----------
    dot_p:
        Probability vector \\dot p in the original indexing.
    dot_g:
        Adjacent gaps \\dot g_r for r = 1, ..., n-1.
    """

    dot_p: FloatArray
    dot_g: FloatArray


def antipignistic_reverse_mapping(order: PossibilityOrder) -> AntipignisticReverse:
    """
    Compute the antipignistic reverse mapping from the ordered possibility
    vector \\tilde\\pi.

    Formula:
      dot p_{sigma(r)} = sum_{j=r}^n (tilde_pi_j - tilde_pi_{j+1}) / j
    with the convention tilde_pi_{n+1} = 0.

    The adjacent gaps are:
      dot g_r = (tilde_pi_r - tilde_pi_{r+1}) / r,   r = 1, ..., n-1.
    """
    tilde_pi = np.asarray(order.tilde_pi, dtype=np.float64)
    if tilde_pi.ndim != 1:
        raise ValueError("antipignistic_reverse_mapping: tilde_pi must be 1D.")

    n = int(tilde_pi.size)

    tilde_ext = np.concatenate(
        [tilde_pi, np.array([0.0], dtype=np.float64)]
    )  # tilde_pi_{n+1} = 0

    diffs = tilde_ext[:-1] - tilde_ext[1:]
    r = np.arange(1, n + 1, dtype=np.float64)

    s = diffs / r
    dot_p_sigma = np.cumsum(s[::-1])[::-1]
    dot_g = s[:-1].copy()

    dot_p = np.empty(n, dtype=np.float64)
    dot_p[order.sigma] = dot_p_sigma
    dot_p = normalize_to_simplex(dot_p)

    return AntipignisticReverse(dot_p=dot_p, dot_g=dot_g)