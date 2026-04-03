from __future__ import annotations

"""Shared helper functions for ChaosNLI ambiguity and slice analysis."""

from typing import Any

import numpy as np


def safe_entropy_from_probs(p: np.ndarray) -> float:
    """Compute entropy while safely ignoring zero-probability entries."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > 0.0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def vote_probs(votes: np.ndarray, y: int, c: int) -> np.ndarray:
    """Convert vote counts to a probability vector.

    If the vote total is zero, fall back to a one-hot vector at ``y``.
    """
    votes = np.asarray(votes, dtype=np.float64)
    total = float(np.sum(votes))
    if total > 0.0:
        return votes / total

    out = np.zeros(int(c), dtype=np.float64)
    out[int(y)] = 1.0
    return out


def sample_identifier(sample: Any, idx: int) -> str:
    """Extract a stable external identifier from a sample object."""
    for name in ("sample_id", "uid", "comment_id", "id"):
        if hasattr(sample, name):
            return str(getattr(sample, name))
    return str(idx)


def subset_from_sample_id(sample_id: str) -> str:
    """Recover the subset prefix from an id such as ``snli::123``."""
    sid = str(sample_id)
    return sid.split("::", 1)[0] if "::" in sid else ""
