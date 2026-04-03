from __future__ import annotations

import math
import numpy as np

from klbox.kl_types import FloatArray


def clip_strictly_positive(x: FloatArray, eps: float) -> FloatArray:
    return np.maximum(x, float(eps))


def l1_norm(x: FloatArray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    if not math.isfinite(m) or m <= 0.0:
        return float(np.sum(x))
    return m * float(np.sum(x / m))


def normalize_to_simplex(x: FloatArray) -> FloatArray:
    s = l1_norm(x)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("normalize_to_simplex: sum must be positive and finite.")
    return np.asarray(x, dtype=np.float64) / float(s)
