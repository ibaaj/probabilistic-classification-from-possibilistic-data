from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from klbox.kl_types import FloatArray
from experiment.sample_adapter import get_system_constraints


def ece_score(probs: FloatArray, y: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    N = float(y.size)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if np.any(mask):
            ece += (np.sum(mask) / N) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(probs: FloatArray, y: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    N, C = probs.shape
    one_hot = np.zeros((N, C), dtype=np.float64)
    one_hot[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


ViolationFn = Callable[[Any, FloatArray, float], Dict[str, float]]


def topk_constraint_violations(sample: Any, p: FloatArray, tau: float) -> Dict[str, float]:
    system, _ = get_system_constraints(sample)
    p = np.asarray(p, dtype=np.float64)

    def _viol(b: np.ndarray, Ap: np.ndarray):
        r = b - Ap
        return float(np.max(np.maximum(r, 0.0), initial=0.0)), float(np.sum(r > tau))

    V_max, n_viol = _viol(system.b, system.A @ p)
    Vpref, viol_pref = _viol(system.b_pref, system.A_pref @ p)
    Vlow, viol_low = _viol(system.b_low, system.D_low @ p)
    Vup, viol_up = _viol(system.b_up, system.D_up @ p)

    return {
        "V_max": V_max,
        "n_viol": n_viol,
        "Vpref": Vpref,
        "Vlow": Vlow,
        "Vup": Vup,
        "viol_pref": viol_pref,
        "viol_low": viol_low,
        "viol_up": viol_up,
    }


def evaluate_metrics(
    samples: List[Any],
    model: Any,
    *,
    tau_eval: float = 0.0,
    violation_fn: Optional[ViolationFn] = None,
) -> Dict[str, Any]:
    X = np.stack([s.x for s in samples], axis=0)
    y = np.asarray([s.y for s in samples], dtype=np.int64)
    plausible = np.stack([s.plausible_mask for s in samples], axis=0).astype(np.float64)
    probs = model.predict_proba(X)
    N = len(samples)
    preds = np.argmax(probs, axis=1)

    out: Dict[str, Any] = {
        "acc": float(np.mean(preds == y)),
        "nll": float(np.mean([-math.log(max(probs[i, y[i]], 1e-15)) for i in range(N)])),
        "brier": brier_score(probs, y),
        "ece": ece_score(probs, y),
        "entropy": float(
            np.mean(
                [-np.sum(probs[i] * np.log(np.clip(probs[i], 1e-15, 1.0))) for i in range(N)]
            )
        ),
        "mass_plausible": float(np.mean(np.sum(probs * plausible, axis=1))),
        "top1_in_plausible": float(np.mean(plausible[np.arange(N), preds])),
    }

    if violation_fn is not None:
        cv = [violation_fn(s, probs[i], tau_eval) for i, s in enumerate(samples)]
        V = np.array([c["V_max"] for c in cv])
        out.update(
            {
                "V_mean": float(np.mean(V)),
                "V_median": float(np.median(V)),
                "V_p90": float(np.quantile(V, 0.9)),
                "V_max": float(np.max(V)),
                "V_le_tau": float(np.mean(V <= tau_eval)),
                "violated_mean": float(np.mean([c["n_viol"] for c in cv])),
                "Vpref_mean": float(np.mean([c["Vpref"] for c in cv])),
                "Vlow_mean": float(np.mean([c["Vlow"] for c in cv])),
                "Vup_mean": float(np.mean([c["Vup"] for c in cv])),
                "viol_pref_mean": float(np.mean([c["viol_pref"] for c in cv])),
                "viol_low_mean": float(np.mean([c["viol_low"] for c in cv])),
                "viol_up_mean": float(np.mean([c["viol_up"] for c in cv])),
            }
        )

    return out
