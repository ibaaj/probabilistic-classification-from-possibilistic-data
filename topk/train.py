from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch

from experiment.metrics import evaluate_metrics as evaluate_metrics_common, topk_constraint_violations
from experiment.target_protocol import ProjectionStats
from experiment.train import train_model
from topk.data import TopKSample
from topk.model import AnyHead, build_head
from topk.targets import PlainProjectionTarget, fixed_dot_p_target

TargetKind = Literal["projection", "fixed_target"]
_TOPK_ONLY_EXCLUDED_METRICS = frozenset({"mass_plausible", "top1_in_plausible"})


def _normalize_target_kind(name: str) -> TargetKind:
    value = str(name).strip().lower()
    if value in {"projection", "a"}:
        return "projection"
    if value in {"fixed_target", "fixed", "dot_p", "b"}:
        return "fixed_target"
    raise ValueError(f"Unknown target kind {name!r}.")


def evaluate_metrics(samples: List[TopKSample], model: Any, tau_eval: float) -> Dict[str, Any]:
    if len(samples) == 0:
        return {}

    metrics = evaluate_metrics_common(
        samples=samples,
        model=model,
        tau_eval=float(tau_eval),
        violation_fn=topk_constraint_violations,
    )
    for metric_name in _TOPK_ONLY_EXCLUDED_METRICS:
        metrics.pop(metric_name, None)
    return metrics


def train_topk_model(
    *,
    samples_train: List[TopKSample],
    samples_test: List[TopKSample],
    target_kind: str,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    proj_tau: float,
    proj_Kmax: int,
    log_clip_eps: float,
    seed: int,
    head: str = "linear",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    proj_engine: str = "cpp_batch",
    proj_n_threads: int = 0,
    no_progress: bool = False,
) -> Tuple[AnyHead, Dict[str, Any]]:
    if len(samples_train) == 0:
        raise ValueError("samples_train must be non-empty.")
    if len(samples_test) == 0:
        raise ValueError("samples_test must be non-empty.")

    normalized_target_kind = _normalize_target_kind(target_kind)
    torch.manual_seed(int(seed))

    d = int(np.asarray(samples_train[0].x, dtype=np.float64).size)
    c = int(np.asarray(samples_train[0].dot_p, dtype=np.float64).size)

    model = build_head(
        head=head,
        d=d,
        C=c,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        seed=int(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    if normalized_target_kind == "projection":
        target_fn = PlainProjectionTarget(
            tau=float(proj_tau),
            K_max=int(proj_Kmax),
            log_clip_eps=float(log_clip_eps),
            backend=str(proj_engine),
            n_threads=int(proj_n_threads),
        )
        model_alias = "A"
    else:
        target_fn = fixed_dot_p_target
        model_alias = "B"

    projection_stats = train_model(
        samples_train=samples_train,
        model=model,
        optimizer=optimizer,
        target_fn=target_fn,
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
        val_items=None,
        scheduler=None,
        no_progress=bool(no_progress),
    )

    tau_eval = float(proj_tau)
    train_metrics = evaluate_metrics(samples_train, model, tau_eval=tau_eval)
    test_metrics = evaluate_metrics(samples_test, model, tau_eval=tau_eval)

    log: Dict[str, Any] = {
        "model_alias": model_alias,
        "target_kind": normalized_target_kind,
        "head": str(head).lower(),
        "optimizer": "adam",
        "lr": float(lr),
        "device_effective": "cpu",
        "dtype_effective": "torch.float64",
        "proj_engine": str(proj_engine),
        "proj_n_threads": int(proj_n_threads),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    if normalized_target_kind == "projection" and projection_stats is not None:
        if not isinstance(projection_stats, ProjectionStats):
            raise TypeError(f"Unexpected projection_stats type: {type(projection_stats).__name__}.")
        log["projection_stats_train"] = {
            "calls": int(projection_stats.calls),
            "cycles_mean": float(projection_stats.cycles_mean),
            "cycles_p90": float(projection_stats.cycles_p90),
            "time_mean_s": float(projection_stats.time_mean_s),
            "finalV_mean": float(projection_stats.finalV_mean),
            "finalV_max": float(projection_stats.finalV_max),
        }

    return model, log

