from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from experiment.projection_common import normalize_prob, projection_stats_from_results, stack_field
from experiment.target_protocol import ProjectionStats
from klbox.dykstra import DykstraResult
from klbox.dykstra_cpp import dykstra_kl_project_cpp, dykstra_kl_project_cpp_batch
from klbox.kl_types import FloatArray

ProjectionBackend = Literal["python", "cpp", "cpp_batch", "auto"]


@dataclass(frozen=True)
class ProjectionConfig:
    tau: float = 1e-10
    K_max: int = 300
    log_clip_eps: float = 1e-15
    backend: ProjectionBackend = "cpp_batch"
    n_threads: int = 0


def _resolve_backend(name: ProjectionBackend) -> ProjectionBackend:
    if name == "auto":
        return "cpp_batch"
    if name not in {"python", "cpp", "cpp_batch"}:
        raise ValueError(f"Unsupported backend={name!r}.")
    return name


def _project_sample_to_full_box(q: FloatArray, sample: Any, cfg: ProjectionConfig) -> DykstraResult:
    q = normalize_prob(q, cfg.log_clip_eps)
    return dykstra_kl_project_cpp(
        q=q,
        order=sample.order,
        gaps=sample.gaps,
        tau=float(cfg.tau),
        K_max=int(cfg.K_max),
        log_clip_eps=float(cfg.log_clip_eps),
        include_prefix_constraints=True,
        include_lower_constraints=True,
        include_upper_constraints=True,
    )


def _project_batch_to_full_box(q_batch: FloatArray, batch: Sequence[Any], cfg: ProjectionConfig) -> list[DykstraResult]:
    normalized = np.stack([normalize_prob(row, cfg.log_clip_eps) for row in np.asarray(q_batch, dtype=np.float64)], axis=0)
    orders = [sample.order for sample in batch]
    gaps_list = [sample.gaps for sample in batch]
    return dykstra_kl_project_cpp_batch(
        q_batch=normalized,
        orders=orders,
        gaps_list=gaps_list,
        tau=float(cfg.tau),
        K_max=int(cfg.K_max),
        log_clip_eps=float(cfg.log_clip_eps),
        n_threads=int(cfg.n_threads),
        include_prefix_constraints=True,
        include_lower_constraints=True,
        include_upper_constraints=True,
    )


def fixed_dot_p_target(q_batch: FloatArray, batch: Sequence[Any]) -> np.ndarray:
    del q_batch
    return stack_field(batch, "dot_p")


class PlainProjectionTarget:
    """One-step KL projection onto the sample-specific full KL-box."""

    def __init__(
        self,
        tau: float = 1e-10,
        K_max: int = 300,
        log_clip_eps: float = 1e-15,
        *,
        backend: ProjectionBackend = "cpp_batch",
        n_threads: int = 0,
    ) -> None:
        self.cfg = ProjectionConfig(
            tau=float(tau),
            K_max=int(K_max),
            log_clip_eps=float(log_clip_eps),
            backend=_resolve_backend(backend),
            n_threads=int(n_threads),
        )
        self._results: list[DykstraResult] = []

    def start_run(self, total_steps: int | None = None) -> None:
        del total_steps
        self._results = []

    def __call__(self, q_batch: FloatArray, batch: Sequence[Any]) -> np.ndarray:
        q_batch = np.asarray(q_batch, dtype=np.float64)
        if q_batch.ndim != 2:
            raise ValueError(f"Expected q_batch.ndim == 2, got {q_batch.ndim}.")
        if q_batch.shape[0] != len(batch):
            raise ValueError(
                f"Batch size mismatch: q_batch has {q_batch.shape[0]} rows, "
                f"but batch has length {len(batch)}."
            )

        backend = _resolve_backend(self.cfg.backend)
        if backend == "cpp_batch":
            results = _project_batch_to_full_box(q_batch, batch, self.cfg)
        else:
            results = [_project_sample_to_full_box(q_batch[i], sample, self.cfg) for i, sample in enumerate(batch)]

        targets = np.empty_like(q_batch, dtype=np.float64)
        for row_index, result in enumerate(results):
            targets[row_index] = normalize_prob(result.p_star, self.cfg.log_clip_eps)
        self._results.extend(results)
        return targets

    def flush_stats(self) -> ProjectionStats:
        stats = projection_stats_from_results(self._results)
        self._results = []
        return stats
