from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from klbox.constraints import build_constraint_family
from klbox.dykstra import DykstraResult, dykstra_kl_project
from klbox.dykstra_cpp import dykstra_kl_project_cpp, dykstra_kl_project_cpp_batch
from klbox.gaps import GapParameters
from klbox.kl_types import FloatArray
from klbox.linear_system import build_linear_system
from klbox.possibility import PossibilityOrder

from experiment.projection_common import normalize_prob, projection_stats_from_results, stack_field
from experiment.target_protocol import ProjectionStats

BackendName = Literal["auto", "python", "cpp", "cpp_batch"]


@dataclass(frozen=True)
class ProjectionConfig:
    tau: float = 1e-10
    K_max: int = 300
    log_clip_eps: float = 1e-15
    backend: BackendName = "auto"
    n_threads: int = 0


def _resolve_backend(name: BackendName) -> BackendName:
    if name == "auto":
        return "cpp_batch"
    if name not in {"python", "cpp", "cpp_batch"}:
        raise ValueError(f"Unsupported backend={name!r}.")
    return name


def _project_full_box(
    q: FloatArray,
    order: PossibilityOrder,
    gaps: GapParameters,
    *,
    cfg: ProjectionConfig,
) -> DykstraResult:
    q = normalize_prob(q, cfg.log_clip_eps)
    backend = _resolve_backend(cfg.backend)

    if backend in {"cpp", "cpp_batch"}:
        return dykstra_kl_project_cpp(
            q=q,
            order=order,
            gaps=gaps,
            tau=float(cfg.tau),
            K_max=int(cfg.K_max),
            log_clip_eps=float(cfg.log_clip_eps),
            include_prefix_constraints=True,
            include_lower_constraints=True,
            include_upper_constraints=True,
        )

    system = build_linear_system(
        order,
        gaps,
        include_prefix_constraints=True,
        include_lower_constraints=True,
        include_upper_constraints=True,
    )
    constraints = build_constraint_family(
        order,
        gaps,
        include_prefix_constraints=True,
        include_lower_constraints=True,
        include_upper_constraints=True,
    )
    return dykstra_kl_project(
        q=q,
        constraints=constraints,
        system=system,
        tau=float(cfg.tau),
        K_max=int(cfg.K_max),
        log_clip_eps=float(cfg.log_clip_eps),
    )


def _get_order_gaps_from_sample(sample: Any) -> tuple[PossibilityOrder, GapParameters]:
    if hasattr(sample, "order") and hasattr(sample, "gaps"):
        return sample.order, sample.gaps

    if all(hasattr(sample, name) for name in ("sigma", "tilde_pi", "underline", "overline")):
        order = PossibilityOrder(
            sigma=np.asarray(sample.sigma, dtype=np.int64),
            tilde_pi=np.asarray(sample.tilde_pi, dtype=np.float64),
        )
        gaps = GapParameters(
            underline=np.asarray(sample.underline, dtype=np.float64),
            overline=np.asarray(sample.overline, dtype=np.float64),
        )
        return order, gaps

    raise TypeError(
        "Cannot recover KL-box order and gaps from sample. "
        "Expected either {order, gaps} or {sigma, tilde_pi, underline, overline}."
    )


def _project_batch_full_box(
    q_batch: FloatArray,
    orders: Sequence[PossibilityOrder],
    gaps_list: Sequence[GapParameters],
    cfg: ProjectionConfig,
) -> list[DykstraResult]:
    normalized = np.stack(
        [normalize_prob(row, cfg.log_clip_eps) for row in np.asarray(q_batch, dtype=np.float64)],
        axis=0,
    )
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


class ProjectionTarget:
    def __init__(
        self,
        tau: float = 1e-10,
        K_max: int = 300,
        log_clip_eps: float = 1e-15,
        engine: BackendName = "auto",
        *,
        n_threads: int = 0,
        identity_tol: float = 1e-6,
        **_: Any,  # pi_eps, tie_tol, eps_cap accepted but unused: order/gaps come from sample fields
    ) -> None:
        self.cfg = ProjectionConfig(
            tau=float(tau),
            K_max=int(K_max),
            log_clip_eps=float(log_clip_eps),
            backend=_resolve_backend(engine),
            n_threads=int(n_threads),
        )
        self.identity_tol = float(identity_tol)
        self._results: list[DykstraResult] = []
        self._l1_moves: list[float] = []

    def start_run(self, total_steps: int | None = None) -> None:
        del total_steps
        self._results = []
        self._l1_moves = []

    def __call__(self, q_batch: FloatArray, batch: Sequence[Any]) -> np.ndarray:
        q_batch = np.asarray(q_batch, dtype=np.float64)
        if q_batch.ndim != 2:
            raise ValueError(f"Expected q_batch.ndim == 2, got {q_batch.ndim}.")
        if q_batch.shape[0] != len(batch):
            raise ValueError(
                f"Batch size mismatch: q_batch has {q_batch.shape[0]} rows, but batch has length {len(batch)}."
            )

        targets = np.empty_like(q_batch, dtype=np.float64)
        backend = _resolve_backend(self.cfg.backend)

        q_rows: list[np.ndarray] = []
        orders: list[PossibilityOrder] = []
        gaps_list: list[GapParameters] = []

        for row_index, sample in enumerate(batch):
            q = normalize_prob(q_batch[row_index], self.cfg.log_clip_eps)
            order, gaps = _get_order_gaps_from_sample(sample)
            q_rows.append(q)
            orders.append(order)
            gaps_list.append(gaps)

        if backend == "cpp_batch":
            results = _project_batch_full_box(
                np.stack(q_rows, axis=0),
                orders,
                gaps_list,
                self.cfg,
            )
        else:
            results = [
                _project_full_box(
                    q=q_rows[i],
                    order=orders[i],
                    gaps=gaps_list[i],
                    cfg=self.cfg,
                )
                for i in range(len(q_rows))
            ]

        for row_index, result in enumerate(results):
            q = q_rows[row_index]
            p_star = normalize_prob(result.p_star, self.cfg.log_clip_eps)
            targets[row_index] = p_star
            self._results.append(result)
            self._l1_moves.append(float(np.sum(np.abs(p_star - q))))

        return targets

    def flush_stats(self) -> ProjectionStats:
        stats = projection_stats_from_results(self._results)
        self._results = []
        return stats

    def flush_diagnostics(self) -> dict[str, float]:
        moves = np.asarray(self._l1_moves, dtype=np.float64)
        self._l1_moves = []

        if moves.size == 0:
            return {}

        return {
            "l1_move_mean": float(np.mean(moves)),
            "l1_move_p90": float(np.quantile(moves, 0.9)),
            "identity_frac": float(np.mean(moves < self.identity_tol)),
        }


def target_B(q_batch: FloatArray, batch: Sequence[Any]) -> np.ndarray:
    del q_batch
    return stack_field(batch, "dot_p")


def target_C(q_batch: FloatArray, batch: Sequence[Any]) -> np.ndarray:
    del q_batch
    return stack_field(batch, "vote_p")