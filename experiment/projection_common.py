from __future__ import annotations

"""Shared projection utilities used by ChaosNLI targets."""

from typing import Any, Sequence

import numpy as np

from experiment.target_protocol import ProjectionStats
from klbox.dykstra import DykstraResult
from klbox.np_utils import clip_strictly_positive, normalize_to_simplex


def normalize_prob(x: np.ndarray, eps: float) -> np.ndarray:
    """Clip to a strictly positive vector and renormalize onto the simplex."""
    x = np.asarray(x, dtype=np.float64)
    x = clip_strictly_positive(x, float(eps))
    return normalize_to_simplex(x)


def empty_projection_stats() -> ProjectionStats:
    """Return the zero-valued projection-statistics record."""
    return ProjectionStats(
        calls=0,
        cycles_mean=0.0,
        cycles_p90=0.0,
        time_mean_s=0.0,
        finalV_mean=0.0,
        finalV_max=0.0,
    )


def projection_stats_from_results(results: Sequence[DykstraResult]) -> ProjectionStats:
    """Aggregate a sequence of Dykstra runs into one summary record."""
    if not results:
        return empty_projection_stats()

    cycles = np.asarray([float(result.cycles) for result in results], dtype=np.float64)
    times = np.asarray([float(result.elapsed_s) for result in results], dtype=np.float64)
    final_v = np.asarray([float(result.final_V) for result in results], dtype=np.float64)

    return ProjectionStats(
        calls=int(len(results)),
        cycles_mean=float(np.mean(cycles)),
        cycles_p90=float(np.quantile(cycles, 0.9)),
        time_mean_s=float(np.mean(times)),
        finalV_mean=float(np.mean(final_v)),
        finalV_max=float(np.max(final_v)),
    )


def stack_field(batch: Sequence[Any], name: str, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """Stack one named array-like field from a batch of samples."""
    rows = [np.asarray(getattr(sample, name), dtype=dtype) for sample in batch]
    if not rows:
        return np.zeros((0, 0), dtype=dtype)

    first_shape = rows[0].shape
    for i, row in enumerate(rows):
        if row.shape != first_shape:
            raise ValueError(
                f"All batch.{name} rows must have the same shape; "
                f"row 0 has shape {first_shape}, row {i} has shape {row.shape}."
            )
    return np.stack(rows, axis=0)
