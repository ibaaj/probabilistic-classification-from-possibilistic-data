from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from klbox.kl_types import FloatArray


TargetFn = Callable[[FloatArray, Sequence[Any]], np.ndarray]


@dataclass(frozen=True)
class ProjectionStats:
    calls: int
    cycles_mean: float
    cycles_p90: float
    time_mean_s: float
    finalV_mean: float
    finalV_max: float


class TargetWithStats(Protocol):
    def __call__(self, q_batch: FloatArray, batch: Sequence[Any]) -> np.ndarray: ...
    def start_run(self, total_steps: int | None = None) -> None: ...
    def flush_stats(self) -> ProjectionStats | None: ...
