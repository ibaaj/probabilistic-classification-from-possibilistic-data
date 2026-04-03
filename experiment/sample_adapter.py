from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from klbox.constraints import Constraint, build_constraint_family
from klbox.gaps import GapParameters
from klbox.linear_system import LinearSystem, build_linear_system
from klbox.possibility import PossibilityOrder


def _has_compact_sample(sample: Any) -> bool:
    return all(hasattr(sample, name) for name in ("sigma", "tilde_pi", "underline", "overline"))


def get_order_gaps(sample: Any) -> Tuple[PossibilityOrder, GapParameters]:
    if hasattr(sample, "order") and hasattr(sample, "gaps"):
        return sample.order, sample.gaps

    if _has_compact_sample(sample):
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
        "Sample is neither a full KL-box sample nor a compact NLP sample. "
        "Expected either {order,gaps,...} or {sigma,tilde_pi,underline,overline}."
    )


def get_system_constraints(sample: Any) -> Tuple[LinearSystem, List[Constraint]]:
    if hasattr(sample, "system") and hasattr(sample, "constraints"):
        return sample.system, sample.constraints

    order, gaps = get_order_gaps(sample)

    if hasattr(sample, "system"):
        system = sample.system
    else:
        system = build_linear_system(order, gaps)

    if hasattr(sample, "constraints"):
        constraints = sample.constraints
    else:
        constraints = build_constraint_family(order, gaps)

    return system, constraints