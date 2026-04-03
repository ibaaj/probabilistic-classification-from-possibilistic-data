from __future__ import annotations

"""Shared TopK configuration builders."""

from topk.data import TopKConfig


def build_topk_config(
    *,
    n_classes: int,
    d: int,
    alpha: float,
    pi_eps: float,
    alpha_noise: float,
    class_sep: float,
    x_noise: float,
    tie_tol: float,
    eps_cap: float,
    pi_stair_step: float,
    pi_stair_m: int,
) -> TopKConfig:
    """Construct a TopKConfig from CLI-style scalar arguments."""
    return TopKConfig(
        n_classes=int(n_classes),
        d=int(d),
        alpha=float(alpha),
        pi_eps=float(pi_eps),
        alpha_noise=float(alpha_noise),
        class_sep=float(class_sep),
        x_noise=float(x_noise),
        tie_tol=float(tie_tol),
        eps_cap=float(eps_cap),
        pi_stair_step=float(pi_stair_step),
        pi_stair_m=int(pi_stair_m),
    )