#!/usr/bin/env python3
from __future__ import annotations

"""Terminal-first audit for the synthetic top-k dataset.

This version is organized for CLI readability:

1. basic sample invariants
2. KL-box feasibility and tie-order invariance
3. independent oracle replay
4. cross-alpha protocol consistency
5. distribution sanity checks

The report begins with the exact configuration and one concrete sample artifact,
then prints compact PASS/FAIL sections with short numeric summaries.
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from klbox.gaps import choose_gap_parameters
from klbox.linear_system import build_linear_system, violation_V
from klbox.possibility import PossibilityOrder, antipignistic_reverse_mapping
from topk.data import TopKConfig, TopKSample, make_topk_dataset


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

STATUS_OK = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"


def status_text(ok: Optional[bool]) -> str:
    if ok is None:
        return STATUS_SKIP
    return STATUS_OK if ok else STATUS_FAIL


def fmt_float(value: float, *, digits: int = 4) -> str:
    x = float(value)
    if not math.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax < 1e-3 or ax >= 1e3:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def fmt_array(values: np.ndarray, *, limit: int = 6, digits: int = 4) -> str:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return fmt_float(float(arr), digits=digits)

    flat = arr.ravel()
    if flat.size == 0:
        return "[]"

    shown = flat[:limit]
    body = ", ".join(fmt_float(float(v), digits=digits) for v in shown)
    suffix = "" if flat.size <= limit else f", ... (len={flat.size})"
    return f"[{body}{suffix}]"


def fmt_int_array(values: np.ndarray, *, limit: int = 8) -> str:
    arr = np.asarray(values, dtype=np.int64).ravel()
    if arr.size == 0:
        return "[]"
    body = ", ".join(str(int(v)) for v in arr[:limit])
    suffix = "" if arr.size <= limit else f", ... (len={arr.size})"
    return f"[{body}{suffix}]"


def max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


# -----------------------------------------------------------------------------
# Report structures
# -----------------------------------------------------------------------------

@dataclass
class AuditCheck:
    title: str
    ok: Optional[bool]
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class AuditSection:
    title: str
    subtitle: str
    checks: list[AuditCheck] = field(default_factory=list)

    @property
    def ok(self) -> Optional[bool]:
        meaningful = [check.ok for check in self.checks if check.ok is not None]
        if not meaningful:
            return None
        return all(bool(x) for x in meaningful)


@dataclass
class AuditContext:
    args: argparse.Namespace
    cfg: TopKConfig
    mu_seed: int
    dataset_seed: int
    mu: np.ndarray
    samples: list[TopKSample]

    @property
    def n_samples(self) -> int:
        return int(len(self.samples))

    @property
    def C(self) -> int:
        return int(self.cfg.n_classes)

    @property
    def d(self) -> int:
        return int(self.cfg.d)


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------

def squared_distances(mu: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum((mu - x[None, :]) ** 2, axis=1)


def ordered_neighbors_by_distance(mu: np.ndarray, x: np.ndarray, y: int) -> np.ndarray:
    idx = np.arange(mu.shape[0], dtype=np.int64)
    mask = idx != int(y)
    idx_other = idx[mask]
    dist_other = squared_distances(mu, x)[mask]
    order = np.lexsort((idx_other, dist_other))
    return idx_other[order]


def stable_order_from_pi(pi: np.ndarray) -> PossibilityOrder:
    pi = np.asarray(pi, dtype=np.float64)
    idx = np.arange(pi.size, dtype=np.int64)
    sigma = np.lexsort((idx, -pi))
    tilde_pi = np.asarray(pi[sigma], dtype=np.float64)
    tilde_pi = tilde_pi / float(tilde_pi[0])
    return PossibilityOrder(sigma=sigma, tilde_pi=tilde_pi)


# -----------------------------------------------------------------------------
# Artifact preview
# -----------------------------------------------------------------------------

def config_lines(ctx: AuditContext) -> list[str]:
    cfg = ctx.cfg
    args = ctx.args
    return [
        f"n_classes={cfg.n_classes}, d={cfg.d}, requested_N={int(args.N)}, actual_N={ctx.n_samples}",
        f"alpha={fmt_float(cfg.alpha)}, alpha2={fmt_float(float(args.alpha2))}, alpha_noise={fmt_float(cfg.alpha_noise)}",
        f"class_sep={fmt_float(cfg.class_sep)}, x_noise={fmt_float(cfg.x_noise)}",
        f"pi_eps={fmt_float(cfg.pi_eps)}, stair_step={fmt_float(cfg.pi_stair_step)}, stair_m={int(cfg.pi_stair_m)}",
        f"tie_tol={fmt_float(cfg.tie_tol)}, eps_cap={fmt_float(cfg.eps_cap)}",
        f"mu_seed={ctx.mu_seed}, dataset_seed={ctx.dataset_seed} (compat base seed={int(args.seed)})",
    ]


def preview_sample_lines(ctx: AuditContext, sample_index: int) -> list[str]:
    if not ctx.samples:
        return ["dataset is empty"]

    idx = int(np.clip(sample_index, 0, len(ctx.samples) - 1))
    sample = ctx.samples[idx]

    x = np.asarray(sample.x, dtype=np.float64)
    pi = np.asarray(sample.pi, dtype=np.float64)
    dot_p = np.asarray(sample.dot_p, dtype=np.float64)
    sigma = np.asarray(sample.order.sigma, dtype=np.int64)
    tilde_pi = np.asarray(sample.order.tilde_pi, dtype=np.float64)
    underline = np.asarray(sample.gaps.underline, dtype=np.float64)
    overline = np.asarray(sample.gaps.overline, dtype=np.float64)

    distances = squared_distances(ctx.mu, x)
    ranked_neighbors = ordered_neighbors_by_distance(ctx.mu, x, int(sample.y))
    violation = float(violation_V(sample.system, dot_p))

    top_pi_order = np.lexsort((np.arange(pi.size, dtype=np.int64), -pi))
    top_pi_pairs = [f"{int(j)}:{fmt_float(pi[j])}" for j in top_pi_order[: min(6, pi.size)]]
    top_dist_pairs = [f"{int(j)}:{fmt_float(distances[j])}" for j in ranked_neighbors[: min(6, ranked_neighbors.size)]]

    return [
        f"sample[{idx}] -> y={int(sample.y)}, ||x||_2={fmt_float(np.linalg.norm(x))}, sum(dot_p)={fmt_float(np.sum(dot_p), digits=6)}",
        f"x[:6]           = {fmt_array(x, limit=6)}",
        f"top pi classes   = [{', '.join(top_pi_pairs)}]",
        f"nearest rivals   = [{', '.join(top_dist_pairs)}]",
        f"sigma[:8]        = {fmt_int_array(sigma[:8])}",
        f"tilde_pi[:8]     = {fmt_array(tilde_pi[:8], limit=8)}",
        f"dot_p[:8]        = {fmt_array(dot_p[:8], limit=8)}",
        f"underline[:8]    = {fmt_array(underline[:8], limit=8, digits=6)}",
        f"overline[:8]     = {fmt_array(overline[:8], limit=8, digits=10)}",
        f"violation_V      = {fmt_float(violation, digits=6)}",
    ]


# -----------------------------------------------------------------------------
# Independent oracle replay
# -----------------------------------------------------------------------------

def oracle_build_pi(
    *,
    y: int,
    ordered_neighbors: np.ndarray,
    C: int,
    alpha_x: float,
    pi_eps: float,
    stair_step: float,
    stair_m: int,
) -> np.ndarray:
    pi = np.zeros(C, dtype=np.float64)
    pi[int(y)] = 1.0

    k_minus_1 = int(ordered_neighbors.size)
    m = k_minus_1 if int(stair_m) <= 0 else min(k_minus_1, int(stair_m))
    floor = 0.0

    if m <= 0:
        last_level = floor
    else:
        last_level = max(float(alpha_x) - float(m - 1) * float(stair_step), floor)

    for r in range(k_minus_1):
        j = int(ordered_neighbors[r])
        if r < m:
            level = max(float(alpha_x) - float(r) * float(stair_step), floor)
        else:
            level = last_level
        pi[j] = level

    pi = pi + float(pi_eps)
    pi[int(y)] = 1.0
    cap = 1.0 - float(pi_eps)
    pi[ordered_neighbors] = np.minimum(pi[ordered_neighbors], cap)
    return pi


def make_topk_dataset_oracle(
    cfg: TopKConfig,
    N: int,
    rng: np.random.Generator,
    mu: np.ndarray,
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    C = int(cfg.n_classes)
    d = int(cfg.d)
    out: list[tuple[np.ndarray, int, np.ndarray]] = []

    for _ in range(int(N)):
        y = int(rng.integers(0, C))
        x = mu[y] + float(cfg.x_noise) * rng.normal(size=(d,)).astype(np.float64)
        ordered_neighbors = ordered_neighbors_by_distance(mu, x, y=y)

        eta = float(rng.normal())
        alpha_x = float(cfg.alpha) + float(cfg.alpha_noise) * eta
        alpha_x = float(np.clip(alpha_x, 0.0, 1.0 - float(cfg.pi_eps)))

        pi = oracle_build_pi(
            y=y,
            ordered_neighbors=ordered_neighbors,
            C=C,
            alpha_x=alpha_x,
            pi_eps=float(cfg.pi_eps),
            stair_step=float(cfg.pi_stair_step),
            stair_m=int(cfg.pi_stair_m),
        )
        out.append((x, y, pi))

    return out


# -----------------------------------------------------------------------------
# Validation sections
# -----------------------------------------------------------------------------

def validate_basic_invariants(ctx: AuditContext) -> AuditSection:
    samples = ctx.samples
    cfg = ctx.cfg
    expected_n = int(ctx.args.N)
    C = int(cfg.n_classes)
    d = int(cfg.d)

    section = AuditSection(
        title="1) Basic sample invariants",
        subtitle="Shapes, ranges, possibility normalization, and simplex validity.",
    )

    if not samples:
        section.checks.append(AuditCheck("dataset non-empty", False, "no samples were generated"))
        return section

    section.checks.append(
        AuditCheck("dataset size", len(samples) == expected_n, f"requested N={expected_n}, actual={len(samples)}")
    )

    shapes_ok = True
    labels_ok = True
    x_finite_ok = True
    pi_finite_ok = True
    pi_positive_ok = True
    pi_max_ok = True
    pi_y_ok = True
    pi_bounds_ok = True
    dotp_ok = True
    rank_ok = True

    worst_pi_max_dev = 0.0
    worst_dotp_sum_dev = 0.0
    worst_dotp_min = 0.0
    worst_rank_diff = 0.0

    for sample in samples:
        x = np.asarray(sample.x, dtype=np.float64)
        pi = np.asarray(sample.pi, dtype=np.float64)
        dot_p = np.asarray(sample.dot_p, dtype=np.float64)
        y = int(sample.y)

        if x.shape != (d,) or pi.shape != (C,) or dot_p.shape != (C,):
            shapes_ok = False
        if not (0 <= y < C):
            labels_ok = False
        if not np.all(np.isfinite(x)):
            x_finite_ok = False
        if not np.all(np.isfinite(pi)):
            pi_finite_ok = False
        if not np.all(pi > 0.0):
            pi_positive_ok = False

        max_dev = abs(float(np.max(pi)) - 1.0)
        worst_pi_max_dev = max(worst_pi_max_dev, max_dev)
        if max_dev > 1e-12:
            pi_max_ok = False

        if abs(float(pi[y]) - 1.0) > 1e-12:
            pi_y_ok = False

        mask = np.ones(C, dtype=bool)
        mask[y] = False
        if np.any(pi[mask] < float(cfg.pi_eps) - 1e-15) or np.any(pi[mask] > (1.0 - float(cfg.pi_eps)) + 1e-15):
            pi_bounds_ok = False

        if np.any(dot_p < -1e-12):
            dotp_ok = False
        dotp_sum_dev = abs(float(np.sum(dot_p)) - 1.0)
        worst_dotp_sum_dev = max(worst_dotp_sum_dev, dotp_sum_dev)
        worst_dotp_min = min(worst_dotp_min, float(np.min(dot_p)))
        if dotp_sum_dev > 1e-10:
            dotp_ok = False

        ranked = ordered_neighbors_by_distance(ctx.mu, x, y)
        ranked_pi = pi[ranked]
        if ranked_pi.size >= 2:
            diffs = ranked_pi[:-1] - ranked_pi[1:]
            min_diff = float(np.min(diffs))
            worst_rank_diff = min(worst_rank_diff, min_diff)
            if np.any(diffs < -1e-15):
                rank_ok = False

    sample0 = samples[0]
    section.checks.extend(
        [
            AuditCheck("sample shapes", shapes_ok, f"expected x:({d},), pi:({C},), dot_p:({C},)"),
            AuditCheck("label range", labels_ok, f"all labels should lie in [0, {C - 1}]"),
            AuditCheck("x finite", x_finite_ok, "all feature values are finite"),
            AuditCheck("pi finite", pi_finite_ok, "all possibility values are finite"),
            AuditCheck("pi strictly positive", pi_positive_ok, f"min(pi) on sample[0]={fmt_float(np.min(sample0.pi), digits=6)}"),
            AuditCheck("max(pi)=1", pi_max_ok, f"worst |max(pi)-1|={fmt_float(worst_pi_max_dev, digits=6)}"),
            AuditCheck("pi_y=1", pi_y_ok, f"sample[0]: y={int(sample0.y)}, pi[y]={fmt_float(float(sample0.pi[int(sample0.y)]), digits=6)}"),
            AuditCheck(
                "non-label pi bounds",
                pi_bounds_ok,
                f"expected pi_j in [{fmt_float(cfg.pi_eps, digits=6)}, {fmt_float(1.0 - cfg.pi_eps, digits=6)}] for j!=y",
            ),
            AuditCheck("distance ranking monotonicity", rank_ok, f"worst min consecutive ranked-pi diff={fmt_float(worst_rank_diff, digits=6)}"),
            AuditCheck(
                "dot_p is simplex-like",
                dotp_ok,
                f"worst |sum(dot_p)-1|={fmt_float(worst_dotp_sum_dev, digits=6)}, worst min(dot_p)={fmt_float(worst_dotp_min, digits=6)}",
            ),
        ]
    )
    return section


def validate_gap_box_and_ties(ctx: AuditContext, *, check_tie_order_invariance: bool) -> AuditSection:
    section = AuditSection(
        title="2) KL-box and tie handling",
        subtitle="Feasibility of the constructed box and invariance under tie permutations.",
    )

    if not ctx.samples:
        section.checks.append(AuditCheck("dataset non-empty", False, "no samples were generated"))
        return section

    tie_tol = float(ctx.cfg.tie_tol)
    eps_cap = float(ctx.cfg.eps_cap)

    equal_gap_ok = True
    feasible_ok = True
    tie_invariance_ok = True

    samples_with_equal = 0
    total_equal_adj = 0
    total_strict_adj = 0
    worst_violation = 0.0
    first_tie_failure = ""

    for sample in ctx.samples:
        diffs = np.asarray(sample.order.tilde_pi[:-1] - sample.order.tilde_pi[1:], dtype=np.float64)
        is_equal = np.abs(diffs) <= tie_tol
        is_strict = diffs > tie_tol

        n_equal = int(np.sum(is_equal))
        if n_equal > 0:
            samples_with_equal += 1
        total_equal_adj += n_equal
        total_strict_adj += int(np.sum(is_strict))

        underline = np.asarray(sample.gaps.underline, dtype=np.float64)
        overline = np.asarray(sample.gaps.overline, dtype=np.float64)
        if np.any(underline[is_equal] != 0.0) or np.any(overline[is_equal] != 0.0):
            equal_gap_ok = False

        v = float(violation_V(sample.system, sample.dot_p))
        worst_violation = max(worst_violation, v)
        if v > 1e-10:
            feasible_ok = False

        if check_tie_order_invariance:
            stable_order = stable_order_from_pi(sample.pi)
            anti_prod = antipignistic_reverse_mapping(sample.order)
            anti_stable = antipignistic_reverse_mapping(stable_order)

            gaps_stable = choose_gap_parameters(
                tilde_pi=stable_order.tilde_pi,
                dot_g=anti_stable.dot_g,
                tie_tol=tie_tol,
                eps_cap=eps_cap,
            )
            system_stable = build_linear_system(stable_order, gaps_stable)

            ok = (
                np.allclose(sample.order.tilde_pi, stable_order.tilde_pi, atol=1e-12, rtol=0.0)
                and np.allclose(anti_prod.dot_p, anti_stable.dot_p, atol=1e-12, rtol=0.0)
                and np.allclose(sample.gaps.underline, gaps_stable.underline, atol=1e-12, rtol=0.0)
                and np.allclose(sample.gaps.overline, gaps_stable.overline, atol=1e-12, rtol=0.0)
                and float(violation_V(system_stable, sample.dot_p)) <= 1e-10
            )
            if not ok:
                tie_invariance_ok = False
                if not first_tie_failure:
                    first_tie_failure = "stable tie-order reconstruction changed a meaningful derived object"

    n = max(ctx.n_samples, 1)
    section.checks.append(
        AuditCheck(
            "equal adjacent tilde_pi => zero gap bounds",
            equal_gap_ok,
            (
                f"samples_with_equal_adjacent={samples_with_equal}/{ctx.n_samples}, "
                f"mean_equal_adjacent={fmt_float(total_equal_adj / n, digits=6)}, "
                f"mean_strict_adjacent={fmt_float(total_strict_adj / n, digits=6)}"
            ),
        )
    )
    section.checks.append(
        AuditCheck(
            "dot_p feasible for constructed box",
            feasible_ok,
            f"worst violation_V(system, dot_p)={fmt_float(worst_violation, digits=6)}",
        )
    )

    if check_tie_order_invariance:
        section.checks.append(
            AuditCheck(
                "tie-order invariance",
                tie_invariance_ok,
                "reordering equal pi values should not change tilde_pi, dot_p, or gap bounds"
                if tie_invariance_ok
                else first_tie_failure,
            )
        )
    else:
        section.checks.append(
            AuditCheck("tie-order invariance", None, "skipped by --skip-tie-order-invariance")
        )

    return section


def validate_oracle_replay(ctx: AuditContext, *, enabled: bool) -> AuditSection:
    section = AuditSection(
        title="3) Oracle replay",
        subtitle="Independent re-generation of (x, y, pi) from the documented construction.",
    )

    if not enabled:
        section.checks.append(AuditCheck("oracle replay", None, "skipped by --skip-oracle-replay"))
        return section

    oracle = make_topk_dataset_oracle(
        cfg=ctx.cfg,
        N=len(ctx.samples),
        rng=np.random.default_rng(int(ctx.dataset_seed)),
        mu=ctx.mu,
    )

    size_ok = len(oracle) == len(ctx.samples)
    y_ok = True
    x_ok = True
    pi_ok = True
    worst_x = 0.0
    worst_pi = 0.0

    for sample, (x_ref, y_ref, pi_ref) in zip(ctx.samples, oracle):
        if int(sample.y) != int(y_ref):
            y_ok = False
        dx = np.asarray(sample.x, dtype=np.float64) - np.asarray(x_ref, dtype=np.float64)
        dp = np.asarray(sample.pi, dtype=np.float64) - np.asarray(pi_ref, dtype=np.float64)
        worst_x = max(worst_x, max_abs(dx))
        worst_pi = max(worst_pi, max_abs(dp))
        if not np.allclose(sample.x, x_ref, rtol=0.0, atol=0.0):
            x_ok = False
        if not np.allclose(sample.pi, pi_ref, rtol=0.0, atol=0.0):
            pi_ok = False

    section.checks.extend(
        [
            AuditCheck("oracle size", size_ok, f"generated={len(ctx.samples)}, oracle={len(oracle)}"),
            AuditCheck("oracle labels y", y_ok, "independent replay matches all labels exactly"),
            AuditCheck("oracle features x", x_ok, f"max abs diff={fmt_float(worst_x, digits=6)}"),
            AuditCheck("oracle possibility pi", pi_ok, f"max abs diff={fmt_float(worst_pi, digits=6)}"),
        ]
    )
    return section


def validate_cross_alpha_same_xy(ctx: AuditContext, *, enabled: bool) -> AuditSection:
    section = AuditSection(
        title="4) Cross-alpha protocol property",
        subtitle="With fixed mu and dataset RNG, changing alpha should change pi only.",
    )

    if not enabled:
        section.checks.append(AuditCheck("cross-alpha check", None, "skipped by --skip-cross-alpha"))
        return section

    cfg_a = ctx.cfg
    cfg_b = build_cfg(ctx.args, alpha=float(ctx.args.alpha2))

    ds_a = make_topk_dataset(
        cfg_a,
        N=int(ctx.args.N),
        rng=np.random.default_rng(int(ctx.dataset_seed)),
        mu=ctx.mu,
    )
    ds_b = make_topk_dataset(
        cfg_b,
        N=int(ctx.args.N),
        rng=np.random.default_rng(int(ctx.dataset_seed)),
        mu=ctx.mu,
    )

    same_y = True
    same_x = True
    changed_pi = 0
    worst_x = 0.0

    for sa, sb in zip(ds_a, ds_b):
        if int(sa.y) != int(sb.y):
            same_y = False
        dx = np.asarray(sa.x, dtype=np.float64) - np.asarray(sb.x, dtype=np.float64)
        worst_x = max(worst_x, max_abs(dx))
        if not np.allclose(sa.x, sb.x, rtol=0.0, atol=0.0):
            same_x = False
        if not np.allclose(sa.pi, sb.pi, rtol=0.0, atol=0.0):
            changed_pi += 1

    section.checks.extend(
        [
            AuditCheck("cross-alpha same size", len(ds_a) == int(ctx.args.N) and len(ds_b) == int(ctx.args.N), f"len(alpha1)={len(ds_a)}, len(alpha2)={len(ds_b)}"),
            AuditCheck("cross-alpha same labels y", same_y, "same seeds should keep y fixed"),
            AuditCheck("cross-alpha same features x", same_x, f"max abs diff={fmt_float(worst_x, digits=6)}"),
            AuditCheck("cross-alpha pi changes", changed_pi > 0, f"pi changed on {changed_pi}/{int(ctx.args.N)} samples"),
        ]
    )
    return section


def validate_distribution_sanity(ctx: AuditContext, *, enabled: bool) -> AuditSection:
    section = AuditSection(
        title="5) Distribution sanity",
        subtitle="Loose statistical checks on labels and residual noise.",
    )

    if not enabled:
        section.checks.append(AuditCheck("distribution sanity", None, "skipped by --skip-distribution-checks"))
        return section

    if not ctx.samples:
        section.checks.append(AuditCheck("dataset non-empty", False, "no samples were generated"))
        return section

    X = np.stack([np.asarray(sample.x, dtype=np.float64) for sample in ctx.samples], axis=0)
    y = np.asarray([int(sample.y) for sample in ctx.samples], dtype=np.int64)

    C = int(ctx.cfg.n_classes)
    d = int(ctx.cfg.d)
    N = int(len(ctx.samples))

    counts = np.bincount(y, minlength=C).astype(np.float64)
    expected = float(N) / float(C)
    p = 1.0 / float(C)
    std_count = math.sqrt(float(N) * p * (1.0 - p))
    max_count_dev = float(np.max(np.abs(counts - expected))) if counts.size else 0.0
    labels_ok = max_count_dev < 6.0 * std_count + 1e-12

    residuals = X - ctx.mu[y]
    mean_per_dim = np.mean(residuals, axis=0)
    var_per_dim = np.var(residuals, axis=0, ddof=0)

    max_abs_mean = float(np.max(np.abs(mean_per_dim))) if mean_per_dim.size else 0.0
    mean_var = float(np.mean(var_per_dim)) if var_per_dim.size else 0.0
    target_var = float(ctx.cfg.x_noise) ** 2

    # Coordinate-wise residual means satisfy:
    #   mean_j ~ approximately N(0, x_noise^2 / N).
    # We therefore use a dimension-aware loose threshold for max_j |mean_j|.
    residual_mean_threshold = 3.5 * float(ctx.cfg.x_noise) * math.sqrt(math.log(max(d, 2)) / max(N, 1))
    mean_ok = max_abs_mean < residual_mean_threshold

    # This is only a smoke test, so keep the variance tolerance loose.
    var_tol = max(0.5, 0.20 * target_var)
    var_ok = abs(mean_var - target_var) < var_tol

    section.checks.extend(
        [
            AuditCheck(
                "label marginal ~ uniform",
                labels_ok,
                f"max |count-expected|={fmt_float(max_count_dev)}, expected={fmt_float(expected)}, 6σ={fmt_float(6.0 * std_count)}",
            ),
            AuditCheck(
                "residual mean near 0",
                mean_ok,
                f"max |mean residual dim|={fmt_float(max_abs_mean, digits=6)}, threshold={fmt_float(residual_mean_threshold, digits=6)}",
            ),
            AuditCheck(
                "residual variance near x_noise^2",
                var_ok,
                f"mean residual var={fmt_float(mean_var)}, target={fmt_float(target_var)}, tol={fmt_float(var_tol)}",
            ),
        ]
    )
    return section


# -----------------------------------------------------------------------------
# Report rendering
# -----------------------------------------------------------------------------

def print_kv_block(title: str, lines: Iterable[str]) -> None:
    print(title)
    print("-" * len(title))
    for line in lines:
        print(f"  {line}")
    print()


def print_section(section: AuditSection) -> None:
    print(section.title)
    print("-" * len(section.title))
    print(f"  {section.subtitle}")
    print()

    for check in section.checks:
        print(f"  [{status_text(check.ok)}] {check.title}")
        print(f"      {check.summary}")
        for detail in check.details:
            print(f"      {detail}")

    print()
    print(f"  Section result: {status_text(section.ok)}")
    print()


def print_final_summary(ctx: AuditContext, sections: Sequence[AuditSection]) -> None:
    meaningful = [section.ok for section in sections if section.ok is not None]
    overall_ok = all(bool(x) for x in meaningful) if meaningful else True

    print("Final summary")
    print("-------------")
    print(f"  requested N : {int(ctx.args.N)}")
    print(f"  actual size : {ctx.n_samples}")
    print(f"  sections    : {len(sections)}")
    for section in sections:
        print(f"    - {section.title}: {status_text(section.ok)}")
    print(f"  overall     : {status_text(overall_ok)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Readable terminal audit for the synthetic top-k dataset and its KL-box artifacts."
    )

    parser.add_argument("--n-classes", type=int, default=20)
    parser.add_argument("--d", type=int, default=30)
    parser.add_argument("--N", type=int, default=2000)

    parser.add_argument("--seed", type=int, default=0, help="compatibility seed: mu_seed=seed, dataset_seed=seed+1")
    parser.add_argument("--mu-seed", type=int, default=None)
    parser.add_argument("--dataset-seed", type=int, default=None)

    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--alpha2", type=float, default=0.95)
    parser.add_argument("--alpha-noise", type=float, default=0.15)
    parser.add_argument("--class-sep", type=float, default=1.5)
    parser.add_argument("--x-noise", type=float, default=2.0)
    parser.add_argument("--pi-eps", type=float, default=1e-6)
    parser.add_argument("--pi-stair-step", type=float, default=0.01)
    parser.add_argument("--pi-stair-m", type=int, default=0)
    parser.add_argument("--tie-tol", type=float, default=0.0)
    parser.add_argument("--eps-cap", type=float, default=1e-9)

    parser.add_argument("--sample-index", type=int, default=0, help="sample shown in the artifact preview block")
    parser.add_argument("--skip-tie-order-invariance", action="store_true")
    parser.add_argument("--skip-oracle-replay", action="store_true")
    parser.add_argument("--skip-cross-alpha", action="store_true")
    parser.add_argument("--skip-distribution-checks", action="store_true")

    return parser.parse_args()


def build_cfg(args: argparse.Namespace, *, alpha: float) -> TopKConfig:
    return TopKConfig(
        n_classes=int(args.n_classes),
        d=int(args.d),
        alpha=float(alpha),
        alpha_noise=float(args.alpha_noise),
        class_sep=float(args.class_sep),
        x_noise=float(args.x_noise),
        pi_eps=float(args.pi_eps),
        pi_stair_step=float(args.pi_stair_step),
        pi_stair_m=int(args.pi_stair_m),
        tie_tol=float(args.tie_tol),
        eps_cap=float(args.eps_cap),
    )


def resolve_seeds(args: argparse.Namespace) -> tuple[int, int]:
    mu_seed = int(args.seed) if args.mu_seed is None else int(args.mu_seed)
    dataset_seed = int(args.seed) + 1 if args.dataset_seed is None else int(args.dataset_seed)
    return mu_seed, dataset_seed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    mu_seed, dataset_seed = resolve_seeds(args)
    cfg = build_cfg(args, alpha=float(args.alpha))

    rng_mu = np.random.default_rng(int(mu_seed))
    mu = rng_mu.normal(size=(cfg.n_classes, cfg.d)).astype(np.float64) * float(cfg.class_sep)

    rng_ds = np.random.default_rng(int(dataset_seed))
    samples = make_topk_dataset(cfg, N=int(args.N), rng=rng_ds, mu=mu)

    ctx = AuditContext(
        args=args,
        cfg=cfg,
        mu_seed=mu_seed,
        dataset_seed=dataset_seed,
        mu=mu,
        samples=list(samples),
    )

    print("Synthetic top-k dataset audit")
    print("=============================")
    print("This report checks generation correctness and shows one concrete sample artifact.")
    print()

    print_kv_block("Configuration", config_lines(ctx))
    print_kv_block("Artifact preview", preview_sample_lines(ctx, int(args.sample_index)))

    sections = [
        validate_basic_invariants(ctx),
        validate_gap_box_and_ties(ctx, check_tie_order_invariance=(not bool(args.skip_tie_order_invariance))),
        validate_oracle_replay(ctx, enabled=(not bool(args.skip_oracle_replay))),
        validate_cross_alpha_same_xy(ctx, enabled=(not bool(args.skip_cross_alpha))),
        validate_distribution_sanity(ctx, enabled=(not bool(args.skip_distribution_checks))),
    ]

    for section in sections:
        print_section(section)

    print_final_summary(ctx, sections)

    meaningful = [section.ok for section in sections if section.ok is not None]
    overall_ok = all(bool(x) for x in meaningful) if meaningful else True
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())