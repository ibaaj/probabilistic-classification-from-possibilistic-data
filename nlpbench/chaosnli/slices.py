from __future__ import annotations

"""Canonical ChaosNLI slice definitions and protocol helpers.

Protocol:
- derive ambiguity statistics from vote counts only
- compute thresholds from the unique-majority subset of train_full only
- reuse those fixed thresholds unchanged for train slices, validation, and test
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

SELECTION_SPLITS: tuple[str, ...] = (
    "val_full",
    "val_S_amb",
    "val_S_easy",
)

VALIDATION_SECTION_ORDER: tuple[str, ...] = (
    "val_full",
    "val_S_amb",
    "val_S_easy",
)

TEST_SECTION_ORDER: tuple[str, ...] = (
    "test_full",
    "test_S_amb",
    "test_S_easy",
)

SECTION_TO_SLICE_KEY = {
    "val_full": None,
    "val_S_amb": "S_amb",
    "val_S_easy": "S_easy",
    "test_full": None,
    "test_S_amb": "S_amb",
    "test_S_easy": "S_easy",
}

TEST_SECTION_FOR_SELECTION_SPLIT = {
    "val_full": "test_full",
    "val_S_amb": "test_S_amb",
    "val_S_easy": "test_S_easy",
}


@dataclass(frozen=True)
class ChaosNLISliceStats:
    """Vote-derived ambiguity statistics for one sample or processed item."""

    sample_id: str
    subset: str
    y: int
    n_raters: int
    support: int
    top_votes: int
    second_votes: int
    top_margin: int
    margin_rate: float
    peak: float
    entropy: float
    Hnorm: float
    unique_top: bool


def _safe_entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    mask = p > 0.0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _vote_probs(votes: np.ndarray, y: int, c: int) -> np.ndarray:
    votes = np.asarray(votes, dtype=np.float64)
    total = float(np.sum(votes))
    if total > 0.0:
        return votes / total

    out = np.zeros(int(c), dtype=np.float64)
    out[int(y)] = 1.0
    return out


def _sample_identifier(sample: Any, idx: int) -> str:
    for name in ("sample_id", "uid", "comment_id", "id"):
        if hasattr(sample, name):
            return str(getattr(sample, name))
    return str(idx)


def _subset_from_sample_id(sample_id: str) -> str:
    sid = str(sample_id)
    return sid.split("::", 1)[0] if "::" in sid else ""


def normalize_selection_split(value: str) -> str:
    name = str(value).strip()
    if name not in SELECTION_SPLITS:
        allowed = ", ".join(SELECTION_SPLITS)
        raise ValueError(f"Unsupported selection_split={value!r}. Allowed values: {allowed}.")
    return name


def test_section_for_selection_split(selection_split: str) -> str:
    normalized = normalize_selection_split(selection_split)
    return TEST_SECTION_FOR_SELECTION_SPLIT[normalized]


def section_to_slice_key(section_name: str) -> str | None:
    name = str(section_name).strip()
    if name not in SECTION_TO_SLICE_KEY:
        allowed = ", ".join(sorted(SECTION_TO_SLICE_KEY))
        raise ValueError(f"Unsupported section_name={section_name!r}. Allowed values: {allowed}.")
    return SECTION_TO_SLICE_KEY[name]


def compute_slice_stats(sample: Any, *, sample_index: int, n_classes: int) -> ChaosNLISliceStats:
    c = int(n_classes)
    if c <= 0:
        raise ValueError("n_classes must be positive.")

    sample_id = _sample_identifier(sample, sample_index)
    votes = np.asarray(sample.votes, dtype=np.float64)
    y = int(sample.y)
    n_raters = int(sample.n_raters)
    top_votes = int(sample.top_votes)
    second_votes = int(sample.second_votes)
    top_margin = int(sample.top_margin)

    probs = _vote_probs(votes, y, c)
    peak = float(np.max(probs)) if probs.size else 0.0
    entropy = _safe_entropy_from_probs(probs)
    log_c = np.log(float(c)) if c > 1 else 1.0
    hnorm = float(entropy / log_c) if c > 1 else 0.0

    return ChaosNLISliceStats(
        sample_id=sample_id,
        subset=_subset_from_sample_id(sample_id),
        y=y,
        n_raters=n_raters,
        support=int(np.sum(votes > 0.0)),
        top_votes=top_votes,
        second_votes=second_votes,
        top_margin=top_margin,
        margin_rate=float(top_margin / max(n_raters, 1)),
        peak=peak,
        entropy=entropy,
        Hnorm=hnorm,
        unique_top=bool(top_votes > second_votes),
    )


def compute_slice_stats_for_split(samples: Sequence[Any], *, n_classes: int) -> list[ChaosNLISliceStats]:
    return [
        compute_slice_stats(sample, sample_index=i, n_classes=int(n_classes))
        for i, sample in enumerate(samples)
    ]


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    probs = [0.0, 0.10, 0.25, 0.50, 0.70, 0.75, 0.90, 1.0]
    out: dict[str, float] = {}
    if values.size == 0:
        for q in probs:
            out[f"q{int(round(100 * q)):02d}"] = float("nan")
        return out

    for q in probs:
        out[f"q{int(round(100 * q)):02d}"] = float(np.quantile(values, q))
    return out


def _decile_edges(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    return [float(np.quantile(values, q)) for q in np.linspace(0.1, 0.9, 9)]


def compute_slice_thresholds(train_stats: Sequence[ChaosNLISliceStats]) -> dict[str, Any]:
    train_stats = list(train_stats)
    if not train_stats:
        raise ValueError("Cannot compute ChaosNLI slice thresholds from an empty train_full split.")

    reference_stats = [row for row in train_stats if bool(row.unique_top)]
    if not reference_stats:
        raise ValueError(
            "Cannot compute ChaosNLI slice thresholds: train_full contains no unique-majority items."
        )

    peak = np.asarray([float(row.peak) for row in reference_stats], dtype=np.float64)
    hnorm = np.asarray([float(row.Hnorm) for row in reference_stats], dtype=np.float64)

    return {
        "reference_split": "train_full",
        "reference_subset": "unique_majority",
        "n_reference": int(len(reference_stats)),
        "T_low_peak": float(np.quantile(peak, 0.30)),
        "T_high_peak": float(np.quantile(peak, 0.70)),
        "T_low_H": float(np.quantile(hnorm, 0.30)),
        "T_high_H": float(np.quantile(hnorm, 0.70)),
        "peak_deciles": _decile_edges(peak),
        "Hnorm_deciles": _decile_edges(hnorm),
        "peak_quantiles": _quantile_summary(peak),
        "Hnorm_quantiles": _quantile_summary(hnorm),
    }


def slice_membership_for_stats(stats: ChaosNLISliceStats, thresholds: dict[str, Any]) -> dict[str, bool]:
    t_low_peak = float(thresholds["T_low_peak"])
    t_high_peak = float(thresholds["T_high_peak"])
    t_low_h = float(thresholds["T_low_H"])
    t_high_h = float(thresholds["T_high_H"])

    low_peak = float(stats.peak) <= t_low_peak
    high_peak = float(stats.peak) >= t_high_peak
    low_h = float(stats.Hnorm) <= t_low_h
    high_h = float(stats.Hnorm) >= t_high_h
    unique_top = bool(stats.unique_top)

    return {
        "anchor_clean": unique_top,
        "low_peak_high_H": low_peak and high_h,
        "low_peak_low_H": low_peak and low_h,
        "high_peak_high_H": high_peak and high_h,
        "high_peak_low_H": high_peak and low_h,
        "S_amb": unique_top and low_peak and high_h,
        "S_easy": unique_top and high_peak and low_h,
    }


def slice_masks_from_stats(
    stats: Sequence[ChaosNLISliceStats],
    thresholds: dict[str, Any],
) -> dict[str, np.ndarray]:
    stats = list(stats)
    masks = {
        "S_full": np.ones(len(stats), dtype=bool),
        "anchor_clean": np.zeros(len(stats), dtype=bool),
        "low_peak_high_H": np.zeros(len(stats), dtype=bool),
        "low_peak_low_H": np.zeros(len(stats), dtype=bool),
        "high_peak_high_H": np.zeros(len(stats), dtype=bool),
        "high_peak_low_H": np.zeros(len(stats), dtype=bool),
        "S_amb": np.zeros(len(stats), dtype=bool),
        "S_easy": np.zeros(len(stats), dtype=bool),
    }

    for i, row in enumerate(stats):
        flags = slice_membership_for_stats(row, thresholds)
        for key, value in flags.items():
            masks[key][i] = bool(value)
    return masks


def slice_masks_from_samples(
    samples: Sequence[Any],
    *,
    thresholds: dict[str, Any],
    n_classes: int,
) -> dict[str, np.ndarray]:
    stats = compute_slice_stats_for_split(samples, n_classes=int(n_classes))
    return slice_masks_from_stats(stats, thresholds)


def build_protocol_sections(
    full_samples: Sequence[Any],
    *,
    split_prefix: str,
    thresholds: dict[str, Any],
    n_classes: int,
) -> dict[str, list[Any]]:
    prefix = str(split_prefix).strip().lower()
    if prefix not in {"val", "test"}:
        raise ValueError(f"split_prefix must be 'val' or 'test', got {split_prefix!r}.")

    samples = list(full_samples)
    masks = slice_masks_from_samples(samples, thresholds=thresholds, n_classes=int(n_classes))

    return {
        f"{prefix}_full": list(samples),
        f"{prefix}_S_amb": [sample for sample, keep in zip(samples, masks["S_amb"]) if bool(keep)],
        f"{prefix}_S_easy": [sample for sample, keep in zip(samples, masks["S_easy"]) if bool(keep)],
    }


def selection_section_samples(
    full_samples: Sequence[Any],
    *,
    section_name: str,
    thresholds: dict[str, Any],
    n_classes: int,
) -> list[Any]:
    slice_key = section_to_slice_key(section_name)
    samples = list(full_samples)
    if slice_key is None:
        return samples

    masks = slice_masks_from_samples(samples, thresholds=thresholds, n_classes=int(n_classes))
    return [sample for sample, keep in zip(samples, masks[slice_key]) if bool(keep)]


def slice_counts(stats: Sequence[ChaosNLISliceStats], thresholds: dict[str, Any]) -> dict[str, int]:
    masks = slice_masks_from_stats(stats, thresholds)
    return {
        "n": int(len(list(stats))),
        "anchor_clean": int(np.sum(masks["anchor_clean"])),
        "low_peak_high_H": int(np.sum(masks["low_peak_high_H"])),
        "low_peak_low_H": int(np.sum(masks["low_peak_low_H"])),
        "high_peak_high_H": int(np.sum(masks["high_peak_high_H"])),
        "high_peak_low_H": int(np.sum(masks["high_peak_low_H"])),
        "S_amb": int(np.sum(masks["S_amb"])),
        "S_easy": int(np.sum(masks["S_easy"])),
    }


def slice_stats_rows(stats: Sequence[ChaosNLISliceStats], *, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stats:
        rows.append(
            {
                "split": str(split_name),
                "sample_id": row.sample_id,
                "subset": row.subset,
                "y": int(row.y),
                "n_raters": int(row.n_raters),
                "support": int(row.support),
                "top_votes": int(row.top_votes),
                "second_votes": int(row.second_votes),
                "top_margin": int(row.top_margin),
                "margin_rate": float(row.margin_rate),
                "peak": float(row.peak),
                "entropy": float(row.entropy),
                "Hnorm": float(row.Hnorm),
                "unique_top": int(bool(row.unique_top)),
            }
        )
    return rows


def slice_stats_summary(stats: Sequence[ChaosNLISliceStats]) -> dict[str, Any]:
    stats = list(stats)
    if not stats:
        empty = np.asarray([], dtype=np.float64)
        return {
            "n": 0,
            "peak_quantiles": _quantile_summary(empty),
            "Hnorm_quantiles": _quantile_summary(empty),
            "margin_rate_quantiles": _quantile_summary(empty),
        }

    peak = np.asarray([float(row.peak) for row in stats], dtype=np.float64)
    hnorm = np.asarray([float(row.Hnorm) for row in stats], dtype=np.float64)
    margin_rate = np.asarray([float(row.margin_rate) for row in stats], dtype=np.float64)

    return {
        "n": int(len(stats)),
        "peak_quantiles": _quantile_summary(peak),
        "Hnorm_quantiles": _quantile_summary(hnorm),
        "margin_rate_quantiles": _quantile_summary(margin_rate),
    }