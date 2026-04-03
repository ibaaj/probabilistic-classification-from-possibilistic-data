from __future__ import annotations

"""Shared deterministic sampling helpers.

This module centralizes the sampling utilities that were previously duplicated
across the synthetic top-k code and the NLP loaders.
"""

from typing import Any

import numpy as np


def resolve_subset_size(*, total_size: int, explicit_size: int, frac: float) -> int:
    """Resolve the effective subset size from either an explicit size or a fraction."""
    if total_size <= 0:
        raise ValueError("total_size must be positive.")
    if explicit_size < 0:
        raise ValueError("explicit_size must be >= 0.")
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must lie in (0, 1].")

    if explicit_size > 0:
        return min(total_size, explicit_size)

    if frac >= 1.0:
        return total_size

    subset_size = int(np.ceil(frac * float(total_size)))
    subset_size = max(1, subset_size)
    return min(total_size, subset_size)


def deterministic_subset(items: list[Any], subset_size: int, seed: int) -> list[Any]:
    """Select a deterministic random subset without replacement."""
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if subset_size >= len(items):
        return list(items)

    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(items), dtype=np.int64)
    rng.shuffle(indices)
    chosen = np.sort(indices[: int(subset_size)])
    return [items[int(i)] for i in chosen]


def stratified_subset_by_label(
    items: list[Any],
    *,
    subset_size: int,
    seed: int,
    label_attr: str = "y",
) -> list[Any]:
    """Select a deterministic stratified subset.

    The allocation preserves class proportions as closely as possible and keeps
    at least one example per class when the requested subset is large enough.
    If the requested label attribute is missing, the function falls back to a
    plain deterministic subset.
    """
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if subset_size >= len(items):
        return list(items)
    if not items:
        return []
    if not all(hasattr(item, label_attr) for item in items):
        return deterministic_subset(items, subset_size=subset_size, seed=seed)

    labels = np.asarray([int(getattr(item, label_attr)) for item in items], dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)

    quotas = subset_size * counts.astype(np.float64) / float(len(items))
    allocation = np.floor(quotas).astype(np.int64)

    if subset_size >= int(classes.size):
        allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts.astype(np.int64))

    while int(allocation.sum()) > subset_size:
        for idx in np.argsort(allocation - quotas, kind="stable")[::-1]:
            floor_min = 1 if subset_size >= int(classes.size) else 0
            if allocation[idx] > floor_min:
                allocation[idx] -= 1
                if int(allocation.sum()) == subset_size:
                    break

    while int(allocation.sum()) < subset_size:
        for idx in np.argsort(quotas - allocation, kind="stable")[::-1]:
            if allocation[idx] < counts[idx]:
                allocation[idx] += 1
                if int(allocation.sum()) == subset_size:
                    break

    rng = np.random.default_rng(int(seed))
    chosen_indices: list[int] = []

    for cls, take in zip(classes.tolist(), allocation.tolist()):
        if take <= 0:
            continue
        class_indices = np.flatnonzero(labels == int(cls)).astype(np.int64)
        rng.shuffle(class_indices)
        chosen_indices.extend(int(i) for i in np.sort(class_indices[: int(take)]))

    chosen_indices = sorted(set(chosen_indices))

    if len(chosen_indices) < subset_size:
        chosen_set = set(chosen_indices)
        remaining = [i for i in range(len(items)) if i not in chosen_set]
        rng.shuffle(remaining)
        chosen_indices.extend(sorted(remaining[: subset_size - len(chosen_indices)]))
    elif len(chosen_indices) > subset_size:
        chosen_indices = chosen_indices[:subset_size]

    return [items[i] for i in chosen_indices]