from __future__ import annotations

"""Deterministic split construction for ChaosNLI."""

import hashlib
from typing import Sequence

import numpy as np

from .constants import LABEL_TO_INDEX
from .schema import ChaosNLIRawItem



def _majority_class_index(item: ChaosNLIRawItem) -> int:
    majority_label = str(getattr(item, "majority_label", "")).strip().lower()
    if majority_label in LABEL_TO_INDEX:
        return int(LABEL_TO_INDEX[majority_label])
    return int(np.argmax(np.asarray(item.votes, dtype=np.int64)))



def _stable_rank(seed: int, key: str) -> str:
    payload = f"{int(seed)}\0{key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()



def split_raw_items(
    raw_items: Sequence[ChaosNLIRawItem],
    *,
    split_seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, list[ChaosNLIRawItem]]:
    """Build deterministic train, validation, and test splits.

    Stratification is by the majority-vote class. Within each class, ordering is
    determined by a stable hash of ``(split_seed, uid)``.
    """
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must lie in (0, 1).")
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError("val_frac must lie in (0, 1).")
    if float(train_frac) + float(val_frac) >= 1.0:
        raise ValueError("train_frac + val_frac must be strictly smaller than 1.")

    items = list(raw_items)
    if not items:
        return {"train": [], "validation": [], "test": []}

    seen = set()
    for item in items:
        if item.uid in seen:
            raise ValueError(f"Duplicate uid before splitting: {item.uid!r}")
        seen.add(item.uid)

    by_class: dict[int, list[ChaosNLIRawItem]] = {}
    for item in items:
        by_class.setdefault(_majority_class_index(item), []).append(item)

    train_items: list[ChaosNLIRawItem] = []
    validation_items: list[ChaosNLIRawItem] = []
    test_items: list[ChaosNLIRawItem] = []

    for cls in sorted(by_class):
        bucket = sorted(by_class[cls], key=lambda item: (_stable_rank(split_seed, item.uid), item.uid))
        n_bucket = len(bucket)
        n_train = int(np.floor(float(train_frac) * n_bucket))
        n_validation = int(np.floor(float(val_frac) * n_bucket))

        train_items.extend(bucket[:n_train])
        validation_items.extend(bucket[n_train : n_train + n_validation])
        test_items.extend(bucket[n_train + n_validation :])

    return {
        "train": sorted(train_items, key=lambda item: item.uid),
        "validation": sorted(validation_items, key=lambda item: item.uid),
        "test": sorted(test_items, key=lambda item: item.uid),
    }
