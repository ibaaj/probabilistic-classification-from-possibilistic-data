# ===== topk/data_splits.py =====
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from common.sampling import (
    deterministic_subset,
    resolve_subset_size,
    stratified_subset_by_label,
)
from topk.data import TopKConfig, TopKSample, make_topk_dataset


def make_train_val_from_seed(
    *,
    cfg: TopKConfig,
    N_train: int,
    N_val: int,
    seed_data: int,
) -> Tuple[List[TopKSample], List[TopKSample]]:
    rng_mu = np.random.default_rng(int(seed_data))
    mu = rng_mu.normal(size=(cfg.n_classes, cfg.d)).astype(np.float64) * float(cfg.class_sep)

    rng_train = np.random.default_rng(int(seed_data) + 1)
    rng_val = np.random.default_rng(int(seed_data) + 2)

    train = make_topk_dataset(cfg, N=int(N_train), rng=rng_train, mu=mu)
    val = make_topk_dataset(cfg, N=int(N_val), rng=rng_val, mu=mu)
    return train, val


def make_train_test_from_seed(
    *,
    cfg: TopKConfig,
    N_train: int,
    N_test: int,
    seed_data: int,
) -> Tuple[List[TopKSample], List[TopKSample]]:
    return make_train_val_from_seed(
        cfg=cfg,
        N_train=int(N_train),
        N_val=int(N_test),
        seed_data=int(seed_data),
    )