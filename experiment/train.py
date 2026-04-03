from __future__ import annotations

import copy
import math
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiment.target_protocol import ProjectionStats, TargetFn

_TINY_POS = np.finfo(np.float64).tiny


def _has_projection_stats_interface(target_fn: Any) -> bool:
    return hasattr(target_fn, "start_run") and hasattr(target_fn, "flush_stats")


def _is_projection_stats(stats: Any) -> bool:
    required = ("calls", "cycles_mean", "cycles_p90", "time_mean_s", "finalV_mean", "finalV_max")
    return all(hasattr(stats, name) for name in required)


def _normalize_target_rows(target_np: np.ndarray) -> np.ndarray:
    target_np = np.asarray(target_np, dtype=np.float64)

    if target_np.ndim != 2:
        raise ValueError(
            f"target_fn must return a 2D array of shape [batch, C], got shape={target_np.shape}."
        )
    if not np.all(np.isfinite(target_np)):
        raise FloatingPointError("target_fn returned non-finite values.")
    if np.any(target_np < 0.0):
        raise ValueError("target_fn returned negative probabilities.")

    target_np = np.maximum(target_np, _TINY_POS)
    row_sums = np.sum(target_np, axis=1, keepdims=True)

    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise ValueError("target_fn returned rows with non-positive or non-finite sums.")

    return target_np / row_sums


def train_model(
    samples_train: List[Any],
    model: Any,
    optimizer: Any,
    target_fn: TargetFn,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    val_items: Optional[List[Any]] = None,
    scheduler: Optional[str] = None,
    no_progress: bool = False,
) -> Optional[ProjectionStats]:
    if len(samples_train) == 0:
        raise ValueError("samples_train must be non-empty.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive.")

    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    idx = np.arange(len(samples_train), dtype=np.int64)

    if _has_projection_stats_interface(target_fn):
        total_steps = int(epochs) * int(math.ceil(len(samples_train) / int(batch_size)))
        target_fn.start_run(total_steps=total_steps)

    sched = None
    if scheduler is None:
        sched = None
    elif scheduler == "cosine":
        base_lr = float(optimizer.param_groups[0]["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(epochs),
            eta_min=base_lr * 0.01,
        )
    else:
        raise ValueError(f"Unsupported scheduler={scheduler!r}. Use None or 'cosine'.")

    best_state: Optional[dict[str, Any]] = None
    best_val_acc = -1.0

    for _ in tqdm(range(int(epochs)), desc="epochs", disable=bool(no_progress)):
        np_rng.shuffle(idx)

        for s0 in range(0, idx.size, int(batch_size)):
            batch_idx = idx[s0 : s0 + int(batch_size)]
            batch = [samples_train[int(i)] for i in batch_idx]

            X_np = np.stack([np.asarray(s.x, dtype=np.float64) for s in batch], axis=0)
            X = torch.from_numpy(X_np).to(dtype=torch.float64)

            logits = model(X)
            log_q = F.log_softmax(logits, dim=1)
            q_np = torch.exp(log_q).detach().cpu().numpy().astype(np.float64)

            target_np = target_fn(q_np, batch)
            target_np = _normalize_target_rows(target_np)

            if target_np.shape != q_np.shape:
                raise ValueError(
                    f"target_fn returned shape={target_np.shape}, expected shape={q_np.shape}."
                )

            target = torch.from_numpy(target_np).to(
                dtype=log_q.dtype,
                device=log_q.device,
            )

            loss = F.kl_div(log_q, target, reduction="batchmean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if sched is not None:
            sched.step()

        if val_items is not None and len(val_items) > 0:
            X_val = np.stack([np.asarray(s.x, dtype=np.float64) for s in val_items], axis=0)
            y_val = np.asarray([int(s.y) for s in val_items], dtype=np.int64)

            probs = model.predict_proba(X_val)
            val_acc = float(np.mean(np.argmax(probs, axis=1) == y_val))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

    if val_items is not None and best_state is not None:
        model.load_state_dict(best_state)

    if _has_projection_stats_interface(target_fn):
        stats = target_fn.flush_stats()
        if stats is not None and not _is_projection_stats(stats):
            raise TypeError(
                f"target_fn.flush_stats() returned {type(stats).__name__}, expected ProjectionStats or None."
            )
        return stats

    return None
