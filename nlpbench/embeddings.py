from __future__ import annotations

"""Cached transformer text embeddings.

This module is intentionally generic. It knows nothing about ChaosNLI or any
other dataset; it only turns a mapping

    split name -> iterable of records with {id, text}

into two cache files per split:

- ``{split}_ids.npy``
- ``{split}_embs.npy``

The cache format matches the simple convention already used in the existing
project snapshot.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class EmbeddingCachePaths:
    """Concrete file paths for one cached split."""

    ids_path: Path
    embs_path: Path



def cache_paths(emb_dir: str | Path, split: str) -> EmbeddingCachePaths:
    """Return the standard cache file paths for one split."""
    root = Path(emb_dir)
    return EmbeddingCachePaths(
        ids_path=root / f"{split}_ids.npy",
        embs_path=root / f"{split}_embs.npy",
    )



def embedding_cache_present(emb_dir: str | Path, split: str) -> bool:
    """Return ``True`` when both cache files for ``split`` already exist."""
    paths = cache_paths(emb_dir, split)
    return paths.ids_path.exists() and paths.embs_path.exists()



def load_embedding_cache(split: str, emb_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load one cached split.

    Returns
    -------
    ids_arr:
        Array of example identifiers.
    embs_arr:
        Array of embeddings. This is memory-mapped for efficiency.
    id_to_row:
        Mapping from example identifier to row index in ``embs_arr``.
    """
    paths = cache_paths(emb_dir, split)
    ids_arr = np.load(str(paths.ids_path), allow_pickle=True)
    embs_arr = np.load(str(paths.embs_path), mmap_mode="r")
    id_to_row = {str(value): int(i) for i, value in enumerate(ids_arr)}
    return ids_arr, embs_arr, id_to_row



def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token embeddings under the attention mask.

    Parameters
    ----------
    last_hidden_state:
        NumPy array of shape ``[batch, seq_len, hidden_dim]``.
    attention_mask:
        NumPy array of shape ``[batch, seq_len]`` with ``0/1`` values.
    """
    mask = attention_mask.astype(np.float64)[..., None]
    masked = last_hidden_state.astype(np.float64) * mask
    denom = np.maximum(mask.sum(axis=1), 1.0)
    return masked.sum(axis=1) / denom



def _encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Encode texts with a transformer encoder.

    The implementation uses the final hidden layer with masked mean pooling.
    This is deliberately simple and explicit.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Transformer embedding generation requires `torch` and `transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = list(texts[start : start + int(batch_size)])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            result = model(**encoded)
            pooled = _mean_pool(
                result.last_hidden_state.detach().cpu().numpy(),
                encoded["attention_mask"].detach().cpu().numpy(),
            )
            outputs.append(pooled)

    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)



def _storage_dtype(name: str) -> np.dtype:
    table = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    key = str(name).strip().lower()
    if key not in table:
        raise ValueError(f"Unsupported storage dtype: {name!r}")
    return table[key]



def ensure_transformer_embeddings(
    *,
    split_to_items: Mapping[str, Iterable[T]],
    emb_dir: str | Path,
    id_getter: Callable[[T], str],
    text_getter: Callable[[T], str],
    model_name: str = "roberta-base",
    batch_size: int = 64,
    max_length: int = 128,
    storage_dtype: str = "float16",
    log_prefix: str = "embeddings",
) -> None:
    """Materialize cached embeddings for all requested splits.

    Existing split caches are left untouched.
    """
    root = Path(emb_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_dtype = _storage_dtype(storage_dtype)

    for split, items_iter in split_to_items.items():
        if embedding_cache_present(root, split):
            print(f"[{log_prefix}] cache already present for split={split} in {root}", flush=True)
            continue

        items = list(items_iter)
        ids = [str(id_getter(item)) for item in items]
        texts = [str(text_getter(item)) for item in items]

        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate ids detected while building embeddings for split={split!r}.")

        print(
            f"[{log_prefix}] encoding split={split} n={len(items)} model={model_name} max_length={max_length}",
            flush=True,
        )
        embs = _encode_texts(
            texts,
            model_name=model_name,
            batch_size=int(batch_size),
            max_length=int(max_length),
        )

        if embs.shape[0] != len(ids):
            raise RuntimeError(
                f"Embedding row count mismatch for split={split!r}: "
                f"expected {len(ids)}, got {embs.shape[0]}"
            )

        paths = cache_paths(root, split)
        np.save(str(paths.ids_path), np.asarray(ids, dtype=object), allow_pickle=True)
        np.save(str(paths.embs_path), np.asarray(embs, dtype=target_dtype))
        print(f"[{log_prefix}] wrote split={split} cache to {paths.embs_path}", flush=True)
