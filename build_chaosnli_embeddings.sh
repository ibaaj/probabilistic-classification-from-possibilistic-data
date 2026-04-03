#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-roberta-base}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
export EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-128}"
export EMBEDDING_STORAGE_DTYPE="${EMBEDDING_STORAGE_DTYPE:-float16}"

export DATASET_URL="${DATASET_URL:-https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1}"
export DATA_ROOT="${DATA_ROOT:-data/chaosnli}"
export EMB_DIR="${EMB_DIR:-out/chaosnli_emb}"
export SOURCE_SUBSETS="${SOURCE_SUBSETS:-snli mnli}"
export SPLIT_SEED="${SPLIT_SEED:-13}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.80}"
export VAL_FRAC="${VAL_FRAC:-0.10}"

"${PYTHON}" - <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlpbench.chaosnli.loader import (
    build_split_to_text_records,
    load_raw_items,
    resolve_embedding_dir,
)
from nlpbench.chaosnli.raw import parse_source_subsets
from nlpbench.chaosnli.splits import split_raw_items
from nlpbench.embeddings import ensure_transformer_embeddings, embedding_cache_present

model_name = os.environ.get("EMBEDDING_MODEL", "roberta-base")
batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
max_length = int(os.environ.get("EMBEDDING_MAX_LENGTH", "128"))
storage_dtype = os.environ.get("EMBEDDING_STORAGE_DTYPE", "float16")
dataset_url = os.environ.get("DATASET_URL", "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1")
data_root = Path(os.environ.get("DATA_ROOT", "data/chaosnli"))
source_subsets = parse_source_subsets(os.environ.get("SOURCE_SUBSETS", "snli mnli"))
split_seed = int(os.environ.get("SPLIT_SEED", "13"))
train_frac = float(os.environ.get("TRAIN_FRAC", "0.80"))
val_frac = float(os.environ.get("VAL_FRAC", "0.10"))

emb_dir = resolve_embedding_dir(
    os.environ.get("EMB_DIR", "out/chaosnli_emb"),
    source_subsets=source_subsets,
    split_seed=split_seed,
    train_frac=train_frac,
    val_frac=val_frac,
    embedding_model=model_name,
    embedding_max_length=max_length,
    embedding_storage_dtype=storage_dtype,
)

raw_items = load_raw_items(
    dataset_url=dataset_url,
    data_root=data_root,
    source_subsets=source_subsets,
)
raw_splits = split_raw_items(
    raw_items,
    split_seed=split_seed,
    train_frac=train_frac,
    val_frac=val_frac,
)

split_to_items = build_split_to_text_records(raw_splits)
ensure_transformer_embeddings(
    split_to_items=split_to_items,
    emb_dir=emb_dir,
    id_getter=lambda row: str(row["uid"]),
    text_getter=lambda row: str(row["text"]),
    model_name=model_name,
    batch_size=batch_size,
    max_length=max_length,
    storage_dtype=storage_dtype,
    log_prefix="chaosnli",
)

missing = [split for split in ["train", "validation", "test"] if not embedding_cache_present(emb_dir, split)]
if missing:
    raise SystemExit(f"Missing cache files after generation for splits={missing} in {emb_dir}")
print(f"[chaosnli] cache ready in {emb_dir} for splits=train,validation,test", flush=True)
PY
