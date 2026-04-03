#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

DATASET_URL="${DATASET_URL:-https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1}"
DATA_ROOT="${DATA_ROOT:-data/chaosnli}"
EMB_DIR="${EMB_DIR:-out/chaosnli_emb}"
SOURCE_SUBSETS="${SOURCE_SUBSETS:-snli mnli}"
SPLIT_SEED="${SPLIT_SEED:-13}"
TRAIN_FRAC="${TRAIN_FRAC:-0.80}"
VAL_FRAC="${VAL_FRAC:-0.10}"
TRAIN_SECTION="${TRAIN_SECTION:-train_full}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-roberta-base}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-128}"
EMBEDDING_STORAGE_DTYPE="${EMBEDDING_STORAGE_DTYPE:-float16}"

PI_EPS="${PI_EPS:-1e-6}"
TIE_TOL="${TIE_TOL:-0.0}"
EPS_CAP="${EPS_CAP:-0.05}"

OUT_DIR="${OUT_DIR:-out/chaosnli_protocol_dump}"

mkdir -p "${OUT_DIR}"
read -r -a SOURCE_SUBSETS_ARR <<< "${SOURCE_SUBSETS}"

"${PYTHON}" tools/export_chaosnli_protocol.py \
  --dataset-url "${DATASET_URL}" \
  --data-root "${DATA_ROOT}" \
  --emb-dir "${EMB_DIR}" \
  --source-subsets "${SOURCE_SUBSETS_ARR[@]}" \
  --train-section "${TRAIN_SECTION}" \
  --split-seed "${SPLIT_SEED}" \
  --train-frac "${TRAIN_FRAC}" \
  --val-frac "${VAL_FRAC}" \
  --pi-eps "${PI_EPS}" \
  --tie-tol "${TIE_TOL}" \
  --eps-cap "${EPS_CAP}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --embedding-max-length "${EMBEDDING_MAX_LENGTH}" \
  --embedding-storage-dtype "${EMBEDDING_STORAGE_DTYPE}" \
  --out-dir "${OUT_DIR}"

"${PYTHON}" tools/chaosnli_slice_sizes.py \
  --input "${OUT_DIR}/chaosnli_slice_membership.csv" \
  --out-tex "${OUT_DIR}/chaosnli_slice_sizes.tex" \
  --out-csv "${OUT_DIR}/chaosnli_slice_sizes.csv"

"${PYTHON}" tools/extract_chaosnli_thresholds.py \
  "${OUT_DIR}" \
  --out-csv "${OUT_DIR}/chaosnli_thresholds_table.csv"

echo
echo "Main files written in ${OUT_DIR}:"
echo "  - chaosnli_split_manifest.csv        # exact sample ids used for split=train/validation/test; train_full is added separately only if it differs from train"
echo "  - chaosnli_split_manifest.jsonl"
echo "  - chaosnli_slice_membership.csv      # exact sample ids + in_S_amb / in_S_easy flags"
echo "  - chaosnli_slice_membership.jsonl"
echo "  - chaosnli_thresholds.json           # raw thresholds JSON"
echo "  - chaosnli_protocol_metadata.json    # protocol metadata"
echo "  - chaosnli_slice_sizes.csv           # small numeric summary"
echo "  - chaosnli_slice_sizes.tex"
echo "  - chaosnli_thresholds_table.csv      # compact thresholds table"