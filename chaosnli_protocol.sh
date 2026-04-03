#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$ROOT_DIR}"

export PYTHON="${PYTHON:-python3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
MASTER_OUT="${OUT_ROOT:-${REPO_ROOT}/out/chaosnli_protocol_${TIMESTAMP}}"
mkdir -p "${MASTER_OUT}"

export DATASET_URL="${DATASET_URL:-https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1}"
export DATA_ROOT="${DATA_ROOT:-./data/chaosnli}"
export EMB_DIR="${EMB_DIR:-./out/chaosnli_emb}"
export SOURCE_SUBSETS="${SOURCE_SUBSETS:-snli mnli}"
export SPLIT_SEED="${SPLIT_SEED:-13}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
export VAL_FRAC="${VAL_FRAC:-0.1}"
export PI_EPS="${PI_EPS:-1e-6}"
export TIE_TOL="${TIE_TOL:-0.0}"
export EPS_CAP="${EPS_CAP:-0.05}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export PROJ_TAU="${PROJ_TAU:-1e-6}"
export PROJ_KMAX="${PROJ_KMAX:-500}"
export LOG_CLIP_EPS="${LOG_CLIP_EPS:-1e-15}"
export PROJ_ENGINE="${PROJ_ENGINE:-cpp}"
export EPOCHS="${EPOCHS:-100}"
export HP_EPOCHS="${HP_EPOCHS:-${EPOCHS}}"
export HP_SEEDS="${HP_SEEDS:-0 1 2}"
export RUN_IDS="${RUN_IDS:-0 1 2 3 4 5 6 7 8 9}"
export SEED_BASE="${SEED_BASE:-7000}"

PRIMARY_HEADS="${PRIMARY_HEADS:-linear}"
PRIMARY_TRAIN_SECTIONS="${PRIMARY_TRAIN_SECTIONS:-train_full train_S_amb train_S_easy}"

for head in ${PRIMARY_HEADS}; do
  export HEADS="${head}"
  for train_section in ${PRIMARY_TRAIN_SECTIONS}; do
    export TRAIN_SECTION="${train_section}"
    export OUT_BASE="${MASTER_OUT}/${train_section}"
    "${ROOT_DIR}/chaosnli.sh"
  done
done

echo "Protocol complete: ${MASTER_OUT}"
