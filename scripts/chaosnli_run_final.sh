#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/chaosnli_common.sh"

REPO_ROOT="${REPO_ROOT:-$(chaosnli_repo_root)}"
OUT_JSON="${OUT_JSON:?OUT_JSON is required}"
HEAD="${HEAD:-linear}"
LR_A="${LR_A:?LR_A is required}"
LR_B="${LR_B:?LR_B is required}"
LR_C="${LR_C:?LR_C is required}"

chaosnli_run_python experiment/chaosnli_cli.py run \
  --dataset-url "${DATASET_URL:-https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1}" \
  --data-root "${DATA_ROOT:-data/chaosnli}" \
  --emb-dir "${EMB_DIR:-out/chaosnli_emb}" \
  --source-subsets ${SOURCE_SUBSETS:-snli mnli} \
  --train-section "${TRAIN_SECTION:-train_full}" \
  --split-seed "${SPLIT_SEED:-13}" \
  --train-frac "${TRAIN_FRAC:-0.8}" \
  --val-frac "${VAL_FRAC:-0.1}" \
  --pi-eps "${PI_EPS:-1e-6}" \
  --tie-tol "${TIE_TOL:-0.0}" \
  --eps-cap "${EPS_CAP:-0.05}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-0}" \
  --train-subset-seed "${TRAIN_SUBSET_SEED:-42}" \
  --selection-split "${SELECTION_SPLIT:-val_full}" \
  --head "${HEAD}" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --proj-tau "${PROJ_TAU:-1e-6}" \
  --proj-kmax "${PROJ_KMAX:-500}" \
  --log-clip-eps "${LOG_CLIP_EPS:-1e-15}" \
  --proj-engine "${PROJ_ENGINE:-cpp}" \
  --mlp-hidden-dim "${MLP_HIDDEN_DIM:-256}" \
  --mlp-dropout "${MLP_DROPOUT:-0.1}" \
  --active-modes A B C \
  --seed-init-A "${SEED_INIT_A:-1000}" \
  --seed-init-B "${SEED_INIT_B:-2000}" \
  --seed-init-C "${SEED_INIT_C:-3000}" \
  --lr-A "${LR_A}" \
  --lr-B "${LR_B}" \
  --lr-C "${LR_C}" \
  --out "${OUT_JSON}"
