#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/chaosnli_common.sh"

REPO_ROOT="${REPO_ROOT:-$(chaosnli_repo_root)}"
OUT_JSON="${OUT_JSON:?OUT_JSON is required}"
HEAD="${HEAD:-linear}"

chaosnli_run_python experiment/chaosnli_cli.py hp-search \
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
  --hp-epochs "${HP_EPOCHS:-100}" \
  --hp-seeds ${HP_SEEDS:-0 1 2} \
  --lr-grid-A ${LR_GRID_A:-1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1} \
  --lr-grid-B ${LR_GRID_B:-1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1} \
  --lr-grid-C ${LR_GRID_C:-1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1} \
  --batch-size "${BATCH_SIZE:-256}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --proj-tau "${PROJ_TAU:-1e-6}" \
  --proj-kmax "${PROJ_KMAX:-500}" \
  --log-clip-eps "${LOG_CLIP_EPS:-1e-15}" \
  --proj-engine "${PROJ_ENGINE:-cpp}" \
  --mlp-hidden-dim "${MLP_HIDDEN_DIM:-256}" \
  --mlp-dropout "${MLP_DROPOUT:-0.1}" \
  --active-modes A B C \
  --hp-train-subset-frac-A "${HP_TRAIN_SUBSET_FRAC_A:-1.0}" \
  --hp-train-subset-size-A "${HP_TRAIN_SUBSET_SIZE_A:-0}" \
  --hp-train-subset-seed-A "${HP_TRAIN_SUBSET_SEED_A:-0}" \
  --hp-confirm-topk-A "${HP_CONFIRM_TOPK_A:-3}" \
  --seed-init-base-A "${SEED_INIT_BASE_A:-1000}" \
  --seed-init-base-B "${SEED_INIT_BASE_B:-2000}" \
  --seed-init-base-C "${SEED_INIT_BASE_C:-3000}" \
  --out "${OUT_JSON}"
