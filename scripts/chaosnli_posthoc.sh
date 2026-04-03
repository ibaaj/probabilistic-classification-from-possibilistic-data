#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/chaosnli_common.sh"

REPO_ROOT="${REPO_ROOT:-$(chaosnli_repo_root)}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
AMBIG_DIR="${OUT_ROOT}/ambiguity"
SLICE_DIR="${OUT_ROOT}/slice_eval"
AGG_DIR="${OUT_ROOT}/aggregate"
THRESH_JSON="${AMBIG_DIR}/chaosnli_ambiguity_thresholds.json"
mkdir -p "${AMBIG_DIR}" "${SLICE_DIR}" "${AGG_DIR}"

chaosnli_run_python experiment/analyze_ambiguity.py \
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
  --train-subset-seed "${TRAIN_SUBSET_SEED:-42}" \
  --out-dir "${AMBIG_DIR}"

find "${RUN_ROOT}" -name '*.json' | while read -r run_json; do
  stem="$(basename "${run_json}" .json)"
  out_csv="${SLICE_DIR}/${stem}.slice_eval.csv"
  chaosnli_run_python experiment/slice_eval.py \
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
    --run-json "${run_json}" \
    --thresholds-json "${THRESH_JSON}" \
    --out "${out_csv}"
done

chaosnli_run_python experiment/aggregate_chaosnli_runs.py --inputs "${RUN_ROOT}" --out-dir "${AGG_DIR}/runs"
chaosnli_run_python experiment/aggregate_chaosnli_slice_eval.py --inputs "${SLICE_DIR}" --out-dir "${AGG_DIR}/slice_eval"
