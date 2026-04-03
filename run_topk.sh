#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CLI="${CLI:-./synthetic_cli.py}"

ROOT_RUN_DIR="${RUN_DIR:-out/runs_gapwide_stair}"
ROOT_AGG_DIR="${AGG_DIR:-out/agg_gapwide_stair}"
DO_AGG="${DO_AGG:-1}"

mkdir -p "${ROOT_RUN_DIR}" "${ROOT_AGG_DIR}"

# ---------------------------------------------------------------------------
# Data / task
# ---------------------------------------------------------------------------
N_CLASSES="${N_CLASSES:-20}"
TEST_N="${TEST_N:-3000}"
X_NOISE="${X_NOISE:-2.0}"
ALPHA_NOISE="${ALPHA_NOISE:-0.15}"
D_LIST=(${D_LIST_OVERRIDE:-30 80 150})

class_sep_for_d() {
  local d="$1"
  case "${d}" in
    30)  echo "1.5" ;;
    80)  echo "0.9" ;;
    150) echo "0.6" ;;
    *)   echo "1.0" ;;
  esac
}

# ---------------------------------------------------------------------------
# π
# ---------------------------------------------------------------------------
PI_EPS="${PI_EPS:-1e-6}"
PI_STAIR_STEP="${PI_STAIR_STEP:-0.01}"
PI_STAIR_M="${PI_STAIR_M:-0}"

# ---------------------------------------------------------------------------
# GAP-WIDE
# ---------------------------------------------------------------------------
TIE_TOL="${TIE_TOL:-0.0}"
EPS_CAP="${EPS_CAP:-1e-9}"

# ---------------------------------------------------------------------------
# Projection / numeric eps
# ---------------------------------------------------------------------------
PROJ_TAU="${PROJ_TAU:-1e-8}"
PROJ_K="${PROJ_K:-2000}"
LOG_CLIP_EPS="${LOG_CLIP_EPS:-1e-15}"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

ALPHAS=(${ALPHAS_OVERRIDE:-0.4 0.6 0.8 0.95})
NTRS=(${NTRS_OVERRIDE:-200 500 1000})
RUNS=(${RUNS_OVERRIDE:-0 1 2 3 4 5 6 7 8 9})

epochs_for_ntr() {
  local ntr="$1"
  if   [[ "${ntr}" -le 200 ]]; then echo "${EPOCHS_SMALL:-80}"
  elif [[ "${ntr}" -le 500 ]]; then echo "${EPOCHS_MED:-60}"
  else                              echo "${EPOCHS_LARGE:-60}"
  fi
}

batch_for_ntr() {
  local ntr="$1"
  if [[ "${ntr}" -le 200 ]]; then echo "${BATCH_SMALL:-64}"
  else                             echo "${BATCH_LARGE:-128}"
  fi
}

float_tag() {
  local x="$1"
  x="${x//-/m}"
  x="${x//./p}"
  echo "${x}"
}

extract_selected_lr() {
  local which="$1"
  local json_path="$2"
  local alpha="$3"

  "${PYTHON}" "${JSON_EXTRACT}" "${which}" \
    --json "${json_path}" \
    --alpha "${alpha}"
}

# ---------------------------------------------------------------------------
# HP-search config
# ---------------------------------------------------------------------------
HP_CRITERION="${HP_CRITERION:-acc}"
HP_LR_GRID_A=(${HP_LR_GRID_A_OVERRIDE:-1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2})
HP_LR_GRID_B=(${HP_LR_GRID_B_OVERRIDE:-1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2})
HP_VAL_SEEDS=(${HP_VAL_SEEDS_OVERRIDE:-100 101 102})

HP_TRAIN_SUBSET_FRAC_A="${HP_TRAIN_SUBSET_FRAC_A:-1.0}"
HP_TRAIN_SUBSET_SIZE_A="${HP_TRAIN_SUBSET_SIZE_A:-0}"
HP_TRAIN_SUBSET_SEED_A="${HP_TRAIN_SUBSET_SEED_A:-0}"
HP_CONFIRM_TOPK_A="${HP_CONFIRM_TOPK_A:-1}"

SEED_INIT_BASE_A="${SEED_INIT_BASE_A:-1000}"
SEED_INIT_BASE_B="${SEED_INIT_BASE_B:-2000}"

JSON_EXTRACT="${JSON_EXTRACT:-./tools/extract_topk_json.py}"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for d in "${D_LIST[@]}"; do
  class_sep="$(class_sep_for_d "${d}")"
  d_tag="$(float_tag "${d}")"
  cs_tag="$(float_tag "${class_sep}")"

  for alpha in "${ALPHAS[@]}"; do
    alpha_tag="$(float_tag "${alpha}")"

    for ntr in "${NTRS[@]}"; do
      epochs="${EPOCHS_OVERRIDE:-$(epochs_for_ntr "${ntr}")}"
      batch="${BATCH_OVERRIDE:-$(batch_for_ntr "${ntr}")}"

      val_n_a="${ntr}"
      val_n_b="${ntr}"

      RUN_DIR_CFG="${ROOT_RUN_DIR}/d${d_tag}_cs${cs_tag}/alpha${alpha_tag}/ntr${ntr}"
      AGG_DIR_CFG="${ROOT_AGG_DIR}/d${d_tag}_cs${cs_tag}/alpha${alpha_tag}/ntr${ntr}"
      mkdir -p "${RUN_DIR_CFG}" "${AGG_DIR_CFG}"

      echo "[CONFIG d=${d} cs=${class_sep} alpha=${alpha} ntr=${ntr}]"
      echo "        [pi: stair eps=${PI_EPS} step=${PI_STAIR_STEP} m=${PI_STAIR_M} | K=N_CLASSES=${N_CLASSES}]"
      echo "        [gap: GAP-WIDE tie_tol=${TIE_TOL} eps_cap=${EPS_CAP}]"
      echo "        [proj: tau=${PROJ_TAU} K=${PROJ_K} | log_clip_eps=${LOG_CLIP_EPS}]"

      # ---- Per-config HP search for A
      hp_json_a="${AGG_DIR_CFG}/hp_search_A.json"
      echo "[HP-A val=${val_n_a} criterion=${HP_CRITERION} lr_grid_A=(${HP_LR_GRID_A[*]}) val_seeds=(${HP_VAL_SEEDS[*]}) subset_frac=${HP_TRAIN_SUBSET_FRAC_A} subset_size=${HP_TRAIN_SUBSET_SIZE_A} subset_seed=${HP_TRAIN_SUBSET_SEED_A} confirm_topk=${HP_CONFIRM_TOPK_A}]"

      "${PYTHON}" "${CLI}" hp-search \
        --n-classes "${N_CLASSES}" \
        --d "${d}" \
        --alpha "${alpha}" \
        --train "${ntr}" \
        --val "${val_n_a}" \
        --epochs "${epochs}" \
        --batch "${batch}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --pi-eps "${PI_EPS}" \
        --pi-stair-step "${PI_STAIR_STEP}" \
        --pi-stair-m "${PI_STAIR_M}" \
        --alpha-noise "${ALPHA_NOISE}" \
        --class-sep "${class_sep}" \
        --x-noise "${X_NOISE}" \
        --tie-tol "${TIE_TOL}" \
        --eps-cap "${EPS_CAP}" \
        --log-clip-eps "${LOG_CLIP_EPS}" \
        --proj-tau-train "${PROJ_TAU}" \
        --proj-K-train "${PROJ_K}" \
        --criterion "${HP_CRITERION}" \
        --lr-grid "${HP_LR_GRID_A[@]}" \
        --val-seeds "${HP_VAL_SEEDS[@]}" \
        --seed-init-base-A "${SEED_INIT_BASE_A}" \
        --hp-mode A \
        --out "${hp_json_a}" \
        --hp-train-subset-frac-A "${HP_TRAIN_SUBSET_FRAC_A}" \
        --hp-train-subset-size-A "${HP_TRAIN_SUBSET_SIZE_A}" \
        --hp-train-subset-seed-A "${HP_TRAIN_SUBSET_SEED_A}" \
        --hp-confirm-topk-A "${HP_CONFIRM_TOPK_A}"

      lrA_selected="$(extract_selected_lr "best-lr-A" "${hp_json_a}" "${alpha}")"

      # ---- Per-config HP search for B
      echo "[HP-B val=${val_n_b} criterion=${HP_CRITERION} lr_grid_B=(${HP_LR_GRID_B[*]}) val_seeds=(${HP_VAL_SEEDS[*]})]"

      hp_json_b="${AGG_DIR_CFG}/hp_search_B.json"
      "${PYTHON}" "${CLI}" hp-search \
        --n-classes "${N_CLASSES}" \
        --d "${d}" \
        --alpha "${alpha}" \
        --train "${ntr}" \
        --val "${val_n_b}" \
        --epochs "${epochs}" \
        --batch "${batch}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --pi-eps "${PI_EPS}" \
        --pi-stair-step "${PI_STAIR_STEP}" \
        --pi-stair-m "${PI_STAIR_M}" \
        --alpha-noise "${ALPHA_NOISE}" \
        --class-sep "${class_sep}" \
        --x-noise "${X_NOISE}" \
        --tie-tol "${TIE_TOL}" \
        --eps-cap "${EPS_CAP}" \
        --log-clip-eps "${LOG_CLIP_EPS}" \
        --proj-tau-train "${PROJ_TAU}" \
        --proj-K-train "${PROJ_K}" \
        --criterion "${HP_CRITERION}" \
        --lr-grid "${HP_LR_GRID_B[@]}" \
        --val-seeds "${HP_VAL_SEEDS[@]}" \
        --seed-init-base-B "${SEED_INIT_BASE_B}" \
        --hp-mode B \
        --out "${hp_json_b}"

      lrB_selected="$(extract_selected_lr "best-lr-B" "${hp_json_b}" "${alpha}")"

      echo "${lrA_selected}" > "${AGG_DIR_CFG}/lrA_selected.txt"
      echo "${lrB_selected}" > "${AGG_DIR_CFG}/lrB_selected.txt"
      echo "        [selected LRs: lrA=${lrA_selected} lrB=${lrB_selected}]"
      acc_csv="${AGG_DIR_CFG}/per_run_acc_AB.csv"
      echo "run,acc_A,acc_B" > "${acc_csv}"

      # ---- Runs
      for run in "${RUNS[@]}"; do
        seed_data="${run}"
        seed_init_A="$((SEED_INIT_BASE_A + run))"
        seed_init_B="$((SEED_INIT_BASE_B + run))"
        
        base="topk_adam_d${d_tag}_cs${cs_tag}_alpha${alpha_tag}_ntr${ntr}_ep${epochs}_bs${batch}_Kall_step$(float_tag "${PI_STAIR_STEP}")_m${PI_STAIR_M}_gapwide_epscap$(float_tag "${EPS_CAP}")_ptau$(float_tag "${PROJ_TAU}")_pK${PROJ_K}_lrA$(float_tag "${lrA_selected}")_lrB$(float_tag "${lrB_selected}")_run${run}"
        out_json="${RUN_DIR_CFG}/${base}.json"
        out_stdout="${RUN_DIR_CFG}/${base}.stdout"
        out_stderr="${RUN_DIR_CFG}/${base}.stderr"

        echo "[RUN d=${d} cs=${class_sep} alpha=${alpha} N_tr=${ntr} ep=${epochs} bs=${batch} K=${N_CLASSES} step=${PI_STAIR_STEP} m=${PI_STAIR_M} lrA=${lrA_selected} lrB=${lrB_selected} run=${run}]"

        "${PYTHON}" "${CLI}" topk-exp \
          --n-classes "${N_CLASSES}" \
          --d "${d}" \
          --alpha "${alpha}" \
          --train "${ntr}" \
          --test "${TEST_N}" \
          --seed-data "${seed_data}" \
          --seed-init-A "${seed_init_A}" \
          --seed-init-B "${seed_init_B}" \
          --lr-A "${lrA_selected}" \
          --lr-B "${lrB_selected}" \
          --epochs "${epochs}" \
          --batch "${batch}" \
          --weight-decay "${WEIGHT_DECAY}" \
          --pi-eps "${PI_EPS}" \
          --pi-stair-step "${PI_STAIR_STEP}" \
          --pi-stair-m "${PI_STAIR_M}" \
          --alpha-noise "${ALPHA_NOISE}" \
          --class-sep "${class_sep}" \
          --x-noise "${X_NOISE}" \
          --tie-tol "${TIE_TOL}" \
          --eps-cap "${EPS_CAP}" \
          --log-clip-eps "${LOG_CLIP_EPS}" \
          --proj-tau-train "${PROJ_TAU}" \
          --proj-K-train "${PROJ_K}" \
          --out "${out_json}" \
          1> "${out_stdout}" 2> "${out_stderr}"

        read -r accA accB < <(
          "${PYTHON}" "${JSON_EXTRACT}" acc-ab --json "${out_json}" --alpha "${alpha}"
        )

        echo "[RESULT run=${run}] acc_A=${accA} acc_B=${accB}"
        echo "${run},${accA},${accB}" >> "${acc_csv}"
      done

      # ---- Aggregate
      if [[ "${DO_AGG}" == "1" ]]; then
        "${PYTHON}" ./tools/aggregate_topk_runs.py \
          --input-dir "${RUN_DIR_CFG}" \
          --alpha "${alpha}" \
          --out-dir "${AGG_DIR_CFG}" \
          --metrics acc nll ece brier V_mean V_p90 V_max

        echo "wrote agg in ${AGG_DIR_CFG}"
        echo "wrote per-run acc CSV: ${acc_csv}"
      fi

    done
  done
done

# ---------------------------------------------------------------------------
# Dykstra implementation tests
# ---------------------------------------------------------------------------
DYK_DIR="${ROOT_AGG_DIR}/dykstra_gapwide"
mkdir -p "${DYK_DIR}"

DY_N="${DY_N:-100}"
DY_RUNS="${DY_RUNS:-100}"
DY_SEED="${DY_SEED:-0}"
DY_KMAX=(${DY_KMAX_OVERRIDE:-1000 10000 50000})
DY_TAU=(${DY_TAU_OVERRIDE:-1e-2 1e-3 1e-4 1e-6 1e-8})

for Kmax in "${DY_KMAX[@]}"; do
  out_tex="${DYK_DIR}/dykstra_gapwide_K${Kmax}.tex"
  echo "[DYKSTRA-SWEEP gap-wide n=${DY_N} runs=${DY_RUNS} Kmax=${Kmax} taus=(${DY_TAU[*]})] -> ${out_tex}"
  "${PYTHON}" "${CLI}" dykstra-sweep \
    --n "${DY_N}" \
    --runs "${DY_RUNS}" \
    --seed "${DY_SEED}" \
    --Kmax "${Kmax}" \
    --tau "${DY_TAU[@]}" \
    --tie-tol "${TIE_TOL}" \
    --eps-cap "${EPS_CAP}" \
    --pi-eps "${PI_EPS}" \
    --log-clip-eps "${LOG_CLIP_EPS}" \
    > "${out_tex}"
done