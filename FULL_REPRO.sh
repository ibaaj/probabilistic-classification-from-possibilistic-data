#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2
  exit 1
}

run() {
  log "$*"
  "$@"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

need_file() {
  [[ -e "$1" ]] || die "Missing required file: $1"
}

trap 'die "Command failed at line ${LINENO}"' ERR

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SYS_PYTHON="${SYS_PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="$VENV_DIR/bin/python"
export PYTHON

DATASET_URL="${DATASET_URL:-https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1}"
DATA_ROOT="${DATA_ROOT:-data/chaosnli}"
EMB_DIR="${EMB_DIR:-out/chaosnli_emb}"

SOURCE_SUBSETS="${SOURCE_SUBSETS:-snli mnli}"
SPLIT_SEED="${SPLIT_SEED:-13}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
VAL_FRAC="${VAL_FRAC:-0.1}"
PI_EPS="${PI_EPS:-1e-6}"
TIE_TOL="${TIE_TOL:-0.0}"
EPS_CAP="${EPS_CAP:-0.05}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
EVAL_APPLY_KEEP_FILTER="${EVAL_APPLY_KEEP_FILTER:-0}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-roberta-base}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-128}"
EMBEDDING_STORAGE_DTYPE="${EMBEDDING_STORAGE_DTYPE:-float16}"

CHAOSNLI_OUT_BASE="${CHAOSNLI_OUT_BASE:-out/real_nlp/chaosnli}"
CHAOSNLI_TRAIN_SECTION_OUT_ROOT="${CHAOSNLI_TRAIN_SECTION_OUT_ROOT:-$CHAOSNLI_OUT_BASE/train_section_sweep}"
CHAOSNLI_TRAIN_SECTIONS="${CHAOSNLI_TRAIN_SECTIONS:-train_full train_S_amb train_S_easy}"
CHAOSNLI_TRAIN_SECTION_HEADS="${CHAOSNLI_TRAIN_SECTION_HEADS:-linear}"
CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR="${CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR:-$CHAOSNLI_TRAIN_SECTION_OUT_ROOT/article}"
CHAOSNLI_TRAIN_SECTION_MERGED_TEX="${CHAOSNLI_TRAIN_SECTION_MERGED_TEX:-$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_article_table.tex}"
CHAOSNLI_TRAIN_SECTION_MERGED_CSV="${CHAOSNLI_TRAIN_SECTION_MERGED_CSV:-$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_article_rows.csv}"

CHAOSNLI_PROTOCOL_EXPORT_OUTDIR="${CHAOSNLI_PROTOCOL_EXPORT_OUTDIR:-$CHAOSNLI_TRAIN_SECTION_OUT_ROOT/export_protocol}"
CHAOSNLI_THRESHOLDS_CSV="${CHAOSNLI_THRESHOLDS_CSV:-$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_thresholds_summary.csv}"
CHAOSNLI_ARTICLE_DIR="${CHAOSNLI_ARTICLE_DIR:-$CHAOSNLI_OUT_BASE/article}"
CHAOSNLI_SLICE_SIZES_TEX="${CHAOSNLI_SLICE_SIZES_TEX:-$CHAOSNLI_ARTICLE_DIR/chaosnli_slice_sizes.tex}"
CHAOSNLI_SLICE_SIZES_CSV="${CHAOSNLI_SLICE_SIZES_CSV:-$CHAOSNLI_ARTICLE_DIR/chaosnli_slice_sizes.csv}"

TOPK_ROOT="${TOPK_ROOT:-out/agg_gapwide_stair}"
TOPK_TABLE_OUTDIR="${TOPK_TABLE_OUTDIR:-out/big_tables}"

DO_CLEAN="${DO_CLEAN:-1}"
DO_INSTALL="${DO_INSTALL:-1}"
DO_EMBEDDINGS="${DO_EMBEDDINGS:-1}"
DO_CHAOSNLI_TRAIN_SECTION_SWEEP="${DO_CHAOSNLI_TRAIN_SECTION_SWEEP:-1}"
DO_CHAOSNLI_TRAIN_SECTION_TABLES="${DO_CHAOSNLI_TRAIN_SECTION_TABLES:-1}"
DO_CHAOSNLI_EXPORTS="${DO_CHAOSNLI_EXPORTS:-1}"
DO_TOPK="${DO_TOPK:-1}"
DO_TOPK_TABLES="${DO_TOPK_TABLES:-1}"

# ------------------------------------------------------------------------------
# Safety / consistency checks
# ------------------------------------------------------------------------------

if [[ "$DO_CLEAN" == "1" && "$DO_INSTALL" != "1" ]]; then
  die "DO_CLEAN=1 removes .venv, so DO_INSTALL must also be 1."
fi

# ------------------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------------------

need_cmd "$SYS_PYTHON"

need_file "tools/cleanup_repo.py"
need_file "build_chaosnli_embeddings.sh"
need_file "chaosnli.sh"
need_file "chaosnli_protocol.sh"
need_file "run_topk.sh"
need_file "tools/build_chaosnli_results_table.py"
need_file "tools/chaosnli_train_section_article_table.py"
need_file "tools/build_topk_accuracy_summary_table.py"
need_file "tools/extract_chaosnli_thresholds.py"
need_file "tools/export_chaosnli_protocol.py"
need_file "tools/chaosnli_slice_sizes.py"
need_file "klbox/setup_cpp.py"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

source_subset_args=()
old_ifs="$IFS"
IFS=', '
read -r -a source_subset_args <<< "$SOURCE_SUBSETS"
IFS="$old_ifs"

[[ ${#source_subset_args[@]} -gt 0 ]] || die "SOURCE_SUBSETS resolved to an empty list."

train_section_args=()
old_ifs="$IFS"
IFS=' '
read -r -a train_section_args <<< "$CHAOSNLI_TRAIN_SECTIONS"
IFS="$old_ifs"

[[ ${#train_section_args[@]} -gt 0 ]] || die "CHAOSNLI_TRAIN_SECTIONS resolved to an empty list."

eval_keep_filter_args=()
if [[ "$EVAL_APPLY_KEEP_FILTER" == "1" ]]; then
  eval_keep_filter_args+=(--eval-apply-keep-filter)
fi

# ------------------------------------------------------------------------------
# 1) Full cleanup
# ------------------------------------------------------------------------------

if [[ "$DO_CLEAN" == "1" ]]; then
  log "Stage 1/7: full cleanup"
  run "$SYS_PYTHON" tools/cleanup_repo.py --root . --include-out --yes
else
  log "Stage 1/7: full cleanup skipped"
fi

# ------------------------------------------------------------------------------
# 2) Install environment and build extension
# ------------------------------------------------------------------------------

if [[ "$DO_INSTALL" == "1" ]]; then
  log "Stage 2/7: install environment and build extension"

  if [[ ! -d "$VENV_DIR" ]]; then
    run "$SYS_PYTHON" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  [[ -x "$PYTHON" ]] || die "Python interpreter not found in venv: $PYTHON"

  run "$PYTHON" -m pip install --upgrade pip
  run "$PYTHON" -m pip install numpy torch tqdm transformers pybind11
  run "$PYTHON" klbox/setup_cpp.py build_ext --inplace
else
  log "Stage 2/7: install skipped"

  [[ -d "$VENV_DIR" ]] || die "Venv not found: $VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  [[ -x "$PYTHON" ]] || die "Python interpreter not found in venv: $PYTHON"
fi

# ------------------------------------------------------------------------------
# 3) Precompute ChaosNLI embeddings
# ------------------------------------------------------------------------------

if [[ "$DO_EMBEDDINGS" == "1" ]]; then
  log "Stage 3/7: precompute ChaosNLI embeddings"
  run env \
    PYTHON="$PYTHON" \
    DATASET_URL="$DATASET_URL" \
    DATA_ROOT="$DATA_ROOT" \
    EMB_DIR="$EMB_DIR" \
    SOURCE_SUBSETS="$SOURCE_SUBSETS" \
    SPLIT_SEED="$SPLIT_SEED" \
    TRAIN_FRAC="$TRAIN_FRAC" \
    VAL_FRAC="$VAL_FRAC" \
    EMBEDDING_MODEL="$EMBEDDING_MODEL" \
    EMBEDDING_BATCH_SIZE="$EMBEDDING_BATCH_SIZE" \
    EMBEDDING_MAX_LENGTH="$EMBEDDING_MAX_LENGTH" \
    EMBEDDING_STORAGE_DTYPE="$EMBEDDING_STORAGE_DTYPE" \
    bash ./build_chaosnli_embeddings.sh
else
  log "Stage 3/7: embeddings skipped"
fi


# ------------------------------------------------------------------------------
# 4) Run ChaosNLI train-section sweep
# ------------------------------------------------------------------------------

if [[ "$DO_CHAOSNLI_TRAIN_SECTION_SWEEP" == "1" ]]; then
  log "Stage 4/7: run ChaosNLI train-section sweep"
  run env \
    PYTHON="$PYTHON" \
    DATASET_URL="$DATASET_URL" \
    DATA_ROOT="$DATA_ROOT" \
    EMB_DIR="$EMB_DIR" \
    SOURCE_SUBSETS="$SOURCE_SUBSETS" \
    SPLIT_SEED="$SPLIT_SEED" \
    TRAIN_FRAC="$TRAIN_FRAC" \
    VAL_FRAC="$VAL_FRAC" \
    PI_EPS="$PI_EPS" \
    TIE_TOL="$TIE_TOL" \
    EPS_CAP="$EPS_CAP" \
    TRAIN_SUBSET_SEED="$TRAIN_SUBSET_SEED" \
    EVAL_APPLY_KEEP_FILTER="$EVAL_APPLY_KEEP_FILTER" \
    EMBEDDING_MODEL="$EMBEDDING_MODEL" \
    EMBEDDING_BATCH_SIZE="$EMBEDDING_BATCH_SIZE" \
    EMBEDDING_MAX_LENGTH="$EMBEDDING_MAX_LENGTH" \
    EMBEDDING_STORAGE_DTYPE="$EMBEDDING_STORAGE_DTYPE" \
    PRIMARY_HEADS="$CHAOSNLI_TRAIN_SECTION_HEADS" \
    PRIMARY_TRAIN_SECTIONS="$CHAOSNLI_TRAIN_SECTIONS" \
    OUT_ROOT="$CHAOSNLI_TRAIN_SECTION_OUT_ROOT" \
    bash ./chaosnli_protocol.sh
else
  log "Stage 4/7: ChaosNLI train-section sweep skipped"
fi

# ------------------------------------------------------------------------------
# 5) Build ChaosNLI train-section tables and protocol artifacts
# ------------------------------------------------------------------------------

if [[ "$DO_CHAOSNLI_TRAIN_SECTION_TABLES" == "1" ]]; then
  log "Stage 5/7: build ChaosNLI train-section paper tables"
  mkdir -p "$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR"

  train_section_csv_val_full="$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_val_full.csv"
  train_section_csv_val_amb="$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_val_S_amb.csv"
  train_section_csv_val_easy="$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_val_S_easy.csv"

  run "$PYTHON" tools/build_chaosnli_results_table.py \
    --run-root "$CHAOSNLI_TRAIN_SECTION_OUT_ROOT" \
    --selection-splits val_full \
    --train-sections "${train_section_args[@]}" \
    --keep-train-section-in-config \
    --out-tex "$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_val_full.tex" \
    --out-csv "$train_section_csv_val_full" \
    --label "tab:chaosnli-train-section-val-full"

  run "$PYTHON" tools/build_chaosnli_results_table.py \
    --run-root "$CHAOSNLI_TRAIN_SECTION_OUT_ROOT" \
    --selection-splits val_S_amb \
    --train-sections "${train_section_args[@]}" \
    --keep-train-section-in-config \
    --out-tex "$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_val_S_amb.tex" \
    --out-csv "$train_section_csv_val_amb" \
    --label "tab:chaosnli-train-section-val-amb"

  run "$PYTHON" tools/build_chaosnli_results_table.py \
    --run-root "$CHAOSNLI_TRAIN_SECTION_OUT_ROOT" \
    --selection-splits val_S_easy \
    --train-sections "${train_section_args[@]}" \
    --keep-train-section-in-config \
    --out-tex "$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR/chaosnli_train_section_val_S_easy.tex" \
    --out-csv "$train_section_csv_val_easy" \
    --label "tab:chaosnli-train-section-val-easy"

  run "$PYTHON" tools/chaosnli_train_section_article_table.py \
    --csv "$train_section_csv_val_full" \
    --csv "$train_section_csv_val_amb" \
    --csv "$train_section_csv_val_easy" \
    --out-tex "$CHAOSNLI_TRAIN_SECTION_MERGED_TEX" \
    --out-csv "$CHAOSNLI_TRAIN_SECTION_MERGED_CSV"
else
  log "Stage 5/7: ChaosNLI train-section paper tables skipped"
fi

if [[ "$DO_CHAOSNLI_EXPORTS" == "1" ]]; then
  log "Stage 5/7: export ChaosNLI thresholds summary, slice sizes, and protocol artifacts"
  mkdir -p "$CHAOSNLI_TRAIN_SECTION_ARTICLE_DIR"
  mkdir -p "$CHAOSNLI_ARTICLE_DIR"
  mkdir -p "$CHAOSNLI_PROTOCOL_EXPORT_OUTDIR"

  run "$PYTHON" tools/extract_chaosnli_thresholds.py \
    "$CHAOSNLI_TRAIN_SECTION_OUT_ROOT/train_full/val_full" \
    --out-csv "$CHAOSNLI_THRESHOLDS_CSV"

  run "$PYTHON" tools/export_chaosnli_protocol.py \
    --dataset-url "$DATASET_URL" \
    --data-root "$DATA_ROOT" \
    --emb-dir "$EMB_DIR" \
    --source-subsets "${source_subset_args[@]}" \
    --train-section train_full \
    --split-seed "$SPLIT_SEED" \
    --train-frac "$TRAIN_FRAC" \
    --val-frac "$VAL_FRAC" \
    --pi-eps "$PI_EPS" \
    --tie-tol "$TIE_TOL" \
    --eps-cap "$EPS_CAP" \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
    --embedding-storage-dtype "$EMBEDDING_STORAGE_DTYPE" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --train-subset-seed "$TRAIN_SUBSET_SEED" \
    "${eval_keep_filter_args[@]}" \
    --out-dir "$CHAOSNLI_PROTOCOL_EXPORT_OUTDIR"

  run "$PYTHON" tools/chaosnli_slice_sizes.py \
    --root "$CHAOSNLI_OUT_BASE" \
    --out-tex "$CHAOSNLI_SLICE_SIZES_TEX" \
    --out-csv "$CHAOSNLI_SLICE_SIZES_CSV"
else
  log "Stage 5/7: ChaosNLI exports skipped"
fi

# ------------------------------------------------------------------------------
# 6) Run synthetic top-k pipeline
# ------------------------------------------------------------------------------

if [[ "$DO_TOPK" == "1" ]]; then
  log "Stage 6/7: run synthetic top-k pipeline"
  run env \
    PYTHON="$PYTHON" \
    AGG_DIR="$TOPK_ROOT" \
    bash ./run_topk.sh
else
  log "Stage 6/7: synthetic top-k pipeline skipped"
fi

# ------------------------------------------------------------------------------
# 7) Build top-k summary tables
# ------------------------------------------------------------------------------

if [[ "$DO_TOPK_TABLES" == "1" ]]; then
  log "Stage 7/7: build top-k summary tables"
  mkdir -p "$TOPK_TABLE_OUTDIR"
  run "$PYTHON" tools/build_topk_accuracy_summary_table.py \
    --root "$TOPK_ROOT" \
    --outdir "$TOPK_TABLE_OUTDIR"
else
  log "Stage 7/7: top-k summary tables skipped"
fi

log "FULL_REPRO completed successfully."
