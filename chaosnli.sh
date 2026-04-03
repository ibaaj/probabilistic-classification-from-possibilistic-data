#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON="${PYTHON:-python3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

OUT_BASE="${OUT_BASE:-out/real_nlp/chaosnli}"
HEADS="${HEADS:-linear}"
SELECTION_SPLITS="${SELECTION_SPLITS:-val_full val_S_amb val_S_easy}"
RUN_IDS="${RUN_IDS:-0 1 2 3 4 5 6 7 8 9}"
SEED_BASE="${SEED_BASE:-7000}"

mkdir -p "${OUT_BASE}"

for selection_split in ${SELECTION_SPLITS}; do
  protocol_out="${OUT_BASE}/${selection_split}"
  mkdir -p "${protocol_out}/hp_search" "${protocol_out}/final_runs" "${protocol_out}/aggregate"

  for head in ${HEADS}; do
    tag="${head}"
    hp_json="${protocol_out}/hp_search/${tag}/hp-search.json"
    mkdir -p "$(dirname "${hp_json}")"

    SELECTION_SPLIT="${selection_split}" HEAD="${head}" OUT_JSON="${hp_json}" \
      "${ROOT}/scripts/chaosnli_hp_search.sh"

    HP_JSON="${hp_json}" "${PYTHON}" - <<'PY' > "${protocol_out}/hp_search/${tag}/selected_lrs.txt"
import json
import os
from pathlib import Path

path = Path(os.environ["HP_JSON"])
payload = json.loads(path.read_text(encoding="utf-8"))

results = payload.get("results", [])
if not isinstance(results, list) or not results:
    raise SystemExit(f"No results found in {path}")

r0 = results[0]
required = ("best_lr_A", "best_lr_B", "best_lr_C")
missing = [key for key in required if r0.get(key) in (None, "")]
if missing:
    raise SystemExit(f"Missing selected learning rates in {path}: {missing}")

for key in required:
    print(r0[key])
PY

    mapfile -t lr_lines < "${protocol_out}/hp_search/${tag}/selected_lrs.txt"
    if [ "${#lr_lines[@]}" -ne 3 ]; then
      echo "ERROR: expected 3 selected learning rates in ${protocol_out}/hp_search/${tag}/selected_lrs.txt" >&2
      exit 1
    fi

    lr_A="${lr_lines[0]}"
    lr_B="${lr_lines[1]}"
    lr_C="${lr_lines[2]}"

    for rid in ${RUN_IDS}; do
      seed=$((SEED_BASE + rid))
      run_json="${protocol_out}/final_runs/${tag}/run_rid${rid}.json"
      mkdir -p "$(dirname "${run_json}")"

      SELECTION_SPLIT="${selection_split}" \
      HEAD="${head}" LR_A="${lr_A}" LR_B="${lr_B}" LR_C="${lr_C}" \
      SEED_INIT_A="${seed}" SEED_INIT_B="${seed}" SEED_INIT_C="${seed}" OUT_JSON="${run_json}" \
        "${ROOT}/scripts/chaosnli_run_final.sh"
    done
  done

  RUN_ROOT="${protocol_out}/final_runs" OUT_ROOT="${protocol_out}" "${ROOT}/scripts/chaosnli_posthoc.sh"
done
