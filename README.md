# Possibilistic supervision experiments

This repository contains two experiment families built around training targets derived from a possibilistic description $\pi$ and its induced admissible KL-box $\mathcal{F}^{\mathrm{box}}(\pi)$:

- a **synthetic top-k** benchmark for controlled multi-class classification experiments,
- a **ChaosNLI** pipeline for real-text experiments on human-disagreement NLI data.

The repository studies how different target constructions behave when supervision is derived from a possibility ordering rather than from a single hard label.

## What is implemented

### Synthetic top-k setting

The synthetic benchmark compares two training rules.

- **Model A (projection target):** at each step, project the current prediction $q(x)$ onto the sample-specific box $\mathcal{F}^{\mathrm{box}}(\pi)$ with Dykstra’s algorithm, then train on
  $$
  \mathrm{KL}\!\left(p^\star(x,\pi)\,\|\,q(x)\right).
  $$
- **Model B (fixed target):** train directly on the fixed target $\dot p(\pi)$ obtained from the antipignistic reverse mapping:
  $$
  \mathrm{KL}\!\left(\dot p(\pi)\,\|\,q(x)\right).
  $$

There is no Model C in the synthetic benchmark.

### ChaosNLI setting

The ChaosNLI pipeline supports three training modes.

- **Mode A (projection target):** projection-based target derived from the example-specific KL-box.
- **Mode B (fixed target):** fixed antipignistic target $\dot p$ derived from the vote-induced possibility ordering.
- **Mode C (vote target):** empirical vote distribution.

In addition to the standard train/validation/test split protocol, the repository supports a **train-section study** in which model fitting uses one of

- `train_full`,
- `train_S_amb`,
- `train_S_easy`,

while validation-section selection and test-section evaluation remain defined by a fixed protocol.

---

## Repository structure

### Core directories

- `klbox/`  
  Possibility ordering, antipignistic reverse mapping, gap selection, linear feasibility systems, Python/C++ Dykstra implementations, and numerical helpers.

- `topk/`  
  Synthetic benchmark code: data generation, model heads, targets, training, experiment runners, and hyperparameter search.

- `nlpbench/`  
  Reusable NLP-side utilities:
  - `nlpbench/embeddings.py` for cached transformer embeddings,
  - `nlpbench/sampling.py` for deterministic sampling helpers,
  - `nlpbench/chaosnli/` for the ChaosNLI loader and data protocol.

- `experiment/`  
  ChaosNLI experiment-layer code: CLI, target definitions, training loop, metrics, ambiguity analysis, slice evaluation, and aggregation.

- `common/`  
  Small shared JSON/CSV I/O helpers and deterministic sampling utilities used across the repository.

### Main shell launchers

- `run_topk.sh`  
  End-to-end launcher for the synthetic benchmark.

- `build_chaosnli_embeddings.sh`  
  Precomputes and caches transformer embeddings for a ChaosNLI split protocol.

- `chaosnli.sh`  
  Main ChaosNLI launcher for one train-section setting across all requested validation-selection protocols.

- `chaosnli_snli.sh`, `chaosnli_mnli.sh`, `chaosnli_all.sh`  
  Convenience wrappers that set `SOURCE_SUBSETS` and `OUT_BASE` before calling `chaosnli.sh`.

- `chaosnli_protocol.sh`  
  Higher-level launcher that sweeps several ChaosNLI train sections under a shared output root.

- `FULL_REPRO.sh`  
  Repository-wide launcher for the default reproducibility workflow.

### Main table/export tools

- `tools/build_chaosnli_results_table.py`  
  Builds per-protocol ChaosNLI summary tables from saved run JSON files.

- `tools/chaosnli_train_section_article_table.py`  
  Merges multiple per-protocol ChaosNLI CSV summaries into one article-facing train-section table.

- `tools/chaosnli_slice_sizes.py`  
  Builds a compact export of train/validation/test section sizes for the ChaosNLI protocol, with optional LaTeX and CSV output.

- `tools/export_chaosnli_protocol.py`  
  Exports split manifests, slice memberships, thresholds, and protocol metadata for ChaosNLI.

- `tools/extract_chaosnli_thresholds.py`  
  Collects saved thresholds into a compact summary table. Accepts both `chaosnli_thresholds.json` and the legacy `chaosnli_ambiguity_thresholds.json`.

- `tools/aggregate_topk_runs.py`  
  Aggregates repeated synthetic runs and writes CSV/LaTeX summaries.

- `tools/build_topk_accuracy_summary_table.py`  
  Builds the global article-facing synthetic accuracy table.

---

## Requirements

The codebase uses Python features such as `str | Path` and `list[str]`, so **Python 3.10+** is the practical requirement.

The main dependencies used by the default workflows are:

- `numpy`
- `torch`
- `tqdm`
- `transformers`
- `pybind11`

The synthetic CLI requires the C++ projection backend. The ChaosNLI scripts also use the C++ backend by default.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy torch tqdm transformers pybind11
python3 klbox/setup_cpp.py build_ext --inplace
```

A quick repository-wide syntax check is:

```bash
python3 -m compileall common experiment klbox nlpbench topk tools
```

A direct benchmark of the projection backend is:

```bash
python3 synthetic_cli.py dykstra-sweep --n 50 --runs 5 --tau 1e-6 1e-8 --Kmax 1000
```

This prints LaTeX tables to standard output.

---

## Quick start

### Synthetic benchmark

```bash
./run_topk.sh
```

### ChaosNLI embeddings only

```bash
SOURCE_SUBSETS="snli mnli" ./build_chaosnli_embeddings.sh
```

### ChaosNLI main launcher

```bash
./chaosnli.sh
```

### ChaosNLI train-section protocol sweep

```bash
./chaosnli_protocol.sh
```

### Repository-wide workflow

```bash
./FULL_REPRO.sh
```

---

## Running the synthetic top-k benchmark

## Main launcher

The synthetic benchmark is driven by:

```bash
./run_topk.sh
```

This script performs, for each configured setting:

1. hyperparameter search for Model A,
2. hyperparameter search for Model B,
3. repeated experiment runs,
4. aggregation of repeated runs,
5. Dykstra-solver sweeps.

## Default grid

By default, `run_topk.sh` loops over:

- feature dimensions `d = 30, 80, 150`,
- training sizes `N_tr = 200, 500, 1000`,
- ambiguity parameters `alpha = 0.4, 0.6, 0.8, 0.95`,
- run ids `0, 1, ..., 9`.

The class-separation parameter is chosen as a function of `d` inside the script:

- `d=30  -> class_sep=1.5`
- `d=80  -> class_sep=0.9`
- `d=150 -> class_sep=0.6`

## What `run_topk.sh` writes

Per-configuration run logs are written under:

- `out/runs_gapwide_stair/`

Aggregated summaries are written under:

- `out/agg_gapwide_stair/`

The final article-style global synthetic table is produced by:

- `tools/build_topk_accuracy_summary_table.py`

and is written by default to:

- `out/big_tables/big_all_test_acc_topk.tex`

## Synthetic smoke test

A small one-configuration smoke test is:

```bash
D_LIST_OVERRIDE="30" \
ALPHAS_OVERRIDE="0.4" \
NTRS_OVERRIDE="200" \
RUNS_OVERRIDE="0" \
HP_VAL_SEEDS_OVERRIDE="100" \
HP_LR_GRID_A_OVERRIDE="1e-3" \
HP_LR_GRID_B_OVERRIDE="1e-3" \
./run_topk.sh
```

---

## Running ChaosNLI

## Data source and subsets

The loader downloads the ChaosNLI release archive from:

- `https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1`

Supported source subsets are:

- `snli`
- `mnli`

Subset specifications may be given in either comma-separated or space-separated form. All of the following are accepted:

```bash
SOURCE_SUBSETS="snli mnli"
SOURCE_SUBSETS="snli,mnli"
SOURCE_SUBSETS="mnli snli"
```

The loader deduplicates subsets and preserves user-facing order, while the embedding-cache naming scheme uses a canonical sorted subset key.

## Embedding caches

The ChaosNLI pipeline uses cached transformer embeddings. A standard precomputation step is:

```bash
SOURCE_SUBSETS="snli mnli" ./build_chaosnli_embeddings.sh
```

By default, embeddings are produced with:

- model: `roberta-base`
- batch size: `64`
- max length: `128`
- storage dtype: `float16`

The cache stores, for each split:

- `{split}_ids.npy`
- `{split}_embs.npy`

for `split` in `train`, `validation`, `test`.

When `EMB_DIR` is left at its default base name `out/chaosnli_emb`, the loader automatically scopes the effective cache directory by:

- subset list,
- split seed,
- train/validation fractions,
- embedding model,
- embedding max length,
- embedding storage dtype.

This avoids silent collisions between incompatible embedding protocols.

## Main launcher

The main ChaosNLI launcher is:

```bash
./chaosnli.sh
```

Convenience wrappers are:

```bash
./chaosnli_snli.sh
./chaosnli_mnli.sh
./chaosnli_all.sh
```

Their defaults are:

- `chaosnli_snli.sh`: `SOURCE_SUBSETS=snli`, `OUT_BASE=out/real_nlp/chaosnli_snli`
- `chaosnli_mnli.sh`: `SOURCE_SUBSETS=mnli`, `OUT_BASE=out/real_nlp/chaosnli_mnli`
- `chaosnli_all.sh`: `SOURCE_SUBSETS=snli,mnli`, `OUT_BASE=out/real_nlp/chaosnli_all`

## What `chaosnli.sh` does

For each requested validation-selection protocol in `SELECTION_SPLITS` and for each requested head in `HEADS`, `chaosnli.sh` performs the following stages.

### 1. Hyperparameter search

It calls:

```bash
scripts/chaosnli_hp_search.sh
```

which runs:

```bash
python3 experiment/chaosnli_cli.py hp-search ...
```

This stage:

- loads the requested ChaosNLI subsets,
- builds deterministic train/validation/test splits,
- ensures embeddings are available,
- evaluates learning-rate grids for Modes A, B, and C,
- saves the selected learning rates.

Outputs are written under:

- `.../hp_search/<head>/hp-search.json`
- `.../hp_search/<head>/selected_lrs.txt`

### 2. Final runs

For each run id in `RUN_IDS`, it calls:

```bash
scripts/chaosnli_run_final.sh
```

which runs:

```bash
python3 experiment/chaosnli_cli.py run ...
```

Each saved run writes:

- one run JSON file,
- `*_test_full_ids.npy`,
- `*_test_full_y.npy`,
- `*_test_full_probs_A.npy`,
- `*_test_full_probs_B.npy`,
- `*_test_full_probs_C.npy`

when the corresponding modes are active.

### 3. Post-hoc ambiguity and slice analysis

After the final runs, `chaosnli.sh` calls:

```bash
scripts/chaosnli_posthoc.sh
```

This stage:

- computes train-derived ambiguity thresholds,
- evaluates every saved run on the ambiguity slices,
- aggregates run-level JSON logs,
- aggregates slice-level CSV outputs.

Outputs are written under:

- `.../ambiguity/`
- `.../slice_eval/`
- `.../aggregate/`

## Validation-selection protocols

The ChaosNLI pipeline uses three validation-selection protocols:

- `val_full`
- `val_S_amb`
- `val_S_easy`

Each validation-selection protocol has a paired test section:

- `val_full  -> test_full`
- `val_S_amb -> test_S_amb`
- `val_S_easy -> test_S_easy`

Saved run JSON files record both:

- `selection_split`
- `selection_test_split`

## Train sections

The data loader always keeps `train_full` available for protocol analysis. Model fitting may instead use one of:

- `train_full`
- `train_S_amb`
- `train_S_easy`

The train-section thresholds are derived from the full train split. The reference subset is the set of training items with a unique top vote if that subset is non-empty; otherwise the whole train split is used.

## ChaosNLI smoke test

A compact SNLI-only smoke test is:

```bash
LR_GRID_A="1e-3" \
LR_GRID_B="1e-3" \
LR_GRID_C="1e-3" \
SOURCE_SUBSETS="snli" \
RUN_IDS="0" \
HP_SEEDS="0" \
./chaosnli_snli.sh
```

This runs one hyperparameter-search seed, one run id, and one learning-rate candidate per mode.

---

## Running the higher-level train-section protocol

The launcher

```bash
./chaosnli_protocol.sh
```

runs several train sections under a shared output root.

By default it sweeps:

- `PRIMARY_HEADS="linear"`
- `PRIMARY_TRAIN_SECTIONS="train_full train_S_amb train_S_easy"`

For each requested train section, it sets:

- `TRAIN_SECTION=<that section>`
- `OUT_BASE=<shared root>/<that section>`

and then calls `chaosnli.sh`.

A typical custom call is:

```bash
PRIMARY_HEADS="linear mlp" \
PRIMARY_TRAIN_SECTIONS="train_full train_S_amb train_S_easy" \
OUT_ROOT="out/real_nlp/chaosnli_protocol_custom" \
./chaosnli_protocol.sh
```

---

## Running the repository-wide workflow

The repository-wide workflow is driven by:

```bash
./FULL_REPRO.sh
```

This is the main reproducibility script for the current repository state.

## What `FULL_REPRO.sh` does

`FULL_REPRO.sh` executes seven stages.

### Stage 1: cleanup

It optionally runs:

```bash
python3 tools/cleanup_repo.py --root . --include-out --yes
```

Important: with the default settings, this removes regenerable artifacts including:

- `.venv`
- `build/`
- compiled extension files
- `data/chaosnli`
- embedding-cache directories matching `out/chaosnli_emb*`
- the whole `out/` directory

Safety check: if `DO_CLEAN=1`, then `DO_INSTALL` must also be `1`.

### Stage 2: environment setup and C++ build

It creates `.venv` if needed, installs the Python dependencies, and builds the C++ extension.

### Stage 3: ChaosNLI embedding precomputation

It runs:

```bash
bash ./build_chaosnli_embeddings.sh
```

with the configured dataset, split, and embedding parameters.

### Stage 4: ChaosNLI train-section sweep

It runs:

```bash
bash ./chaosnli_protocol.sh
```

with the configured train-section sweep parameters.


### Stage 5: ChaosNLI tables and protocol exports

It builds:

- per-protocol ChaosNLI results tables,
- a merged article-facing train-section table,
- a CSV summary of saved thresholds,
- a compact export of ChaosNLI section sizes,
- exported protocol artifacts.

By default this stage writes into:

- `out/real_nlp/chaosnli/train_section_sweep/article/`
- `out/real_nlp/chaosnli/train_section_sweep/export_protocol/`
- `out/real_nlp/chaosnli/article/`


### Stage 6: synthetic top-k pipeline

It runs:

```bash
bash ./run_topk.sh
```

### Stage 7: synthetic top-k summary table

It runs:

```bash
python3 tools/build_topk_accuracy_summary_table.py ...
```

to build the final cross-configuration synthetic table.

## Main `FULL_REPRO.sh` configuration variables

The most important environment variables are:

- cleanup/install:
  - `DO_CLEAN`
  - `DO_INSTALL`

- ChaosNLI data/splits:
  - `DATASET_URL`
  - `DATA_ROOT`
  - `EMB_DIR`
  - `SOURCE_SUBSETS`
  - `SPLIT_SEED`
  - `TRAIN_FRAC`
  - `VAL_FRAC`

- ChaosNLI possibilistic construction:
  - `PI_EPS`
  - `TIE_TOL`
  - `EPS_CAP`
  - `MAX_TRAIN_SAMPLES`
  - `TRAIN_SUBSET_SEED`
  - `EVAL_APPLY_KEEP_FILTER`

- embedding configuration:
  - `EMBEDDING_MODEL`
  - `EMBEDDING_BATCH_SIZE`
  - `EMBEDDING_MAX_LENGTH`
  - `EMBEDDING_STORAGE_DTYPE`

- train-section sweep:
  - `CHAOSNLI_OUT_BASE`
  - `CHAOSNLI_TRAIN_SECTION_OUT_ROOT`
  - `CHAOSNLI_TRAIN_SECTIONS`
  - `CHAOSNLI_TRAIN_SECTION_HEADS`

- synthetic pipeline:
  - `TOPK_ROOT`
  - `TOPK_TABLE_OUTDIR`

## A reduced `FULL_REPRO.sh` example

For a lighter run that keeps the existing environment and skips the synthetic benchmark:

```bash
DO_CLEAN=0 \
DO_INSTALL=0 \
DO_TOPK=0 \
DO_TOPK_TABLES=0 \
./FULL_REPRO.sh
```

---

## ChaosNLI data protocol details

## Raw data parsing

The raw ChaosNLI loader:

- downloads and extracts the official archive if needed,
- reads `chaosNLI_snli.jsonl` and/or `chaosNLI_mnli_m.jsonl`,
- normalizes label names,
- derives deterministic majority labels from vote counts when needed,
- prefixes sample ids with the subset name, e.g. `snli::...` or `mnli::...`.

This namespacing prevents id collisions when multiple subsets are loaded together.

## Split construction

The split builder creates deterministic train/validation/test splits by:

- stratifying on the majority-vote class,
- ordering items within each class by a stable hash of `(split_seed, uid)`,
- taking floor-based train and validation counts per class,
- assigning the remainder to test.

The default FULL_REPRO protocol uses `SPLIT_SEED=13`, `TRAIN_FRAC=0.80`, and `VAL_FRAC=0.10`.

## Vote-derived fields

For each processed ChaosNLI item, the loader computes:

- `y`
- `plausible_mask`
- `top_votes`
- `second_votes`
- `top_margin`
- `pi`
- `sigma`
- `tilde_pi`
- `underline`
- `overline`
- `dot_p`
- `vote_p`

These are then attached to cached embeddings to produce compact training samples.

## Evaluation keep-filter

Training always applies the keep filter.

Validation and test apply it only when:

```bash
--eval-apply-keep-filter
```

or equivalently

```bash
EVAL_APPLY_KEEP_FILTER=1
```

is enabled.

---

## Variable mapping (paper notation to code)

This section keeps the paper-to-code correspondence explicit.

## Synthetic benchmark

### Problem size

- paper `$n$` (number of classes) ↔ code `n_classes`  
  CLI flag: `--n-classes`

- paper `$d$` (feature dimension) ↔ code `d`  
  CLI flag: `--d`

### Data geometry and noise

- paper `$\beta$` (prototype scale) ↔ code `class_sep`  
  CLI flag: `--class-sep`

- paper input noise scale ↔ code `x_noise`  
  CLI flag: `--x-noise`

- noise in $\alpha(x)$ ↔ code `alpha_noise`  
  CLI flag: `--alpha-noise`

### Possibility construction

- paper `$\alpha$` ↔ code `alpha`  
  CLI flag: `--alpha`

- strict positivity floor for $\pi$ ↔ code `pi_eps`  
  CLI flag: `--pi-eps`

- stair-step decrement ↔ code `pi_stair_step`  
  CLI flag: `--pi-stair-step`

- number of explicit stair levels ↔ code `pi_stair_m`  
  CLI flag: `--pi-stair-m`

### KL-box construction

- paper `$\varepsilon_{\mathrm{cap}}$` ↔ code `eps_cap`  
  CLI flag: `--eps-cap`

- tie-handling tolerance ↔ code `tie_tol`  
  CLI flag: `--tie-tol`

### Projection and numerics

- Dykstra stopping tolerance ↔ code `proj_tau_train`  
  CLI flags: `--proj-tau-train`, `--proj-tau`

- maximum Dykstra cycles ↔ code `proj_K_train`  
  CLI flags: `--proj-K-train`, `--proj-kmax`

- log-domain clipping constant ↔ code `log_clip_eps`  
  CLI flag: `--log-clip-eps`

### Training

- weight decay ↔ code `weight_decay`  
  CLI flag: `--weight-decay`

- epochs ↔ code `epochs`  
  CLI flag: `--epochs`

- batch size ↔ code `batch`  
  CLI flag: `--batch`

- learning rates ↔ code `lr_A`, `lr_B`

## ChaosNLI pipeline

### Data protocol

- source subsets ↔ code `source_subsets`  
  CLI flag: `--source-subsets`

- split seed ↔ code `split_seed`  
  CLI flag: `--split-seed`

- train fraction ↔ code `train_frac`  
  CLI flag: `--train-frac`

- validation fraction ↔ code `val_frac`  
  CLI flag: `--val-frac`

- fitted train section ↔ code `train_section`  
  CLI flag: `--train-section`

- optional train subsampling ↔ code `max_train_samples`  
  CLI flag: `--max-train-samples`

- train subsampling seed ↔ code `train_subset_seed`  
  CLI flag: `--train-subset-seed`

### Embeddings

- encoder model ↔ code `embedding_model`  
  CLI flag: `--embedding-model`

- embedding batch size ↔ code `embedding_batch_size`  
  CLI flag: `--embedding-batch-size`

- maximum token length ↔ code `embedding_max_length`  
  CLI flag: `--embedding-max-length`

- cache storage dtype ↔ code `embedding_storage_dtype`  
  CLI flag: `--embedding-storage-dtype`

### Selection protocol

- validation section used for model selection ↔ code `selection_split`  
  CLI flag: `--selection-split`

- paired test section ↔ saved as `selection_test_split` in run outputs

### Possibilistic construction

- strict positivity floor ↔ code `pi_eps`  
  CLI flag: `--pi-eps`

- tie-handling tolerance ↔ code `tie_tol`  
  CLI flag: `--tie-tol`

- upper cap for wide gap constraints ↔ code `eps_cap`  
  CLI flag: `--eps-cap`

### Projection and training

- Dykstra stopping tolerance ↔ code `proj_tau`  
  CLI flag: `--proj-tau`

- maximum Dykstra cycles ↔ code `proj_kmax`  
  CLI flag: `--proj-kmax`

- log clipping ↔ code `log_clip_eps`  
  CLI flag: `--log-clip-eps`

- model head ↔ code `head`  
  CLI flag: `--head`

- MLP hidden dimension ↔ code `mlp_hidden_dim`  
  CLI flag: `--mlp-hidden-dim`

- MLP dropout ↔ code `mlp_dropout`  
  CLI flag: `--mlp-dropout`

- active modes ↔ code `active_modes`  
  CLI flag: `--active-modes`

- learning rates ↔ code `lr_A`, `lr_B`, `lr_C`

---

## Useful checks

## Embedding-directory canonicalization

The embedding directory is canonicalized with respect to subset order. This should print the same directory twice:

```bash
python3 - <<'PY'
from nlpbench.chaosnli.loader import resolve_embedding_dir

a = resolve_embedding_dir(
    "out/chaosnli_emb",
    source_subsets=["snli", "mnli"],
    split_seed=13,
    train_frac=0.8,
    val_frac=0.1,
    embedding_model="roberta-base",
    embedding_max_length=128,
    embedding_storage_dtype="float16",
)

b = resolve_embedding_dir(
    "out/chaosnli_emb",
    source_subsets=["mnli", "snli"],
    split_seed=13,
    train_frac=0.8,
    val_frac=0.1,
    embedding_model="roberta-base",
    embedding_max_length=128,
    embedding_storage_dtype="float16",
)

print(a)
print(b)
assert a == b
print("OK")
PY
```

## Parsing subset specifications

The subset parser accepts both comma-separated and space-separated forms:

```bash
python3 - <<'PY'
from nlpbench.chaosnli.raw import parse_source_subsets

print(parse_source_subsets("snli,mnli"))
print(parse_source_subsets("mnli snli"))
print(parse_source_subsets(["snli", "mnli"]))
PY
```

## Cleanup dry run

Before using the destructive cleanup in `FULL_REPRO.sh`, you can inspect what would be removed:

```bash
python3 tools/cleanup_repo.py --root . --include-out
```

---

## Outputs

Across the repository, the main generated artifacts are:

- JSON logs for single runs,
- NumPy arrays for saved test predictions,
- CSV summary files,
- LaTeX tables.

### ChaosNLI protocol export artifacts

`tools/export_chaosnli_protocol.py` writes:

- `chaosnli_split_manifest.csv`
- `chaosnli_split_manifest.jsonl`
- `chaosnli_thresholds.json`
- `chaosnli_slice_membership.csv`
- `chaosnli_slice_membership.jsonl`
- `chaosnli_protocol_metadata.json`

### Train-section and article artifacts

`FULL_REPRO.sh` stage 5 writes, by default:

- `out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_train_section_val_full.tex`
- `out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_train_section_val_S_amb.tex`
- `out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_train_section_val_S_easy.tex`
- `out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_train_section_article_table.tex`
- `out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_train_section_article_rows.csv`
- `out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_thresholds_summary.csv`
- `out/real_nlp/chaosnli/article/chaosnli_slice_sizes.tex`
- `out/real_nlp/chaosnli/article/chaosnli_slice_sizes.csv`

### Synthetic article artifact

The final synthetic summary table is written to:

- `out/big_tables/big_all_test_acc_topk.tex`