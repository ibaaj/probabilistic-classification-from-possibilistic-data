# QUICK SETUP AND RUN

This note gives the shortest path to install the repository dependencies and run the main workflows.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy torch tqdm transformers pybind11
python3 klbox/setup_cpp.py build_ext --inplace
```

## Standard full run

This runs the default seven-stage reproducibility pipeline defined in `FULL_REPRO.sh`:

```bash
./FULL_REPRO.sh
```

By default, this includes:

1. cleanup,
2. environment setup and C++ build,
3. ChaosNLI embedding precomputation,
4. ChaosNLI train-section sweep,
5. ChaosNLI tables and protocol exports,
6. synthetic top-k runs,
7. synthetic top-k summary tables.

## Reuse existing environment and embeddings

Use this when `.venv` already exists and the ChaosNLI embedding cache is already built:

```bash
DO_CLEAN=0 DO_INSTALL=0 DO_EMBEDDINGS=0 ./FULL_REPRO.sh
```

## ChaosNLI train-section sweep only

This keeps the existing environment, skips embedding recomputation, skips the synthetic benchmark, and runs only the ChaosNLI train-section workflow plus its tables and exports:

```bash
DO_CLEAN=0 \
DO_INSTALL=0 \
DO_EMBEDDINGS=0 \
DO_CHAOSNLI_EXPORTS=1 \
DO_CHAOSNLI_TRAIN_SECTION_SWEEP=1 \
DO_CHAOSNLI_TRAIN_SECTION_TABLES=1 \
DO_TOPK=0 \
DO_TOPK_TABLES=0 \
CHAOSNLI_TRAIN_SECTIONS="train_full train_S_amb train_S_easy" \
CHAOSNLI_TRAIN_SECTION_HEADS="linear" \
bash ./FULL_REPRO.sh
```

## Main train-section outputs

Typical outputs for the ChaosNLI train-section sweep are:

```text
out/real_nlp/chaosnli/train_section_sweep/
out/real_nlp/chaosnli/train_section_sweep/article/chaosnli_train_section_article_table.tex
```

## Notes

- `DO_CLEAN=1` requires `DO_INSTALL=1`, because cleanup removes `.venv`.
- The default embedding cache base directory is `out/chaosnli_emb`.
- The default ChaosNLI output base is `out/real_nlp/chaosnli`.
- The default synthetic aggregation root is `out/agg_gapwide_stair`.
- The default synthetic summary-table output directory is `out/big_tables`.
- The default ChaosNLI protocol in `FULL_REPRO.sh` uses split seed 13 and fractions 0.80 / 0.10.
