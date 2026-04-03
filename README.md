# Probabilistic classification from possibilistic data: computing Kullback–Leibler projection with a possibility distribution

This repository contains the code for the experiments reported in the following paper:

> **Probabilistic classification from possibilistic data: computing Kullback–Leibler projection with a possibility distribution**
> Ismaïl Baaj and Pierre Marquis
> [arXiv:2604.01939](https://arxiv.org/abs/2604.01939)

```bibtex
@misc{baaj2026probabilistic,
      title={Probabilistic classification from possibilistic data: computing
             Kullback-Leibler projection with a possibility distribution},
      author={Isma\"{i}l Baaj and Pierre Marquis},
      year={2026},
      eprint={2604.01939},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.01939},
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Overview

Given a finite set of classes $\mathcal{Y}$ and a normalized possibility distribution $\pi$ on $\mathcal{Y}$, we construct a non-empty closed convex set $\mathcal{F}^{\mathrm{box}}(\pi) \subseteq \Delta_n$ of admissible probability distributions by combining two requirements:

1. **Dominance constraints.** For every event $A \subseteq Y$, the probability measure $P$ satisfies $N(A) \le P(A) \le \Pi(A)$, where $\Pi$ and $N$ are the possibility and necessity measures induced by $\pi$.

2. **Shape constraints.** The qualitative ordering of $\pi$ is preserved: $\pi_k \ge \pi_{k'} \Longleftrightarrow p_k \ge p_{k'}$ for all classes $k, k'$.

Given a strictly positive probability vector $q \in \Delta_n$ produced by a classifier, we compute its Kullback–Leibler projection onto $\mathcal{F}^{\mathrm{box}}(\pi)$:

$$p^\star = \arg\min_{p \in \mathcal{F}^{\mathrm{box}}(\pi)} D_{\mathrm{KL}}(p \| q).$$

The projection is obtained iteratively by Dykstra's algorithm with Bregman projections associated with the negative entropy. Explicit closed-form projections onto each constraint set (dominance constraints, shape constraints) are derived in the paper and implemented here.

The projection $p^\star$ serves as a training target: the per-instance loss is $\ell(\theta; x, \pi) = D_{\mathrm{KL}}(p^\star \| q_\theta(x))$, which quantifies the smallest KL adjustment of $q_\theta(x)$ needed to satisfy the constraints induced by $\pi$.

---

## Experiments

The repository implements three experiments from the paper.

### Experiment 1: Empirical evaluation of Dykstra's algorithm (Section 5.2)

Empirical evaluation of Dykstra's algorithm on synthetic instances with $n = 100$ classes. For each of 100 random runs, a strictly positive possibility distribution $\pi$ and a reference probability vector $q$ are drawn at random. The algorithm is run for several tolerance levels $\tau \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-6}, 10^{-8}\}$ and cycle budgets $K_{\max} \in \{10^3, 10^4, 5 \cdot 10^4\}$. The convergence rate, cycle count, final constraint violation, and computation time are recorded.

### Experiment 2: Synthetic learning with possibilistic supervision (Section 5.3)

A controlled multi-class classification setting with $n = 20$ classes, where each training instance is a pair $(x, \pi)$ with $x \in \mathbb{R}^d$ a feature vector and $\pi$ a possibility distribution on $\mathcal{Y}$. Two models are compared:

- **Model A (projection target):** the target is $p^\star(x, \pi) = \arg\min_{p \in \mathcal{F}^{\mathrm{box}}(\pi)} D_{\mathrm{KL}}(p \| q^A(x))$, recomputed from the current prediction at each training step.
- **Model B (fixed target):** the target is the antipignistic probability $\dot{p}(\pi) \in \Delta_n$ obtained from $\pi$ by the reverse mapping of Section 2.2 of the paper.

The comparison varies over feature dimensions $d \in \{30, 80, 150\}$, training-set sizes $N_{\mathrm{tr}} \in \{200, 500, 1000\}$, and ambiguity levels $\alpha \in \{0.4, 0.6, 0.8, 0.95\}$.

### Experiment 3: ChaosNLI (Section 5.4)

A natural language inference task based on the ChaosNLI dataset (Nie et al., 2020), which provides multiply-annotated examples from SNLI and MultiNLI with $n = 3$ classes (entailment, neutral, contradiction). Three training objectives are compared:

- **Model A (projection target):** KL projection of the current prediction onto $\mathcal{F}^{\mathrm{box}}(\pi)$, where $\pi$ is derived from annotator vote counts.
- **Model B (antipignistic target):** the fixed probability $\dot{p}(\pi)$ derived from the possibility distribution.
- **Model C (vote-proportion target):** the normalized vote proportions $\bar{v}$ used directly as a soft target.

The comparison includes training on the full split, on an ambiguity-focused subset ($\mathcal{S}_{\mathrm{amb}}$), and on an easy subset ($\mathcal{S}_{\mathrm{easy}}$), with model selection on three validation sections and final evaluation on three test sections.

---

## Repository structure

### Core packages

| Directory | Contents |
|---|---|
| `klbox/` | Possibility ordering, antipignistic reverse mapping, gap parameter selection, linear feasibility systems, Bregman projections (Propositions 7 and 8 of the paper), Python and C++ implementations of Dykstra's algorithm, and numerical helpers. |
| `topk/` | Synthetic benchmark: data generation, model heads, projection and fixed targets, training loop, experiment runners, and hyperparameter search. |
| `nlpbench/` | NLP utilities: cached transformer embeddings (`nlpbench/embeddings.py`), deterministic sampling (`nlpbench/sampling.py`), and the ChaosNLI data loader and split protocol (`nlpbench/chaosnli/`). |
| `experiment/` | ChaosNLI experiment layer: CLI, target definitions (Mode A projection, Mode B antipignistic, Mode C vote), training loop, evaluation metrics, ambiguity analysis, slice evaluation, and run aggregation. |
| `common/` | Shared JSON/CSV I/O helpers and deterministic sampling utilities. |

### Shell launchers

| Script | Purpose |
|---|---|
| `run_topk.sh` | End-to-end launcher for the synthetic benchmark (Experiment 2) and the Dykstra convergence sweep (Experiment 1). |
| `build_chaosnli_embeddings.sh` | Precomputes and caches transformer embeddings for a ChaosNLI split protocol. |
| `chaosnli.sh` | Main ChaosNLI launcher for one training-section setting across all requested validation-selection protocols. |
| `chaosnli_snli.sh`, `chaosnli_mnli.sh`, `chaosnli_all.sh` | Convenience wrappers setting `SOURCE_SUBSETS` and `OUT_BASE` before calling `chaosnli.sh`. |
| `chaosnli_protocol.sh` | Higher-level launcher sweeping several train sections under a shared output root. |
| `FULL_REPRO.sh` | Repository-wide reproducibility script executing all stages. |

### Table and export tools

| Script | Purpose |
|---|---|
| `tools/build_chaosnli_results_table.py` | Builds per-protocol ChaosNLI summary tables from saved run JSONs. |
| `tools/chaosnli_train_section_article_table.py` | Merges per-protocol CSVs into one article-facing train-section table. |
| `tools/chaosnli_slice_sizes.py` | Exports train/validation/test section sizes with optional LaTeX output. |
| `tools/export_chaosnli_protocol.py` | Exports split manifests, slice memberships, thresholds, and protocol metadata. |
| `tools/extract_chaosnli_thresholds.py` | Collects saved thresholds into a compact summary table. |
| `tools/aggregate_topk_runs.py` | Aggregates repeated synthetic runs and writes CSV/LaTeX summaries. |
| `tools/build_topk_accuracy_summary_table.py` | Builds the synthetic accuracy table (Table 5 of the paper). |

---

## Requirements

Python 3.10+ is required (the codebase uses `str | Path` and `list[str]` annotations).

Dependencies:

- `numpy`
- `torch`
- `tqdm`
- `transformers`
- `pybind11`

The projection algorithm uses a compiled C++ backend (`klbox/_dykstra_cpp.cpp`) interfaced via pybind11.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy torch tqdm transformers pybind11
python3 klbox/setup_cpp.py build_ext --inplace
```

To verify the installation:

```bash
python3 -m compileall common experiment klbox nlpbench topk tools
python3 synthetic_cli.py dykstra-sweep --n 50 --runs 5 --tau 1e-6 1e-8 --Kmax 1000
```

---

## Reproducing the experiments

### Full reproducibility

```bash
./FULL_REPRO.sh
```

This executes all stages: environment setup, C++ build, ChaosNLI embedding precomputation, ChaosNLI train-section sweep, table generation, synthetic benchmark, and synthetic summary tables.

### Experiment 1 only (Dykstra convergence evaluation)

The convergence sweep is included at the end of `run_topk.sh`. A standalone run is:

```bash
python3 synthetic_cli.py dykstra-sweep \
    --n 100 --runs 100 \
    --tau 1e-2 1e-3 1e-4 1e-6 1e-8 \
    --Kmax 1000 10000 50000
```

### Experiment 2 only (synthetic benchmark)

```bash
./run_topk.sh
```

Default grid: $d \in \{30, 80, 150\}$, $N_{\mathrm{tr}} \in \{200, 500, 1000\}$, $\alpha \in \{0.4, 0.6, 0.8, 0.95\}$, 10 runs per configuration. The class-separation parameter $\beta$ is set as a function of $d$: $\beta = 1.5$ for $d = 30$, $\beta = 0.9$ for $d = 80$, $\beta = 0.6$ for $d = 150$.

Smoke test (one configuration, one run):

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

### Experiment 3 only (ChaosNLI)

Precompute embeddings:

```bash
SOURCE_SUBSETS="snli mnli" ./build_chaosnli_embeddings.sh
```

Run the train-section protocol sweep:

```bash
./chaosnli_protocol.sh
```

Smoke test (SNLI only, one run, one learning rate per mode):

```bash
LR_GRID_A="1e-3" \
LR_GRID_B="1e-3" \
LR_GRID_C="1e-3" \
SOURCE_SUBSETS="snli" \
RUN_IDS="0" \
HP_SEEDS="0" \
./chaosnli_snli.sh
```

---

## Notation correspondence

The following table maps the main symbols used in the paper to their code counterparts.

### Possibility distribution and KL-box

| Paper | Code | CLI flag |
|---|---|---|
| $\pi$ | `pi` | — |
| $\sigma$ (sorting permutation) | `sigma` | — |
| $\tilde{\pi}$ (sorted possibility) | `tilde_pi` | — |
| $\dot{p}$ (antipignistic probability) | `dot_p` | — |
| $\dot{g}_r$ (adjacent gaps of $\dot{p}$) | `dot_g` | — |
| $\underline{\delta}_r$ (lower gap) | `underline` | — |
| $\overline{\delta}_r$ (upper gap) | `overline` | — |
| $\rho_\pi$ (strict-positivity floor) | `pi_eps` | `--pi-eps` |
| $\varepsilon_{\mathrm{cap}}$ (gap epsilon cap) | `eps_cap` | `--eps-cap` |
| Tie-handling tolerance | `tie_tol` | `--tie-tol` |

### Dykstra's algorithm

| Paper | Code | CLI flag |
|---|---|---|
| $\tau$ (stopping tolerance) | `proj_tau` / `proj_tau_train` | `--proj-tau` |
| $K_{\max}$ (maximum cycles) | `proj_K_train` / `proj_kmax` | `--proj-kmax` |
| Log-domain clipping $\varepsilon$ | `log_clip_eps` | `--log-clip-eps` |

### Synthetic benchmark (Experiment 2)

| Paper | Code | CLI flag |
|---|---|---|
| $n$ (number of classes) | `n_classes` | `--n-classes` |
| $d$ (feature dimension) | `d` | `--d` |
| $\alpha$ (plausibility level) | `alpha` | `--alpha` |
| $\beta$ (prototype scale) | `class_sep` | `--class-sep` |
| $s$ (input noise) | `x_noise` | `--x-noise` |
| $s_\alpha$ (annotation noise) | `alpha_noise` | `--alpha-noise` |
| $\delta_\pi$ (stair step) | `pi_stair_step` | `--pi-stair-step` |
| $N_{\mathrm{tr}}$ | `train` | `--train` |
| $N_{\mathrm{te}}$ | `test` | `--test` |

### ChaosNLI (Experiment 3)

| Paper | Code | CLI flag |
|---|---|---|
| Source subsets | `source_subsets` | `--source-subsets` |
| Split seed | `split_seed` | `--split-seed` |
| Train fraction | `train_frac` | `--train-frac` |
| Validation fraction | `val_frac` | `--val-frac` |
| Fitted train section | `train_section` | `--train-section` |
| Validation selection section | `selection_split` | `--selection-split` |
| Embedding model | `embedding_model` | `--embedding-model` |
| Model head | `head` | `--head` |
| Active modes (A, B, C) | `active_modes` | `--active-modes` |

---

## ChaosNLI data protocol

### Splits

The loader downloads the ChaosNLI release archive, reads `chaosNLI_snli.jsonl` and/or `chaosNLI_mnli_m.jsonl`, and constructs deterministic train/validation/test splits stratified by majority-vote class. Items are ordered within each class by a stable hash of `(split_seed, uid)`. The default protocol uses `split_seed = 13`, `train_frac = 0.80`, `val_frac = 0.10`.

### Possibilistic annotation

For each item with vote counts $v = (v_y)_{y \in \mathcal{Y}}$, the possibility distribution is:

$$\pi_y = \max\!\left(\frac{v_y}{v_{\max}},\, \rho_\pi\right),$$

where $v_{\max} = \max_y v_y$ and $\rho_\pi = 10^{-6}$. The admissible set $\mathcal{F}^{\mathrm{box}}(\pi)$ is then constructed with the gap parameters described in Section 5.4.2 of the paper, using $\varepsilon_{\mathrm{cap}} = 0.05$.

### Ambiguity slices

For each item, the peak vote proportion $p_{\max} = {\max}_{y} \bar{v}_{y}$ and the normalized entropy $H_{\mathrm{norm}} = -\sum_y \bar{v}_{y} \log \bar{v}_{y} / \log 3$ are computed. Fixed thresholds ($T_{\mathrm{low\text{-}peak}}$, $T_{\mathrm{high\text{-}peak}}$, $T_{\mathrm{low\text{-}H}}$, $T_{\mathrm{high\text{-}H}}$) are derived from the 30th and 70th percentiles of the unique-majority subset of the training split. The ambiguous subset $\mathcal{S}_{\mathrm{amb}}$ consists of unique-majority items with low peak and high entropy; the easy subset $\mathcal{S}_{\mathrm{easy}}$ consists of unique-majority items with high peak and low entropy.

---

## Output artifacts

The main generated artifacts are:

- **Run logs** (JSON): per-run configuration, learning rates, and per-section metrics for all active modes.
- **Saved predictions** (NumPy): test-set sample ids, labels, and probability matrices for each mode.
- **Aggregated summaries** (CSV): means and standard deviations over repeated runs, grouped by experimental configuration.
- **LaTeX tables**: formatted tables corresponding to Tables 1–3 (Dykstra convergence), Table 5 (synthetic benchmark), and Table 6 (ChaosNLI) of the paper.

---

## `FULL_REPRO.sh` configuration

The main environment variables are listed below. All have defaults matching the paper's experimental protocol.

| Variable | Default | Purpose |
|---|---|---|
| `DO_CLEAN` | `1` | Run cleanup before installation |
| `DO_INSTALL` | `1` | Create venv and build C++ extension |
| `DO_EMBEDDINGS` | `1` | Precompute ChaosNLI embeddings |
| `DO_CHAOSNLI_TRAIN_SECTION_SWEEP` | `1` | Run the ChaosNLI train-section protocol |
| `DO_CHAOSNLI_TRAIN_SECTION_TABLES` | `1` | Build ChaosNLI article tables |
| `DO_CHAOSNLI_EXPORTS` | `1` | Export thresholds, slice sizes, protocol artifacts |
| `DO_TOPK` | `1` | Run the synthetic benchmark |
| `DO_TOPK_TABLES` | `1` | Build the synthetic summary table |
| `SOURCE_SUBSETS` | `snli mnli` | ChaosNLI source subsets |
| `SPLIT_SEED` | `13` | Deterministic split seed |
| `TRAIN_FRAC` | `0.8` | Training fraction |
| `VAL_FRAC` | `0.1` | Validation fraction |
| `PI_EPS` | `1e-6` | Strict-positivity floor for $\pi$ |
| `EPS_CAP` | `0.05` | Gap epsilon cap |
| `EMBEDDING_MODEL` | `roberta-base` | Sentence encoder |

A reduced run that skips the synthetic benchmark:

```bash
DO_CLEAN=0 DO_INSTALL=0 DO_TOPK=0 DO_TOPK_TABLES=0 ./FULL_REPRO.sh
```
