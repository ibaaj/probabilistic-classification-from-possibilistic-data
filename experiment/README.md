# experiment

Training, selection, post-hoc analysis, and aggregation code for the ChaosNLI pipeline, together with a few shared experiment-layer utilities used by the synthetic benchmark.

## Main files

- `chaosnli_cli.py`  
  Main CLI for ChaosNLI hyperparameter search and final runs.

- `chaosnli_slices.py`  
  Shared slice definitions, train-derived thresholds, validation/test protocol sections, and section-selection helpers.

- `chaosnli_analysis_utils.py`  
  Small shared helpers for vote distributions, entropy, and sample identifiers used by ambiguity/slice analysis.

- `train.py`  
  Generic training loop used by the experiment code.

- `targets.py`  
  ChaosNLI target functions for Modes A, B, and C.

- `metrics.py`  
  Evaluation metrics and constraint-violation summaries.

- `sample_adapter.py`  
  Adapts full and compact sample objects to a common constraint interface.

- `projection_common.py`  
  Shared projection-side normalization and projection-statistics utilities.

- `target_protocol.py`  
  Lightweight protocol types for target functions that expose projection statistics.

- `analyze_ambiguity.py`  
  Builds train-derived ambiguity thresholds and split-level ambiguity summaries.

- `slice_eval.py`  
  Post-hoc evaluation of saved run probabilities on the defined ambiguity slices.

- `aggregate_chaosnli_runs.py`  
  Aggregates saved ChaosNLI run JSON files.

- `aggregate_chaosnli_hp_search.py`  
  Aggregates ChaosNLI hyperparameter-search outputs.

- `aggregate_chaosnli_slice_eval.py`  
  Aggregates slice-evaluation outputs.

## Notes

This directory contains the experiment-layer logic that sits between:

- dataset loading in `nlpbench/chaosnli/`,
- projection utilities in `klbox/`,
- final run/table/export scripts in `scripts/` and `tools/`.

The main focus here is the ChaosNLI pipeline: model selection, target construction, train-derived slice protocols, post-hoc evaluation, and aggregation.