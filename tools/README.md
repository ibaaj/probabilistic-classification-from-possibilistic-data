# tools

Utility scripts for cleanup, aggregation, extraction, and table building.

These scripts are mostly used after experiments have run. They read saved outputs, build compact summaries, and export reproducibility artifacts.

## Main scripts

- `cleanup_repo.py`  
  Removes regenerable files, caches, build artifacts, local virtual environments, and optional output directories.

- `agg_common.py`  
  Shared helpers for loading run logs, aggregating metrics, writing CSV summaries, and rendering LaTeX tables.

- `aggregate_topk_runs.py`  
  Aggregates repeated synthetic top-k runs and writes CSV and LaTeX summaries.

- `extract_topk_json.py`  
  Extracts selected values from saved JSON logs, such as accuracies or selected learning rates.

- `audit_topk_dataset.py`  
  Audits the synthetic top-k dataset and checks invariants, feasibility, tie handling, oracle replay, and basic distribution sanity.

- `build_topk_accuracy_summary_table.py`  
  Builds one larger LaTeX summary table from aggregated top-k CSV files.

- `build_chaosnli_results_table.py`  
  Builds a compact ChaosNLI LaTeX table directly from final run JSON files, with optional CSV export.

- `build_chaosnli_results_table.py`  
  Builds a compact ChaosNLI LaTeX results table directly from final run JSON files, with optional CSV export.

- `chaosnli_size_sweep_article_table.py`  
  Thin wrapper that dispatches to `chaosnli_train_section_article_table.py` for the merged train-section table.

- `chaosnli_train_section_article_table.py`  
  Builds one merged LaTeX table for the ChaosNLI train-section study from the per-protocol CSV summaries.

- `extract_chaosnli_thresholds.py`  
  Finds saved ChaosNLI threshold JSON files and summarizes or exports them. Accepts both `chaosnli_thresholds.json` and the legacy `chaosnli_ambiguity_thresholds.json`.

- `export_chaosnli_protocol.py`  
  Exports the exact ChaosNLI split protocol, thresholds, slice membership, and metadata.

- `chaosnli_slice_sizes.py`  
  Builds a compact summary of ChaosNLI train/validation/test section sizes from exported protocol artifacts, with optional LaTeX and CSV output.
  
## Notes

The train-section study is summarized from saved run outputs; the table-building scripts do not rerun experiments.