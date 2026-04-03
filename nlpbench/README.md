# nlpbench

Helpers for the real-text benchmark pipeline.

This package contains reusable NLP utilities, mainly for the ChaosNLI experiments.

## Main files

- `embeddings.py`  
  Transformer embedding cache utilities.

- `sampling.py`  
  Deterministic sampling helpers re-exported from `common.sampling`.

- `chaosnli/`  
  ChaosNLI-specific loading, preprocessing, splits, vote-derived targets, and sample construction.

- `__init__.py`  
  Re-exports `add_chaosnli_data_args` and `load_chaosnli_splits`.

## Notes

This package prepares data and embeddings for the experiment code.