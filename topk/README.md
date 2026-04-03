# topk

Synthetic top-k benchmark code.

This package contains the synthetic experiment pipeline used in the repository: data generation, train/test splits, model heads, targets, training, repeated runs, and learning-rate search.

## Main files

- `data.py`  
  Synthetic sample generation and construction of `pi`, `dot_p`, and KL-box objects.

- `data_splits.py`  
  Deterministic train/validation/test generation from seeds.

- `config_utils.py`  
  Helper for building `TopKConfig`.

- `model.py`  
  Linear and MLP softmax heads.

- `targets.py`  
  Projection target for model A and fixed `dot_p` target for model B.

- `train.py`  
  Training and evaluation code for the synthetic benchmark.

- `hp_search.py`  
  Learning-rate search over validation seeds.

- `experiment_runner.py`  
  Runs synthetic experiments and writes structured JSON logs.

- `experiments.py`  
  Convenience re-exports for the main experiment functions.

## Notes

Aggregation and table-building scripts are in `tools/`.