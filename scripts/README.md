# scripts

Shell wrappers for the ChaosNLI pipeline.

## Main files

- `chaosnli_common.sh`  
  Shared helpers: repository root, logging, and Python dispatch.

- `chaosnli_hp_search.sh`  
  Runs ChaosNLI hyperparameter search.

- `chaosnli_run_final.sh`  
  Runs final ChaosNLI training with selected learning rates.

- `chaosnli_posthoc.sh`  
  Runs ambiguity analysis, slice evaluation, and aggregation.

## Notes

These scripts are the shell entry points used by `chaosnli.sh`.