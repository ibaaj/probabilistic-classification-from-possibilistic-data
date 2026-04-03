# nlpbench.chaosnli

ChaosNLI data loading and preprocessing.

This package handles the ChaosNLI pipeline used in the repository: raw-data loading, deterministic splits, vote-derived targets, embedding caches, and compact sample objects for training.

## Main files

- `loader.py`  
  Main entry point: loads subsets, builds splits, manages embedding caches, and returns train/validation/test samples.

- `raw.py`  
  Downloads, extracts, reads, and validates the raw ChaosNLI JSONL files.

- `votes.py`  
  Derives vote-based fields such as `y`, `vote_p`, `pi`, `dot_p`, and KL-box quantities.

- `splits.py`  
  Deterministic train/validation/test splitting.

- `samples.py`  
  Attaches embeddings and builds compact sample objects.

- `schema.py`  
  Data classes for raw items, processed items, and final samples.

- `text.py`  
  Defines the text sent to the embedding encoder.

- `constants.py`  
  Label names, subset names, and default dataset URL.

## Notes

- This package is specific to ChaosNLI.
- Embedding caches are automatically scoped by subset list, split settings, and encoder settings when the default cache base name is used.