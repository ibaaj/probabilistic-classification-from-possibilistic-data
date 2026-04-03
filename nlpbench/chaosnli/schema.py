from __future__ import annotations

"""ChaosNLI data structures.

The split between raw and processed items is intentional:

- `ChaosNLIRawItem` stores only dataset fields still used downstream.
- `ChaosNLIItem` adds the derived target arrays used by training and analysis.
- `ChaosNLISample` is the compact object passed to the training pipeline after
  embeddings are attached.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChaosNLIRawItem:
    """One parsed row from the original JSONL files."""

    uid: str
    original_uid: str
    subset: str
    premise: str
    hypothesis: str
    old_label: str
    majority_label: str
    votes: np.ndarray
    n_raters: int
    entropy: float


@dataclass(frozen=True)
class ChaosNLIItem:
    """One dataset item after vote-derived targets are computed."""

    uid: str
    original_uid: str
    subset: str
    premise: str
    hypothesis: str
    old_label: str
    majority_label: str
    votes: np.ndarray
    n_raters: int
    entropy: float
    y: int
    plausible_mask: np.ndarray
    top_votes: int
    second_votes: int
    top_margin: int
    pi: np.ndarray
    sigma: np.ndarray
    tilde_pi: np.ndarray
    underline: np.ndarray
    overline: np.ndarray
    dot_p: np.ndarray
    vote_p: np.ndarray


@dataclass(frozen=True)
class ChaosNLISample:
    """Compact training sample with its embedding attached.

    This is the only object that the trainer needs. Its fields are deliberately
    explicit so that the target functions and evaluators do not need to rebuild
    any dataset-specific structures.
    """

    sample_id: str
    x: np.ndarray
    y: int
    dot_p: np.ndarray
    vote_p: np.ndarray
    sigma: np.ndarray
    tilde_pi: np.ndarray
    underline: np.ndarray
    overline: np.ndarray
    plausible_mask: np.ndarray
    top_votes: int
    second_votes: int
    top_margin: int
    n_raters: int
    votes: np.ndarray
