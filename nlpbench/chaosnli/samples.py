from __future__ import annotations

"""Attach embeddings to processed ChaosNLI items."""

from typing import Mapping

import numpy as np

from .schema import ChaosNLIItem, ChaosNLISample



def items_to_samples(
    items: list[ChaosNLIItem],
    *,
    id_to_row: Mapping[str, int],
    embs_arr: np.ndarray,
) -> tuple[list[ChaosNLISample], int]:
    """Convert processed items into compact samples.

    Returns the sample list and the number of items skipped because their
    embedding row was missing.
    """
    samples: list[ChaosNLISample] = []
    missing = 0

    for item in items:
        row = id_to_row.get(str(item.uid))
        if row is None:
            missing += 1
            continue

        samples.append(
            ChaosNLISample(
                sample_id=str(item.uid),
                x=np.asarray(embs_arr[row], dtype=np.float64),
                y=int(item.y),
                dot_p=np.asarray(item.dot_p, dtype=np.float64),
                vote_p=np.asarray(item.vote_p, dtype=np.float64),
                sigma=np.asarray(item.sigma, dtype=np.int64),
                tilde_pi=np.asarray(item.tilde_pi, dtype=np.float64),
                underline=np.asarray(item.underline, dtype=np.float64),
                overline=np.asarray(item.overline, dtype=np.float64),
                plausible_mask=np.asarray(item.plausible_mask, dtype=bool),
                top_votes=int(item.top_votes),
                second_votes=int(item.second_votes),
                top_margin=int(item.top_margin),
                n_raters=int(item.n_raters),
                votes=np.asarray(item.votes, dtype=np.int16),
            )
        )

    return samples, missing
