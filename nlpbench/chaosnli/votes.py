from __future__ import annotations

"""Vote-derived targets for ChaosNLI.

This file keeps the exact derivation path used by the original ChaosNLI code:

- the hard label and plausible mask come directly from vote counts
- `vote_p` is the normalized vote distribution
- `pi` is the proportional vote-to-possibility map
- `sigma`, `tilde_pi`, `underline`, `overline`, and `dot_p` are derived through
  the KL-box utilities already used elsewhere in the project
"""

from dataclasses import dataclass

import numpy as np

from klbox.gaps import choose_gap_parameters
from klbox.possibility import antipignistic_reverse_mapping, compute_possibility_order

from .constants import LABEL_TO_INDEX
from .schema import ChaosNLIItem, ChaosNLIRawItem

@dataclass(frozen=True)
class VoteDerivationConfig:
    pi_eps: float = 1e-6
    tie_tol: float = 0.0
    eps_cap: float = 0.05


@dataclass(frozen=True)
class VoteFields:
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


def stable_argsort_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    indices = np.arange(values.size, dtype=np.int64)
    return np.lexsort((indices, -values))

def summarize_votes(
    votes: np.ndarray,
    *,
    y_override: int | None = None,
) -> tuple[int, np.ndarray, int, int, int]:
    counts = np.asarray(votes, dtype=np.int64)
    order = stable_argsort_desc(counts)
    default_y = int(order[0])

    y = default_y if y_override is None else int(y_override)
    if y < 0 or y >= counts.size:
        raise ValueError(f"y_override out of range: {y}")

    sorted_counts = np.sort(counts)[::-1]
    top_votes = int(sorted_counts[0]) if sorted_counts.size >= 1 else 0
    second_votes = int(sorted_counts[1]) if sorted_counts.size >= 2 else 0

    plausible_mask = counts > 0
    plausible_mask[y] = True
    return y, plausible_mask.astype(bool), top_votes, second_votes, int(top_votes - second_votes)


def counts_to_vote_distribution(votes: np.ndarray, y: int) -> np.ndarray:
    counts = np.asarray(votes, dtype=np.float64)
    total = float(counts.sum())
    if total > 0.0:
        return counts / total

    out = np.zeros(counts.size, dtype=np.float64)
    out[int(y)] = 1.0
    return out


def counts_to_pi(votes: np.ndarray, *, y: int, pi_eps: float) -> np.ndarray:
    counts = np.asarray(votes, dtype=np.float64)
    pi = np.full(counts.size, float(pi_eps), dtype=np.float64)
    max_count = float(np.max(counts)) if counts.size else 0.0

    if max_count > 0.0:
        positive = counts > 0.0
        pi[positive] = np.maximum(counts[positive] / max_count, float(pi_eps))

    if max_count <= 0.0:
        pi[int(y)] = 1.0

    return pi



def derive_vote_fields(
    votes: np.ndarray,
    *,
    config: VoteDerivationConfig,
    y_override: int | None = None,
) -> VoteFields:
    y, plausible_mask, top_votes, second_votes, top_margin = summarize_votes(
        votes,
        y_override=y_override,
    )
    vote_p = counts_to_vote_distribution(votes, y=y)
    pi = counts_to_pi(votes, y=y, pi_eps=float(config.pi_eps))

    order = compute_possibility_order(pi)
    anti = antipignistic_reverse_mapping(order)
    gaps = choose_gap_parameters(
        tilde_pi=order.tilde_pi,
        dot_g=anti.dot_g,
        tie_tol=float(config.tie_tol),
        eps_cap=float(config.eps_cap),
    )

    return VoteFields(
        y=y,
        plausible_mask=plausible_mask,
        top_votes=top_votes,
        second_votes=second_votes,
        top_margin=top_margin,
        pi=np.asarray(pi, dtype=np.float64),
        sigma=np.asarray(order.sigma, dtype=np.int64),
        tilde_pi=np.asarray(order.tilde_pi, dtype=np.float64),
        underline=np.asarray(gaps.underline, dtype=np.float64),
        overline=np.asarray(gaps.overline, dtype=np.float64),
        dot_p=np.asarray(anti.dot_p, dtype=np.float64),
        vote_p=np.asarray(vote_p, dtype=np.float64),
    )



def build_items_for_split(
    raw_items: list[ChaosNLIRawItem],
    *,
    config: VoteDerivationConfig,
    apply_keep_filter: bool,
) -> tuple[list[ChaosNLIItem], dict[str, int]]:
    items: list[ChaosNLIItem] = []
    stats = {"scanned": 0, "kept": 0, "skipped_keep_filter": 0}

    for raw in raw_items:
        stats["scanned"] += 1
        votes = np.asarray(raw.votes, dtype=np.int16)

        if bool(apply_keep_filter) and int(votes.sum()) <= 0:
            stats["skipped_keep_filter"] += 1
            continue

        y_override = int(LABEL_TO_INDEX[raw.majority_label])
        fields = derive_vote_fields(
            votes,
            config=config,
            y_override=y_override,
        )
        items.append(
            ChaosNLIItem(
                uid=raw.uid,
                original_uid=raw.original_uid,
                subset=raw.subset,
                premise=raw.premise,
                hypothesis=raw.hypothesis,
                old_label=raw.old_label,
                majority_label=raw.majority_label,
                votes=votes,
                n_raters=raw.n_raters,
                entropy=raw.entropy,
                y=fields.y,
                plausible_mask=fields.plausible_mask,
                top_votes=fields.top_votes,
                second_votes=fields.second_votes,
                top_margin=fields.top_margin,
                pi=fields.pi,
                sigma=fields.sigma,
                tilde_pi=fields.tilde_pi,
                underline=fields.underline,
                overline=fields.overline,
                dot_p=fields.dot_p,
                vote_p=fields.vote_p,
            )
        )
        stats["kept"] += 1

    return items, stats
