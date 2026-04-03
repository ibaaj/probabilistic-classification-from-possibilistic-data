from __future__ import annotations

"""Main ChaosNLI loader.

This file is the public entrypoint used by the rest of the experiment code. It
keeps all dataset-specific decisions in one readable place:

- where the raw data comes from,
- how splits are built,
- how text embeddings are cached,
- how vote-derived targets are computed,
- how compact training samples are returned.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from nlpbench.embeddings import ensure_transformer_embeddings, load_embedding_cache
from nlpbench.sampling import stratified_subset_by_label
from .slices import compute_slice_stats_for_split, compute_slice_thresholds, slice_masks_from_stats

from .constants import DEFAULT_CHAOSNLI_URL, LABELS
from .raw import download_and_extract, parse_source_subsets, read_chaosnli_jsonl
from .samples import items_to_samples
from .schema import ChaosNLIItem, ChaosNLIRawItem
from .splits import split_raw_items
from .text import format_chaosnli_text
from .votes import VoteDerivationConfig, build_items_for_split

TRAIN_SECTIONS = ("train_full", "train_S_amb", "train_S_easy")


def _float_tag(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _text_tag(value: str) -> str:
    text = str(value).strip().lower()
    pieces: list[str] = []
    for char in text:
        if char.isalnum():
            pieces.append(char)
        elif char in {"-", "_", "."}:
            pieces.append(char)
        else:
            pieces.append("-")

    compact = "".join(pieces).strip("-")
    while "--" in compact:
        compact = compact.replace("--", "-")
    return compact or "default"


def _canonical_subset_key(source_subsets: list[str]) -> list[str]:
    """Canonicalize subset names for cache naming.

    This keeps cache paths stable across equivalent requests such as
    ['snli', 'mnli'] and ['mnli', 'snli'] while leaving user-facing ordering
    elsewhere unchanged.
    """
    return sorted({str(subset).strip().lower() for subset in source_subsets if str(subset).strip()})


def _normalize_train_section(value: str) -> str:
    train_section = str(value).strip()
    if train_section not in TRAIN_SECTIONS:
        allowed = ", ".join(TRAIN_SECTIONS)
        raise ValueError(f"Unsupported train_section={value!r}. Allowed values: {allowed}.")
    return train_section






def resolve_embedding_dir(
    base_dir: str | Path,
    *,
    source_subsets: list[str],
    split_seed: int,
    train_frac: float,
    val_frac: float,
    embedding_model: str,
    embedding_max_length: int,
    embedding_storage_dtype: str,
) -> Path:
    """Resolve the embedding cache directory.

    When the caller leaves the default base name unchanged, the directory is
    automatically scoped by the subset list, split protocol, and embedding
    configuration so that caches do not silently collide across different
    experimental settings.
    """
    emb_dir = Path(base_dir)
    if emb_dir.name != "chaosnli_emb":
        return emb_dir

    subset_tag = "+".join(_canonical_subset_key(source_subsets))
    split_tag = f"seed{int(split_seed)}_tr{_float_tag(train_frac)}_va{_float_tag(val_frac)}"
    encoder_tag = (
        f"enc{_text_tag(embedding_model)}_mx{int(embedding_max_length)}_dt{_text_tag(embedding_storage_dtype)}"
    )
    return emb_dir.parent / f"{emb_dir.name}_{subset_tag}_{split_tag}_{encoder_tag}"



def load_raw_items(
    *,
    dataset_url: str,
    data_root: str | Path,
    source_subsets: list[str],
) -> list[ChaosNLIRawItem]:
    """Download, extract, read, and sort the requested raw ChaosNLI items."""
    paths = download_and_extract(
        url=dataset_url,
        data_root=data_root,
        subsets=source_subsets,
    )

    raw_items: list[ChaosNLIRawItem] = []
    for subset in source_subsets:
        raw_items.extend(read_chaosnli_jsonl(paths[subset], subset=subset))
    return sorted(raw_items, key=lambda item: item.uid)



def build_split_to_text_records(
    raw_splits: dict[str, list[ChaosNLIRawItem]],
) -> dict[str, list[dict[str, str]]]:
    """Convert split raw items into the text-record structure used by embedding caches."""
    return {
        split: [{"uid": item.uid, "text": format_chaosnli_text(item)} for item in rows]
        for split, rows in raw_splits.items()
    }



def load_chaosnli_splits(args: argparse.Namespace) -> dict[str, Any]:
    """Load ChaosNLI and return the compact split payload used by experiments.

    The returned payload always includes ``train_full``. When
    ``max_train_samples`` is positive, ``train`` contains the deterministic
    subsample used for model fitting, while ``train_full`` keeps the full train
    split available for protocol-level analysis such as ambiguity thresholds.

    The optional ``train_section`` further restricts the fitted training split
    to one of ``train_full``, ``train_S_amb``, or ``train_S_easy`` while
    leaving ``train_full`` unchanged.
    """
    dataset_url = str(getattr(args, "dataset_url", DEFAULT_CHAOSNLI_URL))
    data_root = Path(getattr(args, "data_root", "data/chaosnli"))
    source_subsets = parse_source_subsets(getattr(args, "source_subsets", ["snli", "mnli"]))
    train_section = _normalize_train_section(getattr(args, "train_section", "train_full"))

    split_seed = int(getattr(args, "split_seed", 13))
    train_frac = float(getattr(args, "train_frac", 0.80))
    val_frac = float(getattr(args, "val_frac", 0.10))
    eval_apply_keep_filter = bool(getattr(args, "eval_apply_keep_filter", False))

    embedding_model = str(getattr(args, "embedding_model", "roberta-base"))
    embedding_max_length = int(getattr(args, "embedding_max_length", 128))
    embedding_storage_dtype = str(getattr(args, "embedding_storage_dtype", "float16"))

    embedding_dir = resolve_embedding_dir(
        getattr(args, "emb_dir", "out/chaosnli_emb"),
        source_subsets=source_subsets,
        split_seed=split_seed,
        train_frac=train_frac,
        val_frac=val_frac,
        embedding_model=embedding_model,
        embedding_max_length=embedding_max_length,
        embedding_storage_dtype=embedding_storage_dtype,
    )

    derivation_config = VoteDerivationConfig(
        pi_eps=float(getattr(args, "pi_eps", 1e-6)),
        tie_tol=float(getattr(args, "tie_tol", 0.0)),
        eps_cap=float(getattr(args, "eps_cap", 0.05)),
    )

    raw_items = load_raw_items(
        dataset_url=dataset_url,
        data_root=data_root,
        source_subsets=source_subsets,
    )

    raw_splits = split_raw_items(
        raw_items,
        split_seed=split_seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    split_to_text_records = build_split_to_text_records(raw_splits)
    ensure_transformer_embeddings(
        split_to_items=split_to_text_records,
        emb_dir=embedding_dir,
        id_getter=lambda row: str(row["uid"]),
        text_getter=lambda row: str(row["text"]),
        model_name=embedding_model,
        batch_size=int(getattr(args, "embedding_batch_size", 64)),
        max_length=embedding_max_length,
        storage_dtype=embedding_storage_dtype,
        log_prefix="chaosnli",
    )

    train_full_items, train_full_stats = build_items_for_split(
        list(raw_splits["train"]),
        config=derivation_config,
        apply_keep_filter=True,
    )
    validation_items, validation_stats = build_items_for_split(
        list(raw_splits["validation"]),
        config=derivation_config,
        apply_keep_filter=eval_apply_keep_filter,
    )
    test_items, test_stats = build_items_for_split(
        list(raw_splits["test"]),
        config=derivation_config,
        apply_keep_filter=eval_apply_keep_filter,
    )

    train_slice_stats = compute_slice_stats_for_split(
        train_full_items,
        n_classes=len(LABELS),
    )
    train_section_thresholds = compute_slice_thresholds(train_slice_stats)
    train_masks = slice_masks_from_stats(train_slice_stats, train_section_thresholds)

    train_s_amb_items = [
        item for item, keep in zip(train_full_items, train_masks["S_amb"]) if bool(keep)
    ]
    train_s_easy_items = [
        item for item, keep in zip(train_full_items, train_masks["S_easy"]) if bool(keep)
    ]

    section_to_items = {
        "train_full": list(train_full_items),
        "train_S_amb": train_s_amb_items,
        "train_S_easy": train_s_easy_items,
    }
    train_section_items = list(section_to_items[train_section])
    train_section_sizes = {name: int(len(rows)) for name, rows in section_to_items.items()}

    _, train_embs, train_id_to_row = load_embedding_cache("train", embedding_dir)
    _, validation_embs, validation_id_to_row = load_embedding_cache("validation", embedding_dir)
    _, test_embs, test_id_to_row = load_embedding_cache("test", embedding_dir)

    train_full_samples, missing_train_full = items_to_samples(
        train_full_items,
        id_to_row=train_id_to_row,
        embs_arr=train_embs,
    )
    train_section_full_samples, missing_train = items_to_samples(
        train_section_items,
        id_to_row=train_id_to_row,
        embs_arr=train_embs,
    )
    validation_samples, missing_validation = items_to_samples(
        validation_items,
        id_to_row=validation_id_to_row,
        embs_arr=validation_embs,
    )
    test_samples, missing_test = items_to_samples(
        test_items,
        id_to_row=test_id_to_row,
        embs_arr=test_embs,
    )

    train_section_size_before_subsample = len(train_section_full_samples)
    full_train_size_before_subsample = len(train_full_samples)
    train_samples = list(train_section_full_samples)

    requested_train_size = int(getattr(args, "max_train_samples", 0))
    train_subset_seed = int(getattr(args, "train_subset_seed", 42))
    if requested_train_size > 0 and len(train_samples) > requested_train_size:
        train_samples = stratified_subset_by_label(
            train_samples,
            subset_size=requested_train_size,
            seed=train_subset_seed,
            label_attr="y",
        )

    effective_train_size = len(train_samples)
    feature_dim = int(train_embs.shape[1])
    n_classes = len(LABELS)

    print(
        f"[chaosnli] subsets={','.join(source_subsets)} split_seed={split_seed} "
        f"train_frac={train_frac:.4f} val_frac={val_frac:.4f} "
        f"train_section={train_section} "
        f"train={effective_train_size}(section_full={train_section_size_before_subsample},full={full_train_size_before_subsample},miss={missing_train}) "
        f"val_full={len(validation_samples)}(miss={missing_validation}) "
        f"test_full={len(test_samples)}(miss={missing_test}) "
        f"eval_keep_filter={int(eval_apply_keep_filter)} emb_dir={embedding_dir}",
        flush=True,
    )

    return {
        "train": train_samples,
        "train_full": list(train_full_samples),
        "val_full": validation_samples,
        "test_full": test_samples,
        "C": int(n_classes),
        "D": int(feature_dim),
        "label_names": list(LABELS),
        "source_subsets": list(source_subsets),
        "split_seed": split_seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "emb_dir": str(embedding_dir),
        "train_section": train_section,
        "train_section_sizes": dict(train_section_sizes),
        "slice_thresholds": dict(train_section_thresholds),
        "train_section_thresholds": dict(train_section_thresholds),
        "raw_counts": {
            "train": len(raw_splits["train"]),
            "validation": len(raw_splits["validation"]),
            "test": len(raw_splits["test"]),
        },
        "kept_counts": {
            "train": int(train_section_size_before_subsample),
            "validation": int(validation_stats["kept"]),
            "test": int(test_stats["kept"]),
        },
        "kept_counts_full": {
            "train": int(train_full_stats["kept"]),
            "validation": int(validation_stats["kept"]),
            "test": int(test_stats["kept"]),
        },
        "requested_train_size": requested_train_size,
        "effective_train_size": effective_train_size,
        "full_train_size_before_subsample": full_train_size_before_subsample,
        "train_section_size_before_subsample": train_section_size_before_subsample,
        "train_subset_seed": train_subset_seed,
        "missing_embedding_counts": {
            "train": int(missing_train),
            "train_full": int(missing_train_full),
            "validation": int(missing_validation),
            "test": int(missing_test),
        },
    }



def _add_embedding_args(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--embedding-model", default="roberta-base")
    group.add_argument("--embedding-batch-size", type=int, default=64)
    group.add_argument("--embedding-max-length", type=int, default=128)
    group.add_argument(
        "--embedding-storage-dtype",
        default="float16",
        choices=["float16", "float32", "float64"],
    )



def add_chaosnli_data_args(parser: argparse.ArgumentParser) -> None:
    """Register ChaosNLI-specific CLI arguments on ``parser``."""
    group = parser.add_argument_group("ChaosNLI data")
    group.add_argument("--dataset-url", default=DEFAULT_CHAOSNLI_URL)
    group.add_argument("--data-root", default="data/chaosnli")
    group.add_argument(
        "--emb-dir",
        default="out/chaosnli_emb",
        help=(
            "Embedding cache directory. When left at the default base name, the "
            "loader automatically scopes it by subset list, split protocol, and "
            "embedding configuration."
        ),
    )
    group.add_argument(
        "--source-subsets",
        nargs="+",
        default=["snli", "mnli"],
        help="Subset list drawn from {snli, mnli}. Both comma-separated and space-separated forms are accepted.",
    )
    group.add_argument(
        "--train-section",
        default="train_full",
        choices=list(TRAIN_SECTIONS),
        help=(
            "Training subset used for model fitting. train_full keeps the whole train split; "
            "train_S_amb and train_S_easy apply the fixed ambiguity slices derived from the full train split."
        ),
    )
    group.add_argument("--split-seed", type=int, default=13)
    group.add_argument("--train-frac", type=float, default=0.80)
    group.add_argument("--val-frac", type=float, default=0.10)
    _add_embedding_args(group)
    group.add_argument("--pi-eps", type=float, default=1e-6)
    group.add_argument("--tie-tol", type=float, default=0.0)
    group.add_argument("--eps-cap", type=float, default=0.05)
    group.add_argument("--eval-apply-keep-filter", action="store_true")
    group.add_argument("--max-train-samples", type=int, default=0)
    group.add_argument("--train-subset-seed", type=int, default=42)
