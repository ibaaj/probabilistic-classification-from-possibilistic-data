from __future__ import annotations

"""Read the raw ChaosNLI release files."""

import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import DEFAULT_CHAOSNLI_URL, LABELS, SHORT_LABEL_TO_LONG, SUBSET_TO_FILENAME
from .schema import ChaosNLIRawItem


def normalize_nli_label(value: object) -> str:
    """Normalize short and long label names to one canonical form."""
    key = str(value).strip().lower()
    if key not in SHORT_LABEL_TO_LONG:
        raise ValueError(f"Unknown ChaosNLI label: {value!r}")
    return SHORT_LABEL_TO_LONG[key]


def parse_source_subsets(value: str | Sequence[str]) -> list[str]:
    """Parse and validate the requested subset list.

    Accepted forms include both comma-separated and space-separated values.
    """
    if isinstance(value, str):
        raw_tokens = value.replace(",", " ").split()
    else:
        raw_tokens = []
        for item in value:
            raw_tokens.extend(str(item).replace(",", " ").split())

    subsets: list[str] = []
    for token in raw_tokens:
        subset = token.strip().lower()
        if not subset:
            continue
        if subset not in SUBSET_TO_FILENAME:
            allowed = ", ".join(sorted(SUBSET_TO_FILENAME))
            raise ValueError(f"Unsupported ChaosNLI subset {subset!r}. Allowed subsets: {allowed}.")
        if subset not in subsets:
            subsets.append(subset)

    if not subsets:
        raise ValueError("At least one ChaosNLI subset must be requested.")
    return subsets


def _atomic_download(url: str, destination: Path) -> None:
    """Download to a temporary file and move into place atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_url = str(url)
    if "dropbox.com" in normalized_url and "dl=" not in normalized_url:
        normalized_url += "&dl=1" if "?" in normalized_url else "?dl=1"

    request = urllib.request.Request(normalized_url, headers={"User-Agent": "Mozilla/5.0"})
    with tempfile.NamedTemporaryFile(dir=str(destination.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _find_jsonl_files(search_roots: Sequence[Path], expected_filenames: Sequence[str]) -> dict[str, Path]:
    """Find expected ChaosNLI JSONL files, preferring earlier roots.

    This is intentionally ordered: callers should pass the freshly extracted
    archive tree first, then broader fallback locations such as the dataset root.
    """
    wanted = {str(name) for name in expected_filenames}
    found: dict[str, Path] = {}

    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        for path in sorted(root_path.rglob("*.jsonl")):
            if path.name in wanted and path.name not in found:
                found[path.name] = path

        if len(found) == len(wanted):
            break

    missing = sorted(wanted - set(found))
    if missing:
        searched = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(
            f"Could not find ChaosNLI files after extraction: {missing}. "
            f"Searched roots in order: {searched}"
        )
    return found


def download_and_extract(
    url: str = DEFAULT_CHAOSNLI_URL,
    data_root: str | Path = "data/chaosnli",
    subsets: Sequence[str] = ("snli", "mnli"),
) -> dict[str, Path]:
    """Ensure the archive is available locally and return the JSONL paths."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "chaosNLI_v1.0.zip"
    if not zip_path.exists():
        _atomic_download(url, zip_path)

    extract_dir = root / "chaosNLI_v1.0"
    required = {subset: extract_dir / SUBSET_TO_FILENAME[subset] for subset in subsets}
    if all(path.exists() for path in required.values()):
        return required

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root)

    if not all(path.exists() for path in required.values()):
        found = _find_jsonl_files(
            [extract_dir, root],
            [SUBSET_TO_FILENAME[s] for s in subsets],
        )
        for subset, target in required.items():
            source = found[SUBSET_TO_FILENAME[subset]]
            if source.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    return required


def _entropy_from_votes(votes: np.ndarray) -> float:
    votes = np.asarray(votes, dtype=np.float64)
    total = float(votes.sum())
    if total <= 0.0:
        return 0.0
    probabilities = votes / total
    mask = probabilities > 0.0
    return float(-np.sum(probabilities[mask] * np.log(probabilities[mask])))


def _majority_label_from_votes(votes: np.ndarray) -> str:
    """Derive a deterministic majority label from vote counts.

    Ties are broken by canonical label order:
    entailment, neutral, contradiction.
    """
    counts = np.asarray(votes, dtype=np.int64)
    if counts.shape != (len(LABELS),):
        raise ValueError(f"Expected votes with shape {(len(LABELS),)}, got {counts.shape}")
    return LABELS[int(np.argmax(counts))]


def _votes_from_row(row: dict) -> np.ndarray:
    label_count = row.get("label_count")
    if isinstance(label_count, list) and len(label_count) == len(LABELS):
        votes = np.asarray(label_count, dtype=np.int64)
    else:
        counter = row.get("label_counter", {})
        if not isinstance(counter, dict):
            raise ValueError("ChaosNLI row is missing both `label_count` and `label_counter`.")
        votes = np.asarray([
            int(counter.get("e", 0)),
            int(counter.get("n", 0)),
            int(counter.get("c", 0)),
        ], dtype=np.int64)

    if votes.shape != (len(LABELS),):
        raise ValueError(f"Expected votes with shape {(len(LABELS),)}, got {votes.shape}")
    if np.any(votes < 0):
        raise ValueError(f"Negative vote counts are not allowed: {votes.tolist()}")
    return votes.astype(np.int16, copy=False)


def read_chaosnli_jsonl(
    path: str | Path,
    subset: str,
) -> list[ChaosNLIRawItem]:
    """Read one ChaosNLI JSONL file into validated raw items."""
    file_path = Path(path)
    items: list[ChaosNLIRawItem] = []
    seen_uids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            example = row.get("example") if isinstance(row.get("example"), dict) else {}

            premise = str(example.get("premise", "")).strip()
            hypothesis = str(example.get("hypothesis", "")).strip()
            if not premise or not hypothesis:
                continue

            votes = _votes_from_row(row)
            n_raters = int(votes.sum())
            derived_majority_label = _majority_label_from_votes(votes)
            top_vote_count = int(np.max(votes)) if votes.size else 0
            has_unique_majority = int(np.sum(votes == top_vote_count)) == 1

            original_uid = str(row.get("uid", example.get("uid", ""))).strip()
            if not original_uid:
                raise ValueError(f"Missing uid at {file_path}:{line_number}")

            uid = f"{subset}::{original_uid}"
            if uid in seen_uids:
                raise ValueError(f"Duplicate ChaosNLI uid detected: {uid!r}")
            seen_uids.add(uid)

            raw_majority_label = row.get("majority_label", None)
            if raw_majority_label in (None, ""):
                majority_label = derived_majority_label
            else:
                majority_label = normalize_nli_label(raw_majority_label)
                if has_unique_majority and majority_label != derived_majority_label:
                    raise ValueError(
                        f"majority_label mismatch with unique top vote at {file_path}:{line_number}: "
                        f"uid={uid!r}, raw={majority_label!r}, "
                        f"derived={derived_majority_label!r}, votes={votes.tolist()}"
                    )

            raw_old_label = row.get("old_label", None)
            if raw_old_label in (None, ""):
                old_label = majority_label
            else:
                old_label = normalize_nli_label(raw_old_label)
            
            entropy = _entropy_from_votes(votes)

            items.append(
                ChaosNLIRawItem(
                    uid=uid,
                    original_uid=original_uid,
                    subset=str(subset),
                    premise=premise,
                    hypothesis=hypothesis,
                    old_label=old_label,
                    majority_label=majority_label,
                    votes=votes,
                    n_raters=n_raters,
                    entropy=float(entropy),
                )
            )

    return sorted(items, key=lambda item: item.uid)