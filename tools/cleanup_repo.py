#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".nox",
    ".tox",
    ".venv",
    "build",
}

DEFAULT_FILE_NAMES = {
    ".DS_Store",
}

DEFAULT_SUFFIXES = {
    ".so",
    ".pyd",
    ".dylib",
    ".pyc",
    ".pyo",
}

# Exact relative directories to remove.
DEFAULT_RELATIVE_DIRS = {
    "data/chaosnli",
}

# Relative-directory prefixes to remove.
DEFAULT_RELATIVE_DIR_PREFIXES = (
    "out/chaosnli_emb",
)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _matches_relative_dir_prefix(rel: str) -> bool:
    for prefix in DEFAULT_RELATIVE_DIR_PREFIXES:
        if rel == prefix or rel.startswith(prefix + "_"):
            return True
    return False


def iter_matches(root: Path, *, remove_out: bool) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()

    for path in root.rglob("*"):
        if path.is_symlink():
            continue

        rel = path.relative_to(root).as_posix()

        if path.is_dir():
            if path.name in DEFAULT_DIR_NAMES:
                if path not in seen:
                    matches.append(path)
                    seen.add(path)
                continue

            if rel in DEFAULT_RELATIVE_DIRS or _matches_relative_dir_prefix(rel):
                if path not in seen:
                    matches.append(path)
                    seen.add(path)
                continue

            if remove_out and rel == "out":
                if path not in seen:
                    matches.append(path)
                    seen.add(path)
                continue

        elif path.is_file():
            if path.name in DEFAULT_FILE_NAMES or path.suffix in DEFAULT_SUFFIXES:
                parent_is_selected = any(is_relative_to(path.parent, selected) for selected in seen if selected.is_dir())
                if not parent_is_selected and path not in seen:
                    matches.append(path)
                    seen.add(path)

    matches.sort(key=lambda p: (len(p.parts), str(p)), reverse=True)
    return matches


def bytes_for_path(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for sub in path.rglob("*"):
            if sub.is_file() and not sub.is_symlink():
                try:
                    total += sub.stat().st_size
                except OSError:
                    pass
    return total


def human_bytes(n: int) -> str:
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove regenerable/cache/build files from this repository."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--include-out",
        action="store_true",
        help="Also remove the whole ./out directory.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints what would be removed.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Invalid root directory: {root}")

    matches = iter_matches(root, remove_out=bool(args.include_out))
    total_bytes = sum(bytes_for_path(path) for path in matches)

    print(f"root: {root}")
    print(f"matches: {len(matches)}")
    print(f"estimated space: {human_bytes(total_bytes)}")

    if not matches:
        return 0

    print()
    for path in matches:
        kind = "dir " if path.is_dir() else "file"
        print(f"{kind}  {path.relative_to(root)}")

    if not args.yes:
        print()
        print("Dry run only. Re-run with --yes to delete.")
        return 0

    for path in matches:
        remove_path(path)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())