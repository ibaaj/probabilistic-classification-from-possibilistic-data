#!/usr/bin/env bash
set -euo pipefail

chaosnli_repo_root() {
  local this_dir
  this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${this_dir}/.." && pwd
}

chaosnli_log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

chaosnli_die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

chaosnli_run_python() {
  local script="$1"
  shift
  local repo_root="${REPO_ROOT:-$(chaosnli_repo_root)}"
  local python_bin="${PYTHON:-python3}"
  "${python_bin}" "${repo_root}/${script}" "$@"
}
