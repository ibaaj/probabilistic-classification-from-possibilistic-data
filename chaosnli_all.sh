#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOURCE_SUBSETS="${SOURCE_SUBSETS:-snli,mnli}"
export OUT_BASE="${OUT_BASE:-out/real_nlp/chaosnli_all}"
exec "${SCRIPT_DIR}/chaosnli.sh" "$@"
