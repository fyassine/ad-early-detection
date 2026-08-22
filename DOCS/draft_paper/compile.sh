#!/usr/bin/env bash
# Compile DOCS/draft_paper/paper.tex into a PDF using tectonic (self-contained
# TeX engine, no system TeX Live install required beyond what's already
# vendored in the project .venv).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if command -v tectonic >/dev/null 2>&1; then
    TECTONIC=tectonic
elif [ -x "$REPO_ROOT/.venv/bin/tectonic" ]; then
    TECTONIC="$REPO_ROOT/.venv/bin/tectonic"
else
    echo "error: tectonic not found on PATH or at $REPO_ROOT/.venv/bin/tectonic" >&2
    echo "install with: pip install tectonic" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
"$TECTONIC" paper.tex "$@"
echo "Built: $SCRIPT_DIR/paper.pdf"
