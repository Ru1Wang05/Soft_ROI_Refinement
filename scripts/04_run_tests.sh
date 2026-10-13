#!/bin/bash
# Run the unit / smoke checks (CPU is sufficient).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" tests/run_checks.py
