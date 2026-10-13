#!/bin/bash
# Run the unit / smoke tests (CPU is sufficient).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" tests/test_roi_refiner_mapping.py
"$PYTHON_BIN" tests/test_fetalheart_optimization.py
"$PYTHON_BIN" tests/test_baselines_smoke.py
echo "All tests passed."
