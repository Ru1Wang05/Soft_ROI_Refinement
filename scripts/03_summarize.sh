#!/bin/bash
# Aggregate all frozen results into the paper tables:
#   seed_extension_results/summary_5seeds/   five-seed main table + paired t-tests
#   baselines_results/summary_5seeds/        baseline five-seed table
#   final_test_results/summary/              frozen three-seed final test table
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" summarize.py
echo "Summaries written."
