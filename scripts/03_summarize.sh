#!/bin/bash
# Aggregate all frozen results into the paper tables:
#   seed_extension_results/summary_5seeds/   five-seed main table + paired t-tests
#   final_test_results/summary/              frozen three-seed final test table
#   baselines_results/summary_5seeds/        baseline five-seed table
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" summarize_fetalheart_final_test.py
"$PYTHON_BIN" summarize_aish_5seed_statistics.py
"$PYTHON_BIN" summarize_baselines_5seeds.py
echo "Summaries written."
