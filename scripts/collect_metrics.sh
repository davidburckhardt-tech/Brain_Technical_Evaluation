#!/usr/bin/env bash
# Extracts the key metrics (TTFT, ITL, throughput, latency) from the most
# recent GuideLLM run and appends them to results/summary.csv for cross-model
# comparison.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"

LATEST_RUN="$(ls -td "${RESULTS_DIR}"/*/*/ 2>/dev/null | head -1)"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "Error: no benchmark run found under ${RESULTS_DIR}." >&2
  exit 1
fi

BENCHMARK_JSON="${LATEST_RUN}benchmarks.json"
if [[ ! -f "${BENCHMARK_JSON}" ]]; then
  echo "Error: ${BENCHMARK_JSON} not found. Did the benchmark run succeed?" >&2
  exit 1
fi

echo "==> Summarizing ${BENCHMARK_JSON}"

python3 "${PROJECT_ROOT}/scripts/summarize_benchmark.py" \
  --input "${BENCHMARK_JSON}" \
  --summary-csv "${SUMMARY_FILE}" \
  --run-dir "${LATEST_RUN}"

echo "==> Summary appended to ${SUMMARY_FILE}"
