#!/usr/bin/env bash
# Test 3 (RAG) - step 2: runs N questions against the retrieval + generation
# pipeline with a given concurrency, and saves a timestamped JSON summary.
#
# Usage: ./scripts/run_rag_test.sh <num_queries> <concurrency>

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

NUM_QUERIES="${1:?Usage: run_rag_test.sh <num_queries> <concurrency>}"
CONCURRENCY="${2:?Usage: run_rag_test.sh <num_queries> <concurrency>}"

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
# EMBEDDING_HOST lets embeddings run against a different server than
# generation (e.g. Ollama serving nomic-embed-text while vLLM serves the
# generation model) - defaults to OLLAMA_HOST when unset.
EMBEDDING_HOST="${EMBEDDING_HOST:-${OLLAMA_HOST_URL}}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
RAG_COLLECTION="${RAG_COLLECTION:-k8s_docs}"

if [[ ! -f "${VENV_DIR}/bin/python3" ]]; then
  echo "Error: virtual environment not found. Run scripts/setup.sh first." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${PROJECT_ROOT}/results/${TIMESTAMP}_rag_q${NUM_QUERIES}_c${CONCURRENCY}.json"
MONITOR_OUTPUT="${PROJECT_ROOT}/results/${TIMESTAMP}_rag_q${NUM_QUERIES}_c${CONCURRENCY}_resources.csv"

# Resource monitoring: auto-detects GPU (nvidia-smi) vs CPU (psutil), runs in
# the background for the duration of the test.
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_resources.py" \
  --output "${MONITOR_OUTPUT}" --interval 1 &
MONITOR_PID=$!
trap 'kill -TERM "${MONITOR_PID}" 2>/dev/null; wait "${MONITOR_PID}" 2>/dev/null' EXIT

"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/benchmarks/rag/query.py" \
  --num-queries "${NUM_QUERIES}" \
  --concurrency "${CONCURRENCY}" \
  --ollama-host "${OLLAMA_HOST_URL}" \
  --embedding-host "${EMBEDDING_HOST}" \
  --model "${MODEL_NAME}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --collection "${RAG_COLLECTION}" \
  --output "${OUTPUT_FILE}"
