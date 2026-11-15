#!/usr/bin/env bash
# Test 3 (RAG) - step 2: runs N questions against the retrieval + generation
# pipeline with a given concurrency, and saves a timestamped JSON summary.
#
# Usage: ./scripts/run_rag_test.sh <num_queries> <concurrency> [--model <short-name>]
#   --model (see scripts/lib_model.sh) sets MODEL_NAME (the generation
#   model only - EMBEDDING_MODEL/EMBEDDING_HOST are independent, see below)
#   and makes sure vLLM is actually serving it first (downloading it if not
#   already cached).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/lib_model.sh"
parse_model_flag "$@"
set -- ${PARSED_ARGS[@]+"${PARSED_ARGS[@]}"}

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

NUM_QUERIES="${1:?Usage: run_rag_test.sh <num_queries> <concurrency> [--model <short-name>]}"
CONCURRENCY="${2:?Usage: run_rag_test.sh <num_queries> <concurrency> [--model <short-name>]}"

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
if [[ -n "${RESOLVED_MODEL_REPO}" ]]; then
  MODEL_NAME="${RESOLVED_MODEL_REPO}"
  echo "==> --model given: ensuring vLLM is serving ${MODEL_NAME} (downloads it if not cached)..."
  # --model, not an exported VLLM_MODEL env var - see run_benchmark.sh for why
  # (start_inference_vllm.sh's own `source .env` would silently clobber an
  # inherited env var back to .env's model).
  "${PROJECT_ROOT}/scripts/start_inference_vllm.sh" --model "${RESOLVED_MODEL_SHORT_NAME}"
fi
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
SAFE_MODEL_NAME="${MODEL_NAME//[:\/]/_}"
RUN_DIR="${PROJECT_ROOT}/results/${SAFE_MODEL_NAME}"
mkdir -p "${RUN_DIR}"
OUTPUT_FILE="${RUN_DIR}/${TIMESTAMP}_rag_q${NUM_QUERIES}_c${CONCURRENCY}.json"
MONITOR_OUTPUT="${RUN_DIR}/${TIMESTAMP}_rag_q${NUM_QUERIES}_c${CONCURRENCY}_resources.csv"

# Resource monitoring: auto-detects GPU (nvidia-smi) vs CPU (psutil), runs in
# the background for the duration of the test.
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_resources.py" \
  --output "${MONITOR_OUTPUT}" --interval 1 &
MONITOR_PID=$!

# vLLM-specific queue/KV-cache metrics - harmless no-op (blank rows) if
# OLLAMA_HOST_URL isn't actually a vLLM server.
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_vllm_metrics.py" \
  --vllm-url "${OLLAMA_HOST_URL}" \
  --output "${RUN_DIR}/${TIMESTAMP}_rag_q${NUM_QUERIES}_c${CONCURRENCY}_vllm_metrics.csv" --interval 1 &
VLLM_METRICS_PID=$!

trap 'kill -TERM "${MONITOR_PID}" "${VLLM_METRICS_PID}" 2>/dev/null; wait "${MONITOR_PID}" "${VLLM_METRICS_PID}" 2>/dev/null' EXIT

"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/benchmarks/rag/query.py" \
  --num-queries "${NUM_QUERIES}" \
  --concurrency "${CONCURRENCY}" \
  --ollama-host "${OLLAMA_HOST_URL}" \
  --embedding-host "${EMBEDDING_HOST}" \
  --model "${MODEL_NAME}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --collection "${RAG_COLLECTION}" \
  --output "${OUTPUT_FILE}"
