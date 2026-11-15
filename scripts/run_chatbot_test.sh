#!/usr/bin/env bash
# Test 2: concurrent chatbot workload. Fires one request per entry in
# datasets/prompts.jsonl at the same time against Ollama directly (for real,
# instrumented metrics), while Open WebUI (scripts/start_chatbot_ui.sh) can
# be running alongside for the visual/recording layer.
#
# Usage: ./scripts/run_chatbot_test.sh [--model <short-name>]
#   --model (see scripts/lib_model.sh) sets MODEL_NAME and, same as
#   run_benchmark.sh, makes sure vLLM is actually serving that model first
#   (downloading it if not already cached).

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

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
# Optional: point at a bigger prompts file (e.g. datasets/prompts_heavy.jsonl)
# to simulate more concurrent users than the default 4.
PROMPTS_FILE="${PROMPTS_FILE:-}"
# Optional: turns per simulated conversation (default 1 = single-shot,
# original behavior) and the model's context window for utilization %.
CHATBOT_TURNS="${CHATBOT_TURNS:-1}"
CHATBOT_MAX_CONTEXT_TOKENS="${CHATBOT_MAX_CONTEXT_TOKENS:-}"

if [[ -n "${RESOLVED_MODEL_REPO}" ]]; then
  MODEL_NAME="${RESOLVED_MODEL_REPO}"
  echo "==> --model given: ensuring vLLM is serving ${MODEL_NAME} (downloads it if not cached)..."
  # --model, not an exported VLLM_MODEL env var - see run_benchmark.sh for why
  # (start_inference_vllm.sh's own `source .env` would silently clobber an
  # inherited env var back to .env's model).
  "${PROJECT_ROOT}/scripts/start_inference_vllm.sh" --model "${RESOLVED_MODEL_SHORT_NAME}"
fi

if [[ ! -f "${VENV_DIR}/bin/python3" ]]; then
  echo "Error: virtual environment not found. Run scripts/setup.sh first." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_MODEL_NAME="${MODEL_NAME//[:\/]/_}"
RUN_DIR="${PROJECT_ROOT}/results/${SAFE_MODEL_NAME}"
mkdir -p "${RUN_DIR}"
OUTPUT_FILE="${RUN_DIR}/${TIMESTAMP}_chatbot.json"
MONITOR_OUTPUT="${RUN_DIR}/${TIMESTAMP}_chatbot_resources.csv"

# Resource monitoring: auto-detects GPU (nvidia-smi) vs CPU (psutil), runs in
# the background for the duration of the test.
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_resources.py" \
  --output "${MONITOR_OUTPUT}" --interval 1 &
MONITOR_PID=$!

# vLLM-specific queue/KV-cache metrics - harmless no-op (blank rows) if
# OLLAMA_HOST_URL isn't actually a vLLM server.
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_vllm_metrics.py" \
  --vllm-url "${OLLAMA_HOST_URL}" --output "${RUN_DIR}/${TIMESTAMP}_chatbot_vllm_metrics.csv" --interval 1 &
VLLM_METRICS_PID=$!

trap 'kill -TERM "${MONITOR_PID}" "${VLLM_METRICS_PID}" 2>/dev/null; wait "${MONITOR_PID}" "${VLLM_METRICS_PID}" 2>/dev/null' EXIT

"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/benchmarks/chatbot/simulate_users.py" \
  --target "${OLLAMA_HOST_URL}" \
  --model "${MODEL_NAME}" \
  --output "${OUTPUT_FILE}" \
  --ollama-num-parallel "${OLLAMA_NUM_PARALLEL:-}" \
  --turns "${CHATBOT_TURNS}" \
  ${PROMPTS_FILE:+--prompts-file "${PROMPTS_FILE}"} \
  ${CHATBOT_MAX_CONTEXT_TOKENS:+--max-context-tokens "${CHATBOT_MAX_CONTEXT_TOKENS}"}
