#!/usr/bin/env bash
# Test 2: concurrent chatbot workload. Fires one request per entry in
# datasets/prompts.jsonl at the same time against Ollama directly (for real,
# instrumented metrics), while Open WebUI (scripts/start_chatbot_ui.sh) can
# be running alongside for the visual/recording layer.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
# Optional: point at a bigger prompts file (e.g. datasets/prompts_heavy.jsonl)
# to simulate more concurrent users than the default 4.
PROMPTS_FILE="${PROMPTS_FILE:-}"

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
trap 'kill -TERM "${MONITOR_PID}" 2>/dev/null; wait "${MONITOR_PID}" 2>/dev/null' EXIT

"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/benchmarks/chatbot/simulate_users.py" \
  --target "${OLLAMA_HOST_URL}" \
  --model "${MODEL_NAME}" \
  --output "${OUTPUT_FILE}" \
  --ollama-num-parallel "${OLLAMA_NUM_PARALLEL:-}" \
  ${PROMPTS_FILE:+--prompts-file "${PROMPTS_FILE}"}
