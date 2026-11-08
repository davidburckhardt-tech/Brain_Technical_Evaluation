#!/usr/bin/env bash
# Starts the Ollama server if it is not already running, and warms up the
# target model by loading it into memory with a single generate call.
#
# If OLLAMA_NUM_PARALLEL is set in .env, this ALWAYS restarts Ollama with
# that value exported, even if an instance (including the Ollama.app menu
# bar app) is already running - env vars only take effect at process start,
# so there is no way to apply this to an already-running server.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_FILE="${RESULTS_DIR}/ollama_server.log"
PID_FILE="${RESULTS_DIR}/ollama_server.pid"

mkdir -p "${RESULTS_DIR}"

if [[ -n "${OLLAMA_NUM_PARALLEL:-}" ]]; then
  echo "==> OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} requested. Restarting Ollama to apply it..."
  pkill -f "ollama serve" 2>/dev/null || true
  pkill -f "Ollama.app" 2>/dev/null || true
  sleep 2

  echo "Starting Ollama with OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} (logs: ${LOG_FILE})..."
  OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" nohup ollama serve >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"

  echo "Waiting for Ollama to become ready..."
  for i in $(seq 1 30); do
    if curl -sf "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
      echo "Ollama is ready (restarted with OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL})."
      break
    fi
    sleep 1
  done
elif curl -sf "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
  echo "==> Ollama is already running at ${OLLAMA_HOST_URL} (OLLAMA_NUM_PARALLEL not set, reusing it as-is)."
else
  echo "==> Ollama is not running. Starting it in the background (logs: ${LOG_FILE})..."
  nohup ollama serve >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"

  echo "Waiting for Ollama to become ready..."
  for i in $(seq 1 30); do
    if curl -sf "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
      echo "Ollama is ready."
      break
    fi
    sleep 1
  done
fi

echo "==> Warming up model: ${MODEL_NAME}"
curl -sf "${OLLAMA_HOST_URL}/api/generate" \
  -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"Hello\", \"stream\": false}" >/dev/null

echo "==> Model is loaded and ready to serve requests."
echo "OpenAI-compatible endpoint: ${OLLAMA_HOST_URL}/v1"
