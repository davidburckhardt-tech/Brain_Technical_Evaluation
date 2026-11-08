#!/usr/bin/env bash
# Starts Open WebUI (via Docker Compose), connected to the Ollama server
# running on the host. This is the visual layer for Test 2 (concurrent
# chatbot) - the actual load/metrics come from benchmarks/chatbot/simulate_users.py,
# run separately against Ollama directly.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://localhost:11434}"

echo "==> Checking that Ollama is reachable at ${OLLAMA_HOST_URL}..."
if ! curl -sf "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
  echo "Error: Ollama is not running. Run scripts/start_inference_ollama.sh first." >&2
  exit 1
fi

echo "==> Starting Open WebUI via Docker Compose..."
(cd "${PROJECT_ROOT}" && docker compose up -d open-webui)

echo "==> Waiting for Open WebUI to become ready..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:3000/health" >/dev/null 2>&1; then
    echo "Open WebUI is ready."
    break
  fi
  sleep 2
done

echo "==> Open WebUI is available at: http://localhost:3000"
echo "    (WEBUI_AUTH is disabled for this demo - no login required)"
