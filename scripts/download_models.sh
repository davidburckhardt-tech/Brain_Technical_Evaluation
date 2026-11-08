#!/usr/bin/env bash
# Pulls the model needed for the current phase via the Ollama CLI.
# Reads MODEL_NAME from .env if present, otherwise falls back to the
# Phase 0 smoke-test model.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

MODEL_NAME="${MODEL_NAME:-llama3:8b}"

echo "==> Pulling model: ${MODEL_NAME}"
ollama pull "${MODEL_NAME}"

echo "==> Model ready. Verify with: ollama list"
