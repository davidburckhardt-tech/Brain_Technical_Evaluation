#!/usr/bin/env bash
# Phase 0 setup: create a Python virtual environment and install the benchmark
# tooling (GuideLLM). Also verifies that the Ollama CLI is installed, since it
# is the inference runtime used in this phase.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "==> Checking for the Ollama CLI..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: Ollama is not installed. Install it from https://ollama.com before continuing." >&2
  exit 1
fi
echo "Found: $(ollama --version 2>&1 | head -1)"

echo "==> Creating Python virtual environment at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"

echo "==> Installing GuideLLM into the virtual environment..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install "guidellm[recommended]"

echo "==> Installing psutil (CPU/memory monitoring, used when no GPU is detected)..."
"${VENV_DIR}/bin/pip" install psutil

echo "==> Setup complete."
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
