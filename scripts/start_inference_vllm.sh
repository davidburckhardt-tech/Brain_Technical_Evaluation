#!/usr/bin/env bash
# vLLM equivalent of start_inference_ollama.sh, for the real Brain hardware
# (NVIDIA GPUs). Downloads the target model from Hugging Face, starts vLLM's
# OpenAI-compatible server, and warms it up with a real request.
#
# Validated end-to-end on a real NVIDIA test server (2x RTX 4070 SUPER,
# driver 560.35.03 / CUDA 12.6, Ubuntu 24.04). Two fixes were needed to get
# there, both handled automatically below:
#   1. vLLM is pinned to 0.19.1. Versions 0.20.0+ ship a precompiled native
#      extension statically linked against CUDA 13.0, which fails with
#      "NVIDIA driver on your system is too old" / "libcudart.so.13: cannot
#      open shared object file" on any driver older than ~570.x (CUDA 13).
#      0.19.1 pulls in torch built against CUDA 12.8, which only needs
#      driver >=550.54 - compatible with this and most currently deployed
#      drivers. If the Brain hardware ships a newer driver, this pin can be
#      revisited.
#   2. CUDA_VISIBLE_DEVICES, if already set in the shell/system environment
#      using GPU UUIDs (e.g. "GPU-28a2decb-...", which `nvidia-smi -L` also
#      prints), crashes vLLM 0.19.1 with "ValueError: invalid literal for
#      int() ... 'GPU-28a2decb-...'" - it expects plain numeric indices.
#      This script normalizes it to numeric indices below.
#
# Required .env variables (see .env.example):
#   VLLM_MODEL                 Hugging Face repo id (e.g. meta-llama/Llama-3.1-8B-Instruct)
# Optional:
#   VLLM_HOST (default 0.0.0.0)       VLLM_PORT (default 8000)
#   VLLM_TENSOR_PARALLEL_SIZE (default 1)      - set to the number of GPUs for one model
#   VLLM_GPU_MEMORY_UTILIZATION (default 0.90) - fraction of VRAM vLLM is allowed to reserve
#   VLLM_MAX_MODEL_LEN (default: model's own config)
#   VLLM_MAX_NUM_SEQS (default: vLLM's own default, ~256) - caps concurrent sequences;
#       this is vLLM's rough equivalent of Ollama's OLLAMA_NUM_PARALLEL, except vLLM
#       batches them continuously instead of just admitting N in parallel
#   VLLM_DTYPE (e.g. bfloat16, float16, auto)
#   VLLM_QUANTIZATION (e.g. awq, gptq, fp8 - only if the checkpoint supports it)
#   VLLM_EXTRA_ARGS             Any additional flags, passed through as-is
#   HF_TOKEN                    Needed for gated models (e.g. Llama); export before running

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv-vllm"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

VLLM_MODEL="${VLLM_MODEL:?Set VLLM_MODEL in .env to a Hugging Face repo id, e.g. meta-llama/Llama-3.1-8B-Instruct}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_URL="http://localhost:${VLLM_PORT}"

# Normalize CUDA_VISIBLE_DEVICES from GPU UUIDs (which some systems set by
# default) to plain numeric indices - see fix #2 above.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *"GPU-"* ]]; then
  echo "==> CUDA_VISIBLE_DEVICES is set to GPU UUIDs; normalizing to numeric indices for vLLM compatibility."
  NUM_GPUS="$(nvidia-smi -L | wc -l)"
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_FILE="${RESULTS_DIR}/vllm_server.log"
PID_FILE="${RESULTS_DIR}/vllm_server.pid"
mkdir -p "${RESULTS_DIR}"

# --- 1. Environment: a DEDICATED venv, separate from .venv (GuideLLM's).
# vLLM pulls in CUDA-specific torch builds that can conflict with other
# packages; keeping it isolated avoids fighting GuideLLM's own dependencies.
#
# On a fresh Ubuntu box, `python3 -m venv` can fail with "ensurepip is not
# available" because python3-venv is a separate apt package. If that
# happens: `sudo apt install -y python3.$(python3 -c 'import sys;print(sys.version_info[1])')-venv`
# (confirmed needed on Ubuntu 24.04 / Python 3.12).
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating vLLM virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
  echo "==> Installing vLLM (this downloads a large CUDA-enabled build, can take several minutes)..."
  "${VENV_DIR}/bin/pip" install --upgrade pip
  "${VENV_DIR}/bin/pip" install "vllm==0.19.1" huggingface_hub[cli]
else
  echo "==> Reusing existing vLLM virtual environment at ${VENV_DIR}."
fi

# --- 2. Pre-download the model's weights explicitly, as its OWN timed step.
# vLLM would otherwise download on first `vllm serve` call, hiding download
# time inside "startup" time - worth keeping separate so we know, inside the
# 4-hour window, how much time weight downloads actually cost.
#
# Uses `hf` (huggingface_hub>=0.24's CLI, replacing the old `huggingface-cli`
# name) - confirmed on real hardware (huggingface_hub 2.4.0) that the old
# name no longer exists as a console script.
echo "==> Downloading ${VLLM_MODEL} from Hugging Face (skips files already cached)..."
time "${VENV_DIR}/bin/hf" download "${VLLM_MODEL}" \
  ${HF_TOKEN:+--token "${HF_TOKEN}"}

# --- 3. Start vLLM's OpenAI-compatible server, if not already serving this model.
if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
  CURRENT_MODEL="$(curl -sf "${VLLM_URL}/v1/models" | "${VENV_DIR}/bin/python3" -c \
    "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")"
  if [[ "${CURRENT_MODEL}" == "${VLLM_MODEL}" ]]; then
    echo "==> vLLM is already serving ${VLLM_MODEL} at ${VLLM_URL}."
  else
    echo "==> vLLM is running a different model (${CURRENT_MODEL:-unknown}). Restarting with ${VLLM_MODEL}..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
  fi
fi

if ! curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
  echo "==> Starting vLLM server (logs: ${LOG_FILE})..."
  VLLM_ARGS=(
    serve "${VLLM_MODEL}"
    --host "${VLLM_HOST}" --port "${VLLM_PORT}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  )
  [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]] && VLLM_ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
  [[ -n "${VLLM_MAX_NUM_SEQS:-}" ]] && VLLM_ARGS+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
  [[ -n "${VLLM_DTYPE:-}" ]] && VLLM_ARGS+=(--dtype "${VLLM_DTYPE}")
  [[ -n "${VLLM_QUANTIZATION:-}" ]] && VLLM_ARGS+=(--quantization "${VLLM_QUANTIZATION}")
  # shellcheck disable=SC2206
  [[ -n "${VLLM_EXTRA_ARGS:-}" ]] && VLLM_ARGS+=(${VLLM_EXTRA_ARGS})

  nohup "${VENV_DIR}/bin/vllm" "${VLLM_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"

  echo "==> Waiting for vLLM to become ready (model load can take a few minutes for large models)..."
  for i in $(seq 1 180); do
    if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
      echo "vLLM is ready."
      break
    fi
    sleep 2
  done
  if ! curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
    echo "Error: vLLM did not become ready in time. Check ${LOG_FILE}." >&2
    exit 1
  fi
fi

# --- 4. Warm up with a real request, same role as the Ollama script's warmup.
echo "==> Warming up model: ${VLLM_MODEL}"
curl -sf "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${VLLM_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 5}" \
  >/dev/null

echo "==> Model is loaded and ready to serve requests."
echo "OpenAI-compatible endpoint: ${VLLM_URL}/v1"
echo ""
echo "Note: point BENCHMARK_TARGET / --ollama-host-style args at ${VLLM_URL} (no /v1) for the other scripts."
