#!/usr/bin/env bash
# Switches between single-GPU and multi-GPU (tensor-parallel) vLLM config in
# .env, and stops any running vLLM server so the next script that needs it
# (run_benchmark.sh/run_chatbot_test.sh/run_rag_test.sh --model <name>, or
# start_inference_vllm.sh directly) restarts it with the new config.
#
# Usage: ./scripts/set_gpu_mode.sh <num_gpus>
#   num_gpus: how many GPUs vLLM should tensor-parallel the model across.
#
# VLLM_GPU_MEMORY_UTILIZATION is set alongside VLLM_TENSOR_PARALLEL_SIZE
# because they're coupled, not independent knobs - confirmed on real
# hardware (2x RTX 4070 SUPER): 0.90 works fine on 1 GPU, but OOMs at
# startup on 2+ GPUs because each rank's own CUDA context/NCCL overhead
# needs headroom beyond what a single-GPU budget leaves (see
# start_inference_vllm.sh's header and RESULTS_0.5_NVIDIA.md). 0.80 is the
# value validated here for 2 GPUs; for other GPU counts this is a starting
# point, not a guarantee - watch for an OOM on startup and lower it further
# if needed.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

NUM_GPUS="${1:?Usage: set_gpu_mode.sh <num_gpus>}"

if [[ "${NUM_GPUS}" -le 1 ]]; then
  GPU_MEMORY_UTILIZATION="0.90"
else
  GPU_MEMORY_UTILIZATION="0.80"
fi

echo "==> Stopping any running vLLM server..."
pkill -f "vllm serve" 2>/dev/null || true
# A killed vLLM process can leave its worker subprocesses (VLLM::EngineCore,
# VLLM::Worker_TPx) alive and holding VRAM even after the parent is gone -
# the next start would OOM immediately otherwise.
pkill -9 -f "VLLM::" 2>/dev/null || true
sleep 3

touch "${ENV_FILE}"
if grep -q "^VLLM_TENSOR_PARALLEL_SIZE=" "${ENV_FILE}"; then
  sed -i.bak "s/^VLLM_TENSOR_PARALLEL_SIZE=.*/VLLM_TENSOR_PARALLEL_SIZE=${NUM_GPUS}/" "${ENV_FILE}"
else
  echo "VLLM_TENSOR_PARALLEL_SIZE=${NUM_GPUS}" >> "${ENV_FILE}"
fi
if grep -q "^VLLM_GPU_MEMORY_UTILIZATION=" "${ENV_FILE}"; then
  sed -i.bak "s/^VLLM_GPU_MEMORY_UTILIZATION=.*/VLLM_GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}/" "${ENV_FILE}"
else
  echo "VLLM_GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" >> "${ENV_FILE}"
fi
rm -f "${ENV_FILE}.bak"

echo "==> .env updated: VLLM_TENSOR_PARALLEL_SIZE=${NUM_GPUS}, VLLM_GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "==> vLLM is stopped. It will restart automatically with this config the next time"
echo "    a script with --model <name> runs (or run ./scripts/start_inference_vllm.sh --model <name> directly)."
