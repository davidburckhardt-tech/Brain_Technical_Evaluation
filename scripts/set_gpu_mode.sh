#!/usr/bin/env bash
# Switches between single-GPU, tensor-parallel, and data-parallel vLLM
# config in .env, and stops any running vLLM server so the next script that
# needs it (run_benchmark.sh/run_chatbot_test.sh/run_rag_test.sh --model
# <name>, or start_inference_vllm.sh directly) restarts it with the new
# config.
#
# Usage: ./scripts/set_gpu_mode.sh <mode>
#   mode: "1"  - single GPU, one full model copy (VLLM_TENSOR_PARALLEL_SIZE=1,
#                VLLM_DATA_PARALLEL_SIZE=1).
#         "2"  - tensor-parallel across both GPUs: ONE model copy sharded
#                across 2 GPUs, synchronized (NCCL) on every forward pass
#                (VLLM_TENSOR_PARALLEL_SIZE=2, VLLM_DATA_PARALLEL_SIZE=1).
#                Only worth it when the model doesn't fit on one GPU - for
#                this project's small models it adds sync overhead without
#                being necessary, see "dp" below.
#         "dp" - data-parallel across both GPUs: TWO independent, FULL model
#                replicas, one per GPU, no cross-GPU sync - vLLM's own
#                APIServer load-balances requests between them
#                (VLLM_TENSOR_PARALLEL_SIZE=1, VLLM_DATA_PARALLEL_SIZE=2).
#                For a model that fits on a single GPU, this generally
#                serves more concurrent requests than tensor-parallel: no
#                NCCL overhead, and no KV-cache budget shared between
#                unrelated requests.
#
# VLLM_GPU_MEMORY_UTILIZATION is set alongside the above because they're
# coupled, not independent knobs - confirmed on real hardware (2x RTX 4070
# SUPER): 0.90 works fine with one model copy per GPU (modes "1" and "dp" -
# data-parallel replicas don't share a process/CUDA context, so each one
# gets the same single-GPU headroom), but OOMs at startup under
# tensor-parallel ("2") because each rank's own CUDA context/NCCL overhead
# needs headroom beyond what a single-GPU budget leaves (see
# start_inference_vllm.sh's header and RESULTS_0.5_NVIDIA.md). 0.80 is the
# value validated here for tensor-parallel on 2 GPUs; for other GPU counts
# this is a starting point, not a guarantee - watch for an OOM on startup
# and lower it further if needed.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

MODE="${1:?Usage: set_gpu_mode.sh <1|2|dp>}"

case "${MODE}" in
  1)
    TENSOR_PARALLEL_SIZE=1
    DATA_PARALLEL_SIZE=1
    GPU_MEMORY_UTILIZATION="0.90"
    ;;
  2)
    TENSOR_PARALLEL_SIZE=2
    DATA_PARALLEL_SIZE=1
    GPU_MEMORY_UTILIZATION="0.80"
    ;;
  dp)
    TENSOR_PARALLEL_SIZE=1
    DATA_PARALLEL_SIZE=2
    GPU_MEMORY_UTILIZATION="0.90"
    ;;
  *)
    echo "Error: unknown mode '${MODE}'. Expected 1, 2, or dp." >&2
    exit 1
    ;;
esac

echo "==> Stopping any running vLLM server..."
pkill -f "vllm serve" 2>/dev/null || true
# A killed vLLM process can leave its worker subprocesses (VLLM::EngineCore,
# VLLM::Worker_TPx) alive and holding VRAM even after the parent is gone -
# the next start would OOM immediately otherwise.
pkill -9 -f "VLLM::" 2>/dev/null || true
sleep 3

touch "${ENV_FILE}"

set_env_var() {
  local name="$1" value="$2"
  if grep -q "^${name}=" "${ENV_FILE}"; then
    sed -i.bak "s/^${name}=.*/${name}=${value}/" "${ENV_FILE}"
  else
    echo "${name}=${value}" >> "${ENV_FILE}"
  fi
}

set_env_var VLLM_TENSOR_PARALLEL_SIZE "${TENSOR_PARALLEL_SIZE}"
set_env_var VLLM_DATA_PARALLEL_SIZE "${DATA_PARALLEL_SIZE}"
set_env_var VLLM_GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
rm -f "${ENV_FILE}.bak"

echo "==> .env updated: VLLM_TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}, VLLM_DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE}, VLLM_GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "==> vLLM is stopped. It will restart automatically with this config the next time"
echo "    a script with --model <name> runs (or run ./scripts/start_inference_vllm.sh --model <name> directly)."
