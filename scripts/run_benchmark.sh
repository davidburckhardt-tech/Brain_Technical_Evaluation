#!/usr/bin/env bash
# Runs the GuideLLM benchmark against the running inference server (Ollama in
# Phase 0, vLLM in later phases) and stores the raw results under results/.
#
# Profile is "concurrent" with a FIXED stream count (BENCHMARK_CONCURRENCY in
# .env), not GuideLLM's "sweep" profile: sweep auto-derives its own
# concurrency/rate levels from a throughput probe, and on this slow local
# hardware some of those derived rates were too low to complete even one
# request within their time budget (see PROJECT_PLAN.md §9). A fixed stream
# count sidesteps that entirely, and still exercises real concurrency.
#
# Note: GuideLLM's CLI changed significantly across versions (this was
# validated against guidellm==0.8.1). If flags below error out, check:
#   .venv/bin/guidellm run --help

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Preserve any of these already set in the calling shell (e.g.
# `BENCHMARK_CONCURRENCY=8 ./run_benchmark.sh` for a concurrency sweep) -
# otherwise `source .env` below silently overwrites them with .env's own
# literal values, since .env's assignments aren't conditional.
for _v in BENCHMARK_TARGET MODEL_NAME TOKENIZER_MODEL BENCHMARK_REQUEST_COUNT BENCHMARK_CONCURRENCY; do
  if [[ -v "${_v}" ]]; then declare "_PRESET_${_v}=${!_v}"; fi
done

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

for _v in BENCHMARK_TARGET MODEL_NAME TOKENIZER_MODEL BENCHMARK_REQUEST_COUNT BENCHMARK_CONCURRENCY; do
  _preset_name="_PRESET_${_v}"
  if [[ -v "${_preset_name}" ]]; then declare "${_v}=${!_preset_name}"; fi
done

BENCHMARK_TARGET="${BENCHMARK_TARGET:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-NousResearch/Meta-Llama-3-8B}"
BENCHMARK_REQUEST_COUNT="${BENCHMARK_REQUEST_COUNT:-10}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-2}"

if [[ ! -x "${VENV_DIR}/bin/guidellm" ]]; then
  echo "Error: GuideLLM is not installed. Run scripts/setup.sh first." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_MODEL_NAME="${MODEL_NAME//[:\/]/_}"
RUN_DIR="${PROJECT_ROOT}/results/${SAFE_MODEL_NAME}/${TIMESTAMP}_c${BENCHMARK_CONCURRENCY}"
mkdir -p "${RUN_DIR}"

echo "==> Running GuideLLM against ${BENCHMARK_TARGET} (model: ${MODEL_NAME}, tokenizer: ${TOKENIZER_MODEL})"
echo "==> Fixed concurrency: ${BENCHMARK_CONCURRENCY} streams, ${BENCHMARK_REQUEST_COUNT} requests total"
echo "==> Results will be stored in ${RUN_DIR}"

# Resource monitoring: auto-detects GPU (nvidia-smi) vs CPU (psutil), runs in
# the background for the duration of the benchmark.
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_resources.py" \
  --output "${RUN_DIR}/resource_usage.csv" --interval 1 &
MONITOR_PID=$!
trap 'kill -TERM "${MONITOR_PID}" 2>/dev/null; wait "${MONITOR_PID}" 2>/dev/null' EXIT

(
  cd "${RUN_DIR}"
  "${VENV_DIR}/bin/guidellm" run \
    --backend "kind=openai_http,target=${BENCHMARK_TARGET},model=${MODEL_NAME},validate_backend=false" \
    --data "kind=synthetic_text,prompt_tokens=550,output_tokens=150" \
    --tokenizer "kind=huggingface_auto,model=${TOKENIZER_MODEL}" \
    --profile "kind=concurrent,streams=${BENCHMARK_CONCURRENCY}" \
    --constraint "kind=max_requests,count=${BENCHMARK_REQUEST_COUNT}" \
    --output "kind=json,path=benchmarks.json" \
    --output "kind=csv,path=benchmarks.csv" \
    --disable-console-interactive
)

echo "==> Benchmark complete. Raw output (benchmarks.json / benchmarks.csv / resource_usage.csv) saved in ${RUN_DIR}"
