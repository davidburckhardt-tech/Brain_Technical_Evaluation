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
#
# Usage: ./scripts/run_benchmark.sh [--model <short-name>]
#   --model (see scripts/lib_model.sh for the list) sets MODEL_NAME and
#   TOKENIZER_MODEL together (they must match - see "Bugs found" in
#   RESULTS_0.5_NVIDIA.md for why keeping them in sync by hand is risky),
#   AND makes sure vLLM is actually serving that model first (downloading it
#   if not already cached) by calling start_inference_vllm.sh - one command
#   gets you a reproducible run against the right model, no manual .env
#   edits or separate server restart needed.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/lib_model.sh"
parse_model_flag "$@"
set -- "${PARSED_ARGS[@]}"

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

if [[ -n "${RESOLVED_MODEL_REPO}" ]]; then
  MODEL_NAME="${RESOLVED_MODEL_REPO}"
  TOKENIZER_MODEL="${RESOLVED_MODEL_REPO}"
  echo "==> --model given: ensuring vLLM is serving ${MODEL_NAME} (downloads it if not cached)..."
  # Pass --model through, NOT an exported VLLM_MODEL env var: start_inference_vllm.sh
  # sources .env itself, which would silently overwrite an inherited env var
  # with .env's own VLLM_MODEL= line - --model is applied AFTER that source,
  # so it's the only override that reliably wins (confirmed the hard way: a
  # run_benchmark.sh --model call silently benchmarked .env's model instead).
  "${PROJECT_ROOT}/scripts/start_inference_vllm.sh" --model "${RESOLVED_MODEL_SHORT_NAME}"
fi

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

# vLLM-specific queue/KV-cache metrics - harmless no-op (blank rows) if
# BENCHMARK_TARGET isn't actually a vLLM server (e.g. Ollama in Phase 0).
"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_vllm_metrics.py" \
  --vllm-url "${BENCHMARK_TARGET}" --output "${RUN_DIR}/vllm_metrics.csv" --interval 1 &
VLLM_METRICS_PID=$!

trap 'kill -TERM "${MONITOR_PID}" "${VLLM_METRICS_PID}" 2>/dev/null; wait "${MONITOR_PID}" "${VLLM_METRICS_PID}" 2>/dev/null' EXIT

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

"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/extract_benchmark_requests.py" \
  --input "${RUN_DIR}/benchmarks.json" --output "${RUN_DIR}/requests_detail.csv"

echo "==> Benchmark complete. Raw output (benchmarks.json / benchmarks.csv / requests_detail.csv / resource_usage.csv) saved in ${RUN_DIR}"
