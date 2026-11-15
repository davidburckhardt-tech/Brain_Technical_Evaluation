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
# Usage: ./scripts/run_benchmark.sh [--model <short-name>] [--sweep]
#   --model (see scripts/lib_model.sh for the list) sets MODEL_NAME and
#   TOKENIZER_MODEL together (they must match - see "Bugs found" in
#   RESULTS_0.5_NVIDIA.md for why keeping them in sync by hand is risky),
#   AND makes sure vLLM is actually serving that model first (downloading it
#   if not already cached) by calling start_inference_vllm.sh - one command
#   gets you a reproducible run against the right model, no manual .env
#   edits or separate server restart needed.
#   --sweep runs the standard concurrency sweep (1,2,4,8,16,32,64,128 - the
#   same levels used throughout RESULTS_0.5_NVIDIA.md) as ONE command instead
#   of a hand-written shell loop, so a full Test 1 pass is reproducible with
#   a single invocation. Override the levels with BENCHMARK_CONCURRENCY_SWEEP
#   (comma-separated, e.g. "1,4,16,64"). Without --sweep, behaves exactly as
#   before: one run at BENCHMARK_CONCURRENCY (env/.env, default 2).
#   Each level's request count defaults to 100, except 300 at concurrency>=100
#   - a lower count there can't actually sustain that many requests in flight
#   long enough for GuideLLM's "Concurrency Mean" to reach the target level
#   (seen directly on real hardware: streams=128 with only 100 requests
#   measured ~99 concurrent, not 128). Override via BENCHMARK_REQUEST_COUNT.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/lib_model.sh"
parse_model_flag "$@"
set -- ${PARSED_ARGS[@]+"${PARSED_ARGS[@]}"}

USE_SWEEP=0
_REMAINING_ARGS=()
for _arg in "$@"; do
  if [[ "${_arg}" == "--sweep" ]]; then
    USE_SWEEP=1
  else
    _REMAINING_ARGS+=("${_arg}")
  fi
done
set -- ${_REMAINING_ARGS[@]+"${_REMAINING_ARGS[@]}"}

# Preserve any of these already set in the calling shell (e.g.
# `BENCHMARK_CONCURRENCY=8 ./run_benchmark.sh` for a one-off concurrency
# override) - otherwise `source .env` below silently overwrites them with
# .env's own literal values, since .env's assignments aren't conditional.
for _v in BENCHMARK_TARGET MODEL_NAME TOKENIZER_MODEL BENCHMARK_REQUEST_COUNT BENCHMARK_CONCURRENCY BENCHMARK_CONCURRENCY_SWEEP; do
  if [[ -v "${_v}" ]]; then declare "_PRESET_${_v}=${!_v}"; fi
done

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

for _v in BENCHMARK_TARGET MODEL_NAME TOKENIZER_MODEL BENCHMARK_REQUEST_COUNT BENCHMARK_CONCURRENCY BENCHMARK_CONCURRENCY_SWEEP; do
  _preset_name="_PRESET_${_v}"
  if [[ -v "${_preset_name}" ]]; then declare "${_v}=${!_preset_name}"; fi
done

BENCHMARK_TARGET="${BENCHMARK_TARGET:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-llama3:8b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-NousResearch/Meta-Llama-3-8B}"
# "__AUTO__" (rather than a fixed number) lets run_one_concurrency pick a
# sensible per-level default (see its own comment) when nothing overrides
# it - but an explicit BENCHMARK_REQUEST_COUNT (env var or .env) always wins.
BENCHMARK_REQUEST_COUNT="${BENCHMARK_REQUEST_COUNT:-__AUTO__}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-2}"
BENCHMARK_CONCURRENCY_SWEEP="${BENCHMARK_CONCURRENCY_SWEEP:-1,2,4,8,16,32,64,128}"

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

# Backstop for `set -e` aborting mid-iteration inside the loop below (e.g. a
# crashed vLLM server) - the per-iteration RETURN trap handles the normal
# case, this just makes sure no monitor_resources.py/monitor_vllm_metrics.py
# is left running in the background if the script dies abnormally.
trap 'jobs -p | xargs -r kill 2>/dev/null' EXIT

run_one_concurrency() {
  local concurrency="$1"
  local request_count="${BENCHMARK_REQUEST_COUNT}"
  if [[ "${request_count}" == "__AUTO__" ]]; then
    # 100 requests can't actually sustain high concurrency in flight long
    # enough for GuideLLM to report it accurately - see the header note.
    if [[ "${concurrency}" -ge 100 ]]; then request_count=300; else request_count=100; fi
  fi

  local timestamp safe_model_name run_dir
  timestamp="$(date +%Y%m%d_%H%M%S)"
  safe_model_name="${MODEL_NAME//[:\/]/_}"
  run_dir="${PROJECT_ROOT}/results/${safe_model_name}/${timestamp}_c${concurrency}"
  mkdir -p "${run_dir}"
  # Snapshot of the serving config (TP vs DP) in effect for this run, written
  # by start_inference_vllm.sh just above when --model was given - lets the
  # dashboard tell a tensor-parallel run apart from a data-parallel one
  # (both can show 2 active GPUs). Not meaningful for a non-vLLM target
  # (e.g. Ollama, Phase 0), so only copy it when --model triggered vLLM.
  if [[ -n "${RESOLVED_MODEL_REPO}" && -f "${PROJECT_ROOT}/results/current_serving_config.json" ]]; then
    cp "${PROJECT_ROOT}/results/current_serving_config.json" "${run_dir}/serving_config.json"
  fi

  echo "==> Running GuideLLM against ${BENCHMARK_TARGET} (model: ${MODEL_NAME}, tokenizer: ${TOKENIZER_MODEL})"
  echo "==> Fixed concurrency: ${concurrency} streams, ${request_count} requests total"
  echo "==> Results will be stored in ${run_dir}"

  # Resource monitoring: auto-detects GPU (nvidia-smi) vs CPU (psutil), runs in
  # the background for the duration of the benchmark.
  "${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_resources.py" \
    --output "${run_dir}/resource_usage.csv" --interval 1 &
  local monitor_pid=$!

  # vLLM-specific queue/KV-cache metrics - harmless no-op (blank rows) if
  # BENCHMARK_TARGET isn't actually a vLLM server (e.g. Ollama in Phase 0).
  "${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/monitor_vllm_metrics.py" \
    --vllm-url "${BENCHMARK_TARGET}" --output "${run_dir}/vllm_metrics.csv" --interval 1 &
  local vllm_metrics_pid=$!

  trap 'kill -TERM "${monitor_pid}" "${vllm_metrics_pid}" 2>/dev/null; wait "${monitor_pid}" "${vllm_metrics_pid}" 2>/dev/null' RETURN

  (
    cd "${run_dir}"
    "${VENV_DIR}/bin/guidellm" run \
      --backend "kind=openai_http,target=${BENCHMARK_TARGET},model=${MODEL_NAME},validate_backend=false" \
      --data "kind=synthetic_text,prompt_tokens=550,output_tokens=150" \
      --tokenizer "kind=huggingface_auto,model=${TOKENIZER_MODEL}" \
      --profile "kind=concurrent,streams=${concurrency}" \
      --constraint "kind=max_requests,count=${request_count}" \
      --output "kind=json,path=benchmarks.json" \
      --output "kind=csv,path=benchmarks.csv" \
      --disable-console-interactive
  )

  "${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/scripts/extract_benchmark_requests.py" \
    --input "${run_dir}/benchmarks.json" --output "${run_dir}/requests_detail.csv"

  echo "==> Benchmark complete. Raw output (benchmarks.json / benchmarks.csv / requests_detail.csv / resource_usage.csv) saved in ${run_dir}"
}

if [[ "${USE_SWEEP}" -eq 1 ]]; then
  echo "==> Sweep mode: concurrency levels ${BENCHMARK_CONCURRENCY_SWEEP}"
  IFS=',' read -ra _SWEEP_LEVELS <<< "${BENCHMARK_CONCURRENCY_SWEEP}"
  for _level in "${_SWEEP_LEVELS[@]}"; do
    echo ""
    echo "=== concurrency ${_level} ==="
    run_one_concurrency "${_level}"
  done
  echo ""
  echo "==> Sweep complete (${BENCHMARK_CONCURRENCY_SWEEP})."
else
  run_one_concurrency "${BENCHMARK_CONCURRENCY}"
fi
