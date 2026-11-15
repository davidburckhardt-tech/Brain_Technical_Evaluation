#!/usr/bin/env python3
"""Scans results/ and builds dashboard/index.html - a self-contained HTML
page (data embedded inline; Chart.js loaded from CDN for the line /
stacked-bar / histogram visualizations that a hand-rolled CSS bar chart
can't do) grouping every run by model, so it stays useful once more models
are tested.

Re-run this any time new results land under results/.
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Historical chatbot runs from before --ollama-num-parallel was added to
# simulate_users.py. Known from the order they were run in this session.
KNOWN_NUM_PARALLEL_BY_FILENAME = {
    "20261107_204825_chatbot.json": "1",
    "20261107_205940_chatbot.json": "2",
    "20261107_210047_chatbot.json": "3",
}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_resource_usage(path: Path) -> list[dict]:
    rows = []
    for row in read_csv_rows(path):
        rows.append({
            "elapsed_s": _float(row.get("elapsed_s")),
            "mode": row.get("mode"),
            "device": row.get("device"),
            "util_percent": _float(row.get("util_percent")),
            "memory_used_mb": _float(row.get("memory_used_mb")),
            "memory_total_mb": _float(row.get("memory_total_mb")),
        })
    return rows


def read_vllm_metrics(path: Path) -> list[dict]:
    rows = []
    for row in read_csv_rows(path):
        rows.append({
            "elapsed_s": _float(row.get("elapsed_s")),
            "num_requests_running": _float(row.get("num_requests_running")),
            "num_requests_waiting": _float(row.get("num_requests_waiting")),
            "gpu_cache_usage_perc": _float(row.get("gpu_cache_usage_perc")),
        })
    return rows


def infer_gpu_count(resource_usage_rows: list[dict]) -> int | None:
    """How many GPUs vLLM was actually tensor-parallel across for this run,
    not how many the monitor sampled - monitor_resources.py always samples
    every physical GPU on the box, so in 1-GPU (TP=1) mode gpu1 still shows
    up in the CSV but sits idle (~18MB, just the driver's baseline). A GPU
    only counts as "active" if its mean memory usage is well above that."""
    gpu_mem = {}
    for row in resource_usage_rows:
        dev = row.get("device")
        if not dev or not dev.startswith("gpu"):
            continue
        gpu_mem.setdefault(dev, []).append(row.get("memory_used_mb") or 0)
    if not gpu_mem:
        return None
    active = [dev for dev, vals in gpu_mem.items() if vals and (sum(vals) / len(vals)) > 500]
    return len(active) if active else 1


def read_serving_config(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_serving_mode(serving_config: dict | None, gpu_count_fallback: int | None) -> str | None:
    """"1gpu" / "2gpu-tp" (tensor-parallel: one model sharded across 2 GPUs,
    synced every forward pass) / "2gpu-dp" (data-parallel: 2 independent
    full replicas, one per GPU, no sync). Both TP and DP show 2 active GPUs
    in resource_usage.csv, so that alone can't tell them apart - this needs
    the serving_config.json sidecar (written by start_inference_vllm.sh) to
    disambiguate. Runs from before that sidecar existed (DP didn't exist
    yet) fall back to the GPU-count heuristic, where 2 active GPUs can only
    mean tensor-parallel."""
    if serving_config is not None:
        if serving_config.get("data_parallel_size", 1) > 1:
            return "2gpu-dp"
        if serving_config.get("tensor_parallel_size", 1) > 1:
            return "2gpu-tp"
        return "1gpu"
    if gpu_count_fallback == 1:
        return "1gpu"
    if gpu_count_fallback == 2:
        return "2gpu-tp"
    return None


def collect_test1_runs() -> list[dict]:
    runs = []
    for benchmarks_json in sorted(RESULTS_DIR.glob("*/*/benchmarks.json")):
        run_dir = benchmarks_json.parent
        with open(benchmarks_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for benchmark in data.get("benchmarks", []):
            config = benchmark.get("config", {})
            metrics = benchmark.get("metrics", {})
            model = config.get("backend", {}).get("model", "unknown")
            target = config.get("backend", {}).get("target")
            strategy = config.get("strategy", {})

            def stat(metric_key, stat_key="mean"):
                try:
                    return metrics[metric_key]["successful"][stat_key]
                except (KeyError, TypeError):
                    return None

            concurrency = strategy.get("max_concurrency") or strategy.get("streams") or strategy.get("worker_count")

            requests_detail = []
            for row in read_csv_rows(run_dir / "requests_detail.csv"):
                requests_detail.append({
                    "request_number": int(row["request_number"]) if row.get("request_number") else None,
                    "status": row.get("status"),
                    "latency_s": _float(row.get("request_latency_s")),
                    "ttft_ms": _float(row.get("ttft_ms")),
                })

            resource_usage = read_resource_usage(run_dir / "resource_usage.csv")
            gpu_count = infer_gpu_count(resource_usage)
            serving_config = read_serving_config(run_dir / "serving_config.json")
            runs.append({
                "run_dir": run_dir.name,
                "model": model,
                "target": target,
                "strategy": strategy.get("type_"),
                "concurrency": concurrency,
                "gpu_count": gpu_count,
                "serving_mode": classify_serving_mode(serving_config, gpu_count),
                "duration_s": benchmark.get("duration"),
                "ttft_ms_mean": stat("time_to_first_token_ms"),
                "tpot_ms_mean": stat("time_per_output_token_ms"),
                "itl_ms_mean": stat("inter_token_latency_ms"),
                "request_latency_s_mean": stat("request_latency"),
                "input_tokens_per_second_mean": stat("prompt_tokens_per_second"),
                "output_tokens_per_second_mean": stat("output_tokens_per_second"),
                "total_tokens_per_second_mean": stat("tokens_per_second"),
                "error_rate": (
                    metrics.get("request_totals", {}).get("errored", 0) / metrics["request_totals"]["total"]
                    if metrics.get("request_totals", {}).get("total")
                    else None
                ),
                "requests_detail": requests_detail,
                "resource_usage": resource_usage,
                "vllm_metrics": read_vllm_metrics(run_dir / "vllm_metrics.csv"),
            })
    return runs


def normalize_turns(user: dict) -> list[dict]:
    """Supports both the current multi-turn result format (user["turns"]
    present) and legacy single-shot result files recorded before --turns
    existed, by synthesizing a single turn-1 record from the old top-level
    fields - so old and new chatbot runs can share one dashboard code path."""
    if "turns" in user:
        return user["turns"]
    return [{
        "turn_id": 1,
        "error": user.get("error"),
        "ttft_s": user.get("ttft_s"),
        "generation_duration_s": None,
        "total_latency_s": user.get("total_latency_s"),
        "context_tokens": None,
        "output_tokens": user.get("output_tokens_approx"),
        "output_tokens_per_second": user.get("output_tokens_per_second_approx"),
        "user_input_tokens_approx": None,
        "cumulative_input_tokens_approx": None,
        "cumulative_output_tokens": user.get("output_tokens_approx"),
        "context_window_utilization": None,
    }]


def collect_test2_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*/*_chatbot.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        num_parallel = data.get("ollama_num_parallel") or KNOWN_NUM_PARALLEL_BY_FILENAME.get(path.name)
        users = data.get("users", [])

        normalized_users = [
            {
                "user_id": u["user_id"],
                "category": u.get("category"),
                "ttft_s": u.get("ttft_s"),
                "total_latency_s": u.get("total_latency_s"),
                "output_tokens_per_second_approx": u.get("output_tokens_per_second_approx"),
                "turns": normalize_turns(u),
            }
            for u in users
        ]

        resource_csv = path.with_name(path.stem + "_resources.csv")
        resource_usage = read_resource_usage(resource_csv)
        serving_config = read_serving_config(path.with_name(path.stem + "_serving_config.json"))
        runs.append({
            "run_file": path.name,
            "model": data.get("model", "unknown"),
            "target": data.get("target"),
            "ollama_num_parallel": num_parallel,
            "concurrent_users": data.get("concurrent_users"),
            "turns_per_conversation": data.get("turns_per_conversation", 1),
            "max_context_tokens": data.get("max_context_tokens"),
            "overall_duration_s": data.get("overall_duration_s"),
            "error_rate": data.get("error_rate"),
            "serving_mode": classify_serving_mode(serving_config, infer_gpu_count(resource_usage)),
            "users": normalized_users,
            "resource_usage": resource_usage,
        })
    return runs


def collect_test3_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*/*_rag_q*_c*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = data.get("queries", [])
        succeeded = [q for q in queries if not q.get("error")]

        def avg(key):
            vals = [q.get(key) for q in succeeded if q.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        hit_rate = (
            sum(1 for q in succeeded if q.get("expected_chunk_found_in_top_k")) / len(succeeded)
            if succeeded else None
        )

        per_request = [
            {
                "index": q.get("index"),
                "system_prompt_tokens_approx": q.get("system_prompt_tokens_approx"),
                "user_query_tokens_approx": q.get("user_query_tokens_approx"),
                "context_tokens_approx": q.get("context_tokens_approx"),
                "output_tokens": q.get("output_tokens"),
            }
            for q in succeeded
        ]

        resource_csv = path.with_name(path.stem + "_resources.csv")
        resource_usage = read_resource_usage(resource_csv)
        serving_config = read_serving_config(path.with_name(path.stem + "_serving_config.json"))
        runs.append({
            "run_file": path.name,
            "model": data.get("model", "unknown"),
            "num_queries": data.get("num_queries"),
            "concurrency": data.get("concurrency"),
            "top_k": data.get("top_k"),
            "wall_duration_s": data.get("wall_duration_s"),
            "error_rate": data.get("error_rate"),
            "serving_mode": classify_serving_mode(serving_config, infer_gpu_count(resource_usage)),
            "avg_embedding_s": avg("embedding_duration_s"),
            "avg_vector_search_s": avg("vector_search_duration_s"),
            "avg_retrieval_s": avg("retrieval_duration_s"),
            "avg_generation_s": avg("generation_duration_s"),
            "avg_input_tokens": avg("input_tokens"),
            "avg_output_tokens": avg("output_tokens"),
            "avg_similarity_score": avg("avg_similarity_score"),
            "expected_chunk_hit_rate": hit_rate,
            "per_request": per_request,
            "resource_usage": resource_usage,
        })
    return runs


# Phase = which hardware/environment a model's runs came from, not the
# model itself - Phase 0 (Mac, Ollama) and Phase 0.5 (brainlab-01 server,
# vLLM on real NVIDIA GPUs) used disjoint model sets, so every run for a
# given model name came from exactly one phase in this project's history.
# Detected from the GuideLLM backend target's port rather than hand-tracked,
# since that's recorded in every benchmarks.json without extra bookkeeping.
# Extend PHASE_PORTS/PHASE_LABELS/PHASE_ORDER when a new phase's hardware
# (a different host/port) comes online - anything undetected falls into
# "unknown" rather than being silently dropped or misfiled.
PHASE_PORTS = {"11434": "phase0", "8000": "phase0.5"}
PHASE_LABELS = {
    "phase0": "Phase 0 : Mac (Ollama)",
    "phase0.5": "Phase 0.5 : Server (vLLM, NVIDIA)",
    "unknown": "Unclassified",
}
PHASE_ORDER = ["phase0", "phase0.5"]


def detect_phase(target: str | None) -> str:
    if not target:
        return "unknown"
    for port, phase in PHASE_PORTS.items():
        if f":{port}" in target:
            return phase
    return "unknown"


def group_by_model(test1, test2, test3) -> dict:
    models = set(r["model"] for r in test1) | set(r["model"] for r in test2) | set(r["model"] for r in test3)

    def model_phase(model: str) -> str:
        target = next((r.get("target") for r in test1 if r["model"] == model and r.get("target")), None)
        if target is None:
            target = next((r.get("target") for r in test2 if r["model"] == model and r.get("target")), None)
        return detect_phase(target)

    grouped: dict[str, dict] = {}
    for model in sorted(models):
        phase = model_phase(model)
        grouped.setdefault(phase, {})[model] = {
            "test1": [r for r in test1 if r["model"] == model],
            "test2": [r for r in test2 if r["model"] == model],
            "test3": [r for r in test3 if r["model"] == model],
        }

    def phase_sort_key(phase):
        return (0, PHASE_ORDER.index(phase)) if phase in PHASE_ORDER else (1, phase)

    return {phase: grouped[phase] for phase in sorted(grouped.keys(), key=phase_sort_key)}


HTML_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Brain Demo - Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    color-scheme: light dark;
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #666666;
    --border: #e2e2e2;
    --accent: #3b6fd6;
    --accent2: #e8824a;
    --accent3: #4caf7d;
    --accent4: #9b6fd6;
    --bar-bg: #eef2fb;
    --card-bg: #f7f8fa;
    --test1: #3b6fd6;
    --test1-bg: #eef3fd;
    --test2: #0f9d6b;
    --test2-bg: #e9f8f1;
    --test3: #8a3fd1;
    --test3-bg: #f4edfc;
    --warn: #c97a1a;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #15181d;
      --fg: #e8e8e8;
      --muted: #9aa0a6;
      --border: #30343a;
      --accent: #6d9bff;
      --accent2: #ff9d6d;
      --accent3: #6dd6a0;
      --accent4: #c09bff;
      --bar-bg: #202636;
      --card-bg: #1b1f26;
      --test1: #6d9bff;
      --test1-bg: #182236;
      --test2: #3fd6a0;
      --test2-bg: #122a22;
      --test3: #c09bff;
      --test3-bg: #241c33;
      --warn: #e8a33d;
    }
  }
  :root[data-theme="dark"] {
    --bg: #15181d;
    --fg: #e8e8e8;
    --muted: #9aa0a6;
    --border: #30343a;
    --accent: #6d9bff;
    --accent2: #ff9d6d;
    --accent3: #6dd6a0;
    --accent4: #c09bff;
    --bar-bg: #202636;
    --card-bg: #1b1f26;
    --test1: #6d9bff;
    --test1-bg: #182236;
    --test2: #3fd6a0;
    --test2-bg: #122a22;
    --test3: #c09bff;
    --test3-bg: #241c33;
    --warn: #e8a33d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 24px 16px 48px;
  }
  main { max-width: 1040px; margin: 0 auto; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .subtitle { color: var(--muted); margin-bottom: 24px; font-size: 13px; }

  /* Phase is the top-level split (different hardware/environment entirely -
     Mac vs the NVIDIA server - not comparable numbers, so it sits above even
     the model picker) - a dropdown rather than a tab row, since the list is
     meant to grow as more phases/hardware are added. */
  .phase-select-row { margin-bottom: 16px; }
  .phase-select-row label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-right: 8px; }
  #phaseSelect { font-size: 14px; font-weight: 700; padding: 8px 12px; }

  .model-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .model-tab {
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 4px;
    cursor: pointer; background: transparent; color: var(--fg); font-size: 13px;
  }
  .model-tab[aria-selected="true"] { background: var(--accent); color: white; border-color: var(--accent); }

  /* Test-level tabs: one big, unmistakably color-coded control per test, so
     switching between Test 1/2/3 reads as switching sections, not just
     changing a filter - each test keeps its color everywhere below (the
     active-tab fill, the panel's top border and badge, the section note). */
  .test-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
  .test-tab {
    text-align: left; cursor: pointer; border: 2px solid var(--border); border-radius: 4px;
    background: var(--card-bg); color: var(--fg); padding: 12px 14px;
    display: flex; flex-direction: column; gap: 2px;
  }
  .test-tab .test-tab-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 700; }
  .test-tab .test-tab-title { font-size: 14px; font-weight: 600; }
  .test-tab[data-test="test1"][aria-selected="true"] { border-color: var(--test1); background: var(--test1-bg); }
  .test-tab[data-test="test2"][aria-selected="true"] { border-color: var(--test2); background: var(--test2-bg); }
  .test-tab[data-test="test3"][aria-selected="true"] { border-color: var(--test3); background: var(--test3-bg); }
  .test-tab[data-test="test1"][aria-selected="true"] .test-tab-label { color: var(--test1); }
  .test-tab[data-test="test2"][aria-selected="true"] .test-tab-label { color: var(--test2); }
  .test-tab[data-test="test3"][aria-selected="true"] .test-tab-label { color: var(--test3); }

  .test-panel { border-top: 4px solid var(--border); border-radius: 0 0 4px 4px; padding: 20px 0 4px; }
  .test-panel[data-test="test1"] { border-top-color: var(--test1); }
  .test-panel[data-test="test2"] { border-top-color: var(--test2); }
  .test-panel[data-test="test3"] { border-top-color: var(--test3); }
  section { margin-bottom: 36px; }
  h2 { font-size: 18px; margin: 0 0 4px; display: flex; align-items: center; gap: 8px; }
  h2 .test-badge { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
  .test-panel[data-test="test1"] h2 .test-badge { background: var(--test1); }
  .test-panel[data-test="test2"] h2 .test-badge { background: var(--test2); }
  .test-panel[data-test="test3"] h2 .test-badge { background: var(--test3); }
  /* Within Test 1, a second-level toggle for 1-GPU vs 2-GPU (tensor-parallel)
     runs - visually smaller/subordinate to the big test-tabs above so it
     reads as "a view on Test 1", not another top-level section. */
  .gpu-tabs { display: flex; gap: 8px; margin-bottom: 18px; }
  .gpu-tab {
    padding: 6px 16px; border-radius: 4px; border: 1.5px solid var(--border);
    background: var(--card-bg); color: var(--muted); font-size: 13px; font-weight: 600;
    cursor: pointer;
  }
  .gpu-tab[aria-selected="true"] { background: var(--test1); border-color: var(--test1); color: #fff; }
  h3 { font-size: 13px; margin: 20px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.02em; }
  h3 .step-num {
    display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px;
    border-radius: 50%; background: var(--bar-bg); color: var(--fg); font-size: 10.5px; margin-right: 6px;
  }
  .test-note { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
  .chart-note { color: var(--muted); font-size: 11.5px; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-bottom: 16px; overflow-x: auto; display: block; }
  thead, tbody { display: table; width: 100%; table-layout: fixed; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .bar-chart-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 4px; padding: 16px 18px; margin-bottom: 20px; }
  .bar-chart-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
  .bar-chart-title { font-size: 15px; font-weight: 700; }
  .bar-chart-unit { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; background: var(--bar-bg); color: var(--accent); flex-shrink: 0; }
  .bar-chart-subtitle { font-size: 12px; color: var(--muted); margin: 2px 0 14px; }
  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: flex; align-items: center; gap: 12px; }
  .bar-label { min-width: 48px; flex-shrink: 0; font-size: 12.5px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; background: var(--bar-bg); border-radius: 4px; height: 20px; position: relative; min-width: 40px; }
  .bar-fill { background: var(--accent); height: 100%; border-radius: 4px; min-width: 6px; }
  .bar-fill.highlight { background: var(--warn); }
  .bar-value { width: 76px; flex-shrink: 0; text-align: right; font-size: 13.5px; color: var(--fg); white-space: nowrap; }
  .bar-value.highlight { color: var(--warn); font-weight: 700; }
  .bar-chart-caption { font-size: 11.5px; color: var(--muted); margin-top: 12px; }
  .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
  .chart-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 4px; padding: 14px; margin-bottom: 20px; }
  .chart-card canvas { max-height: 320px; }
  .run-select { margin-bottom: 10px; font-size: 12.5px; }
  .run-select .chart-note { display: inline; margin-top: 0; margin-left: 10px; }
  select {
    background: var(--bg); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; font-size: 12.5px; max-width: 100%;
  }
  .stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }
  .stat-row b { color: var(--fg); }

  /* "Performance overview" tiles: the few numbers worth seeing before
     scrolling to any chart - one highlighted headline metric (colored in
     the test's own color) plus supporting ones in plain text. */
  .overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 20px; margin-bottom: 24px; }
  .overview-tile { padding: 2px 0; }
  .overview-label { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .overview-value { font-size: 30px; font-weight: 700; line-height: 1.1; }
  .overview-value.headline { color: var(--test1); }
  .test-panel[data-test="test1"] .overview-value.headline { color: var(--test1); }
  .test-panel[data-test="test2"] .overview-value.headline { color: var(--test2); }
  .test-panel[data-test="test3"] .overview-value.headline { color: var(--test3); }
  .overview-caption { font-size: 11.5px; color: var(--muted); margin-top: 4px; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Brain Demo : Benchmark Results</h1>
  <p class="subtitle">Generated by scripts/build_dashboard.py</p>

  <div class="phase-select-row"><label for="phaseSelect">Phase</label><select id="phaseSelect"></select></div>
  <div class="model-tabs" id="modelTabs"></div>

  <div class="test-tabs" id="testTabs"></div>
  <div class="test-panel" id="testPanel"></div>
</main>

<script>
const DATA = __DATA_JSON__;
const PHASE_LABELS = __PHASE_LABELS_JSON__;
const CHART_COLORS = {
  accent: getCss('--accent'), accent2: getCss('--accent2'), accent3: getCss('--accent3'),
  accent4: getCss('--accent4'), muted: getCss('--muted'), border: getCss('--border'), fg: getCss('--fg'),
};
function getCss(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return '&ndash;';
  return Number(n).toFixed(digits);
}
function pct(v) {
  return (v === null || v === undefined) ? '-' : Math.round(v * 100) + '%';
}
function fmtInt(n) {
  return (n === null || n === undefined) ? '&ndash;' : Math.round(n).toLocaleString();
}
function overviewTile(label, value, caption, headline) {
  return `<div class="overview-tile">
    <div class="overview-label">${label}</div>
    <div class="overview-value${headline ? ' headline' : ''}">${value}</div>
    <div class="overview-caption">${caption}</div>
  </div>`;
}
function overviewGrid(tiles) {
  return `<div class="overview-grid">${tiles.join('')}</div>`;
}
function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[idx];
}

const activeCharts = [];
function destroyCharts() {
  while (activeCharts.length) activeCharts.pop().destroy();
}
function makeChart(canvasId, config) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  const chart = new Chart(el.getContext('2d'), config);
  activeCharts.push(chart);
  return chart;
}

// A titled card around a plain horizontal-bar list: label left, bar,
// value right. The single highest value is called out in the warn color
// rather than accent - just a visual "this one stands out", not a verdict
// on good/bad (e.g. TTFT climbing with concurrency is expected, not a
// regression), so the caption never says "lower/higher is better".
function barChart(rows, labelFn, valueFn, opts) {
  const { title, subtitle, unit = '', caption, digits = 2 } = opts;
  const values = rows.map(valueFn);
  const maxVal = Math.max(...values.filter(v => v !== null && v !== undefined), 0.001);
  let html = `<div class="bar-chart-card">
    <div class="bar-chart-header"><div class="bar-chart-title">${title}</div>${unit ? `<div class="bar-chart-unit">${unit}</div>` : ''}</div>
    ${subtitle ? `<div class="bar-chart-subtitle">${subtitle}</div>` : ''}
    <div class="bars">`;
  for (let i = 0; i < rows.length; i++) {
    const v = values[i];
    const pctw = v === null || v === undefined ? 0 : Math.max((v / maxVal) * 100, 2);
    const isMax = v !== null && v !== undefined && v === maxVal;
    html += `<div class="bar-row">
      <div class="bar-label">${labelFn(rows[i])}</div>
      <div class="bar-track"><div class="bar-fill${isMax ? ' highlight' : ''}" style="width:${pctw}%"></div></div>
      <div class="bar-value${isMax ? ' highlight' : ''}">${fmt(v, digits)}</div>
    </div>`;
  }
  html += '</div>';
  if (caption) html += `<div class="bar-chart-caption">${caption}</div>`;
  return html + '</div>';
}

// ---------- Serving-mode (1 GPU / tensor-parallel / data-parallel) sub-tabs,
// shared by all three tests - both tensor-parallel and data-parallel show 2
// active GPUs in resource_usage.csv, so classification comes from the
// serving_config.json sidecar (see classify_serving_mode in the Python
// build script), with GPU-count as a fallback for older runs. ----------

const SERVING_MODE_LABELS = {
  '1gpu': '1 GPU',
  '2gpu-tp': '2 GPUs (tensor-parallel)',
  '2gpu-dp': '2 GPUs (data-parallel)',
};
const SERVING_MODE_ORDER = ['1gpu', '2gpu-tp', '2gpu-dp'];

function servingModeTabsHtml(allRuns) {
  const modes = SERVING_MODE_ORDER.filter(m => allRuns.some(r => r.serving_mode === m));
  if (modes.length === 0) return { modes, html: '' };
  if (currentServingMode == null || !modes.includes(currentServingMode)) currentServingMode = modes[0];
  if (modes.length < 2) return { modes, html: '' };
  const html = '<div class="gpu-tabs">' + modes.map(m =>
    `<button class="gpu-tab" data-mode="${m}" aria-selected="${m === currentServingMode}">${SERVING_MODE_LABELS[m] || m}</button>`
  ).join('') + '</div>';
  return { modes, html };
}

function filterByServingMode(allRuns, modes) {
  return modes.length ? allRuns.filter(r => r.serving_mode === currentServingMode) : allRuns;
}

function wireServingModeTabs() {
  document.querySelectorAll('#testPanel .gpu-tab').forEach(btn => {
    btn.onclick = () => {
      currentServingMode = btn.getAttribute('data-mode');
      renderTestPanel();
    };
  });
}

// ---------- Test 1: Hardware / Model Serving ----------

function renderTest1(allRuns) {
  if (!allRuns.length) return '<p class="empty">No Test 1 data for this model.</p>';
  const { modes, html: gpuTabs } = servingModeTabsHtml(allRuns);
  const runs = filterByServingMode(allRuns, modes);
  if (!runs.length) return gpuTabs + '<p class="empty">No Test 1 data for this serving mode.</p>';

  const maxConcurrencyRun = [...runs].filter(r => r.concurrency != null).sort((a, b) => b.concurrency - a.concurrency)[0];
  let html = gpuTabs;
  if (maxConcurrencyRun) {
    html += overviewGrid([
      overviewTile('Output throughput', fmtInt(maxConcurrencyRun.output_tokens_per_second_mean), 'tokens/sec', true),
      overviewTile('Mean TTFT', fmt(maxConcurrencyRun.ttft_ms_mean, 1), `ms &middot; concurrency ${maxConcurrencyRun.concurrency}`),
      overviewTile('Mean TPOT', fmt(maxConcurrencyRun.tpot_ms_mean, 2), 'ms'),
    ]);
  }
  html += '<div class="test-note">Synthetic load, GuideLLM. "Concurrency" = number of simultaneous users/requests sent at once (fixed, not auto-scaled).</div>';
  html += barChart(runs, r => (r.concurrency != null ? `${r.concurrency}x` : r.strategy), r => r.ttft_ms_mean, {
    title: 'TTFT by concurrency', subtitle: 'Synthetic load &middot; GuideLLM', unit: 'ms', caption: 'Values represent mean TTFT.',
  });
  html += '<table><thead><tr><th>Strategy</th><th>Concurrency</th><th>Duration (s)</th><th>Mean TTFT (ms)</th><th>Mean TPOT (ms)</th><th>Latency/req (s)</th><th>Output tok/s</th><th>Error rate</th></tr></thead><tbody>';
  for (const r of runs) {
    html += `<tr><td>${r.strategy ?? '-'}</td><td>${r.concurrency ?? '-'}</td><td>${fmt(r.duration_s)}</td>
      <td>${fmt(r.ttft_ms_mean)}</td><td>${fmt(r.tpot_ms_mean)}</td><td>${fmt(r.request_latency_s_mean)}</td>
      <td>${fmt(r.output_tokens_per_second_mean)}</td><td>${pct(r.error_rate)}</td></tr>`;
  }
  html += '</tbody></table>';

  const runsWithDetail = runs.filter(r => r.requests_detail && r.requests_detail.length);
  html += '<h3><span class="step-num">1</span>Request latency vs request number</h3>';
  if (runsWithDetail.length) {
    html += '<div class="run-select">Concurrent users: <select id="t1RunSelect"></select></div>';
    html += '<div class="chart-card"><canvas id="t1LatencyChart"></canvas></div>';
  } else {
    html += '<p class="empty">No per-request detail available for these runs (requires requests_detail.csv).</p>';
  }

  html += '<h3><span class="step-num">2</span>Throughput vs concurrency</h3>';
  html += '<div class="chart-card"><canvas id="t1ThroughputChart"></canvas><div class="chart-note" id="t1SaturationNote"></div></div>';

  html += '<h3><span class="step-num">3</span>Hardware utilization vs time</h3>';
  const runsWithResources = runs.filter(r => r.resource_usage && r.resource_usage.length);
  if (runsWithResources.length) {
    html += '<div class="run-select">Concurrent users: <select id="t1HwRunSelect"></select></div>';
    html += '<div class="chart-card"><canvas id="t1HwChart"></canvas></div>';
  } else {
    html += '<p class="empty">No resource_usage.csv found for these runs.</p>';
  }

  return html;
}

function wireTest1Charts(allRuns) {
  wireServingModeTabs();
  const modes = SERVING_MODE_ORDER.filter(m => allRuns.some(r => r.serving_mode === m));
  const runs = filterByServingMode(allRuns, modes);
  if (!runs.length) return;

  const runsWithDetail = runs.filter(r => r.requests_detail && r.requests_detail.length);
  const runSelect = document.getElementById('t1RunSelect');
  if (runSelect) {
    runSelect.innerHTML = buildRunOptions(runsWithDetail, r => r.concurrency ?? '?');
    const draw = () => {
      const r = runsWithDetail[Number(runSelect.value)];
      const successful = r.requests_detail.filter(d => d.status === 'successful');
      makeChart('t1LatencyChart', {
        type: 'line',
        data: {
          labels: successful.map(d => d.request_number),
          datasets: [
            { label: 'Latency (s)', data: successful.map(d => d.latency_s), borderColor: CHART_COLORS.accent, backgroundColor: CHART_COLORS.accent, yAxisID: 'y', pointRadius: 2, tension: 0.15 },
            { label: 'TTFT (ms)', data: successful.map(d => d.ttft_ms), borderColor: CHART_COLORS.accent2, backgroundColor: CHART_COLORS.accent2, yAxisID: 'y1', pointRadius: 2, tension: 0.15 },
          ],
        },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: 'Request number (submission order)' } },
            y: { title: { display: true, text: 'Latency (s)' }, position: 'left' },
            y1: { title: { display: true, text: 'TTFT (ms)' }, position: 'right', grid: { drawOnChartArea: false } },
          },
        },
      });
    };
    runSelect.onchange = () => { destroyChartById('t1LatencyChart'); draw(); };
    draw();
  }

  makeChart('t1ThroughputChart', (() => {
    const sorted = [...runs].filter(r => r.concurrency != null).sort((a, b) => a.concurrency - b.concurrency);
    const best = sorted.reduce((acc, r) => (r.output_tokens_per_second_mean > (acc?.output_tokens_per_second_mean ?? -1) ? r : acc), null);
    const note = document.getElementById('t1SaturationNote');
    if (note) {
      note.textContent = best ? `Apparent saturation point: concurrency=${best.concurrency} (highest output tok/s observed: ${fmt(best.output_tokens_per_second_mean, 1)}). Beyond this, more concurrency stopped increasing output throughput.` : '';
    }
    return {
      type: 'line',
      data: {
        labels: sorted.map(r => r.concurrency),
        datasets: [
          { label: 'Input tok/s', data: sorted.map(r => r.input_tokens_per_second_mean), borderColor: CHART_COLORS.accent, backgroundColor: CHART_COLORS.accent, tension: 0.15 },
          { label: 'Output tok/s', data: sorted.map(r => r.output_tokens_per_second_mean), borderColor: CHART_COLORS.accent2, backgroundColor: CHART_COLORS.accent2, tension: 0.15 },
          { label: 'Total tok/s', data: sorted.map(r => r.total_tokens_per_second_mean), borderColor: CHART_COLORS.accent3, backgroundColor: CHART_COLORS.accent3, tension: 0.15 },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Concurrency (configured streams)' } },
          y: { title: { display: true, text: 'Tokens / second' } },
        },
      },
    };
  })());

  const runsWithResources = runs.filter(r => r.resource_usage && r.resource_usage.length);
  const hwSelect = document.getElementById('t1HwRunSelect');
  if (hwSelect) {
    hwSelect.innerHTML = buildRunOptions(runsWithResources, r => r.concurrency ?? '?');
    const draw = () => {
      const r = runsWithResources[Number(hwSelect.value)];
      const devices = [...new Set(r.resource_usage.map(d => d.device))];
      const colors = [CHART_COLORS.accent, CHART_COLORS.accent2, CHART_COLORS.accent3, CHART_COLORS.accent4];
      const datasets = devices.map((dev, i) => ({
        label: `${dev} util %`,
        data: r.resource_usage.filter(d => d.device === dev).map(d => ({ x: d.elapsed_s, y: d.util_percent })),
        borderColor: colors[i % colors.length], backgroundColor: colors[i % colors.length],
        pointRadius: 0, tension: 0.1,
      }));
      makeChart('t1HwChart', {
        type: 'line',
        data: { datasets },
        options: {
          responsive: true,
          parsing: false,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Elapsed time (s)' } },
            y: { title: { display: true, text: 'Utilization %' }, min: 0, max: 100 },
          },
        },
      });
    };
    hwSelect.onchange = () => { destroyChartById('t1HwChart'); draw(); };
    draw();
  }
}

// Builds <option> labels for a "concurrent users" run selector. Plain
// labelFn(run) output ("4", "2") would be ambiguous when two separate runs
// share the same concurrency level (e.g. a repeated/validation run) - each
// one in that selector is otherwise a silent duplicate the user has no way
// to tell apart, so duplicates get "(run 1 of 2)" etc appended.
function buildRunOptions(runs, labelFn) {
  const labels = runs.map(labelFn);
  const counts = {};
  labels.forEach(l => { counts[l] = (counts[l] || 0) + 1; });
  const seen = {};
  return runs.map((r, i) => {
    const label = labels[i];
    let text = String(label);
    if (counts[label] > 1) {
      seen[label] = (seen[label] || 0) + 1;
      text = `${label} (run ${seen[label]} of ${counts[label]})`;
    }
    return `<option value="${i}">${text}</option>`;
  }).join('');
}

function destroyChartById(canvasId) {
  const idx = activeCharts.findIndex(c => c.canvas && c.canvas.id === canvasId);
  if (idx >= 0) activeCharts.splice(idx, 1)[0].destroy();
}

// ---------- Test 2: Chatbot ----------

function meanTurnField(run, field) {
  const vals = [];
  for (const u of run.users) {
    for (const t of u.turns) {
      if (!t.error && t[field] != null) vals.push(t[field]);
    }
  }
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

function renderTest2(allRuns) {
  if (!allRuns.length) return '<p class="empty">No Test 2 data for this model.</p>';
  const { modes, html: servingTabs } = servingModeTabsHtml(allRuns);
  const runs = filterByServingMode(allRuns, modes);
  if (!runs.length) return servingTabs + '<p class="empty">No Test 2 data for this serving mode.</p>';

  const maxUsersRun = [...runs].filter(r => r.concurrent_users != null).sort((a, b) => b.concurrent_users - a.concurrent_users)[0];
  let html = servingTabs;
  if (maxUsersRun) {
    html += overviewGrid([
      overviewTile('Mean end-to-end latency', fmt(meanTurnField(maxUsersRun, 'total_latency_s'), 2), `s &middot; ${maxUsersRun.concurrent_users} users`, true),
      overviewTile('Mean TTFT', fmt(meanTurnField(maxUsersRun, 'ttft_s'), 2), 's'),
    ]);
  }
  html += '<div class="test-note">Simulated concurrent users (general, code, summary, knowledge), each running a conversation of 1+ turns, against the inference server directly.</div>';
  html += barChart(runs, r => `${r.concurrent_users ?? '?'}u`, r => r.overall_duration_s, {
    title: 'Total duration by concurrent users', subtitle: 'Simulated chatbot workload', unit: 's', caption: 'Values represent total wall-clock duration for the run.',
  });
  html += '<table><thead><tr><th>Target</th><th>Parallelism setting</th><th>Users</th><th>Turns/conv.</th><th>Total duration (s)</th><th>Error rate</th></tr></thead><tbody>';
  for (const r of runs) {
    html += `<tr><td>${r.target ?? '-'}</td><td>${parallelismLabel(r)}</td><td>${r.concurrent_users ?? '-'}</td>
      <td>${r.turns_per_conversation ?? 1}</td><td>${fmt(r.overall_duration_s)}</td><td>${pct(r.error_rate)}</td></tr>`;
  }
  html += '</tbody></table>';
  if (runs.some(r => r.error_rate)) {
    html += '<div class="chart-note">A non-zero error rate at high concurrency can partly reflect HTTP timeout/connection-pool limits in the test client itself under heavy load, not only the model/server failing - see per-run error detail before treating it as a hard capacity ceiling.</div>';
  }

  html += '<div class="run-select">Concurrent users: <select id="t2RunSelect"></select><span class="chart-note" id="t2TurnsNote"></span></div>';
  html += '<h3><span class="step-num">1</span>End-to-end latency per turn</h3>';
  html += '<div class="chart-card"><canvas id="t2LatencyChart"></canvas></div>';
  html += '<h3><span class="step-num">2</span>Context growth vs conversation turns</h3>';
  html += '<div class="chart-card"><canvas id="t2ContextChart"></canvas></div>';
  html += '<h3><span class="step-num">3</span>Latency distribution (all turns in this run)</h3>';
  html += '<div class="chart-card"><canvas id="t2DistChart"></canvas><div class="stat-row" id="t2DistStats"></div></div>';

  return html;
}

function parallelismLabel(r) {
  return r.ollama_num_parallel ? `OLLAMA_NUM_PARALLEL=${r.ollama_num_parallel}` : 'n/a (not Ollama, or unset)';
}

function wireTest2Charts(allRuns) {
  wireServingModeTabs();
  const modes = SERVING_MODE_ORDER.filter(m => allRuns.some(r => r.serving_mode === m));
  const runs = filterByServingMode(allRuns, modes);
  if (!runs.length) return;

  const select = document.getElementById('t2RunSelect');
  if (!select) return;
  select.innerHTML = buildRunOptions(runs, r => r.concurrent_users ?? '?');
  const turnsNote = document.getElementById('t2TurnsNote');

  const draw = () => {
    const r = runs[Number(select.value)];
    if (turnsNote) turnsNote.textContent = `Each user conversation has ${r.turns_per_conversation ?? 1} iterations.`;
    const maxTurns = Math.max(1, ...r.users.map(u => u.turns.length));

    const perTurn = [];
    for (let t = 1; t <= maxTurns; t++) {
      const vals = r.users.map(u => u.turns.find(x => x.turn_id === t)).filter(Boolean).filter(x => !x.error);
      const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
      perTurn.push({
        turn: t,
        ttft: mean(vals.map(v => v.ttft_s).filter(v => v != null)),
        generation: mean(vals.map(v => v.generation_duration_s).filter(v => v != null)),
        total: mean(vals.map(v => v.total_latency_s).filter(v => v != null)),
        userInput: mean(vals.map(v => v.user_input_tokens_approx).filter(v => v != null)),
        context: mean(vals.map(v => v.context_tokens).filter(v => v != null)),
        cumOutput: mean(vals.map(v => v.cumulative_output_tokens).filter(v => v != null)),
      });
    }

    destroyChartById('t2LatencyChart');
    makeChart('t2LatencyChart', {
      type: 'line',
      data: {
        labels: perTurn.map(p => p.turn),
        datasets: [
          { label: 'TTFT (s)', data: perTurn.map(p => p.ttft), borderColor: CHART_COLORS.accent, backgroundColor: CHART_COLORS.accent, tension: 0.15 },
          { label: 'Generation time (s)', data: perTurn.map(p => p.generation), borderColor: CHART_COLORS.accent2, backgroundColor: CHART_COLORS.accent2, tension: 0.15 },
          { label: 'End-to-end latency (s)', data: perTurn.map(p => p.total), borderColor: CHART_COLORS.accent3, backgroundColor: CHART_COLORS.accent3, tension: 0.15 },
        ],
      },
      options: { responsive: true, scales: { x: { title: { display: true, text: 'Conversation turn' } }, y: { title: { display: true, text: 'Seconds (mean across users)' } } } },
    });

    destroyChartById('t2ContextChart');
    const contextDatasets = [
      { label: 'User input tokens', data: perTurn.map(p => p.userInput), borderColor: CHART_COLORS.accent, backgroundColor: CHART_COLORS.accent, tension: 0.15 },
      { label: 'Context tokens sent (full history)', data: perTurn.map(p => p.context), borderColor: CHART_COLORS.accent2, backgroundColor: CHART_COLORS.accent2, tension: 0.15 },
      { label: 'Cumulative output tokens', data: perTurn.map(p => p.cumOutput), borderColor: CHART_COLORS.accent3, backgroundColor: CHART_COLORS.accent3, tension: 0.15 },
    ];
    if (r.max_context_tokens) {
      contextDatasets.push({ label: `Context window limit (${r.max_context_tokens})`, data: perTurn.map(() => r.max_context_tokens), borderColor: CHART_COLORS.muted, borderDash: [6, 4], pointRadius: 0 });
    }
    makeChart('t2ContextChart', {
      type: 'line',
      data: { labels: perTurn.map(p => p.turn), datasets: contextDatasets },
      options: { responsive: true, scales: { x: { title: { display: true, text: 'Conversation turn' } }, y: { title: { display: true, text: 'Tokens (mean across users)' } } } },
    });

    const allLatencies = r.users.flatMap(u => u.turns.filter(t => !t.error && t.total_latency_s != null).map(t => t.total_latency_s)).sort((a, b) => a - b);
    const binCount = Math.min(12, Math.max(4, Math.ceil(Math.sqrt(allLatencies.length))));
    let bins = [], binLabels = [];
    if (allLatencies.length) {
      const min = allLatencies[0], max = allLatencies[allLatencies.length - 1];
      const width = (max - min) / binCount || 1;
      bins = new Array(binCount).fill(0);
      for (const v of allLatencies) {
        const idx = Math.min(binCount - 1, Math.floor((v - min) / width));
        bins[idx]++;
      }
      binLabels = bins.map((_, i) => `${(min + i * width).toFixed(1)}-${(min + (i + 1) * width).toFixed(1)}s`);
    }
    destroyChartById('t2DistChart');
    makeChart('t2DistChart', {
      type: 'bar',
      data: { labels: binLabels, datasets: [{ label: 'Turns', data: bins, backgroundColor: CHART_COLORS.accent }] },
      options: { responsive: true, scales: { x: { title: { display: true, text: 'Latency bucket' } }, y: { title: { display: true, text: 'Count' }, beginAtZero: true } } },
    });
    const statsEl = document.getElementById('t2DistStats');
    if (statsEl) {
      statsEl.innerHTML = allLatencies.length
        ? `<span>P50: <b>${fmt(percentile(allLatencies, 0.50))}s</b></span><span>P95: <b>${fmt(percentile(allLatencies, 0.95))}s</b></span><span>P99: <b>${fmt(percentile(allLatencies, 0.99))}s</b></span><span>n=${allLatencies.length} turns</span>`
        : '';
    }
  };
  select.onchange = draw;
  draw();
}

// ---------- Test 3: RAG ----------

function renderTest3(allRuns) {
  if (!allRuns.length) return '<p class="empty">No Test 3 data for this model.</p>';
  const { modes, html: servingTabs } = servingModeTabsHtml(allRuns);
  const runs = filterByServingMode(allRuns, modes);
  if (!runs.length) return servingTabs + '<p class="empty">No Test 3 data for this serving mode.</p>';

  const largestRun = [...runs].filter(r => r.num_queries != null).sort((a, b) => b.num_queries - a.num_queries)[0];
  let html = servingTabs;
  if (largestRun) {
    html += overviewGrid([
      overviewTile('Mean generation latency', fmt(largestRun.avg_generation_s, 2), `s &middot; q${largestRun.num_queries} c${largestRun.concurrency}`, true),
      overviewTile('Mean retrieval latency', fmt(largestRun.avg_retrieval_s, 3), 's (embedding + vector search)'),
    ]);
  }
  html += '<div class="test-note">Retrieval (Qdrant + nomic-embed-text) + generation, over the Kubernetes docs corpus. No reranking or keyword/hybrid search stage exists in this pipeline (vector-only) - charts below mark those as n/a rather than fabricating a value.</div>';
  html += barChart(runs, r => `q${r.num_queries}/c${r.concurrency}`, r => r.wall_duration_s, {
    title: 'Wall-clock time by scale', subtitle: 'RAG pipeline &middot; retrieval + generation', unit: 's', caption: 'Values represent total wall-clock duration (q = queries, c = concurrency).',
  });
  html += '<table><thead><tr><th>Queries</th><th>Concurrency</th><th>Top-K</th><th>Wall-clock (s)</th><th>Mean retrieval (s)</th><th>Mean generation (s)</th><th>Tokens in/out</th><th>Avg similarity</th><th>Error rate</th><th>Hit rate top-k</th></tr></thead><tbody>';
  for (const r of runs) {
    html += `<tr><td>${r.num_queries ?? '-'}</td><td>${r.concurrency ?? '-'}</td><td>${r.top_k ?? '-'}</td><td>${fmt(r.wall_duration_s)}</td>
      <td>${fmt(r.avg_retrieval_s, 3)}</td><td>${fmt(r.avg_generation_s)}</td>
      <td>${fmt(r.avg_input_tokens, 0)} / ${fmt(r.avg_output_tokens, 0)}</td>
      <td>${fmt(r.avg_similarity_score, 3)}</td>
      <td>${pct(r.error_rate)}</td>
      <td>${pct(r.expected_chunk_hit_rate)}</td></tr>`;
  }
  html += '</tbody></table>';

  html += '<h3><span class="step-num">1</span>RAG pipeline latency breakdown</h3>';
  html += '<div class="chart-card"><canvas id="t3BreakdownChart"></canvas><div class="chart-note">Reranking: not implemented in this pipeline (n/a, not estimated).</div></div>';

  html += '<h3><span class="step-num">2</span>Token distribution per RAG request</h3>';
  html += '<div class="run-select">Concurrent users: <select id="t3RunSelect"></select></div>';
  html += '<div class="chart-card"><canvas id="t3TokensChart"></canvas></div>';

  return html;
}

function wireTest3Charts(allRuns) {
  wireServingModeTabs();
  const modes = SERVING_MODE_ORDER.filter(m => allRuns.some(r => r.serving_mode === m));
  const runs = filterByServingMode(allRuns, modes);
  if (!runs.length) return;

  const sortedByScale = [...runs].sort((a, b) => (a.num_queries ?? 0) - (b.num_queries ?? 0));
  makeChart('t3BreakdownChart', {
    type: 'bar',
    data: {
      labels: sortedByScale.map(r => `q${r.num_queries} c${r.concurrency}`),
      datasets: [
        { label: 'Embedding (s)', data: sortedByScale.map(r => r.avg_embedding_s), backgroundColor: CHART_COLORS.accent, stack: 's' },
        { label: 'Vector search (s)', data: sortedByScale.map(r => r.avg_vector_search_s), backgroundColor: CHART_COLORS.accent2, stack: 's' },
        { label: 'Generation (s)', data: sortedByScale.map(r => r.avg_generation_s), backgroundColor: CHART_COLORS.accent3, stack: 's' },
      ],
    },
    options: { responsive: true, scales: { x: { stacked: true, title: { display: true, text: 'Run (scale)' } }, y: { stacked: true, title: { display: true, text: 'Avg time (s)' } } } },
  });

  const select = document.getElementById('t3RunSelect');
  if (select) {
    select.innerHTML = buildRunOptions(runs, r => r.concurrency ?? '?');
    const draw = () => {
      const r = runs[Number(select.value)];
      destroyChartById('t3TokensChart');
      makeChart('t3TokensChart', {
        type: 'bar',
        data: {
          labels: r.per_request.map(q => q.index),
          datasets: [
            { label: 'System prompt tokens (approx)', data: r.per_request.map(q => q.system_prompt_tokens_approx), backgroundColor: CHART_COLORS.accent, stack: 's' },
            { label: 'User query tokens (approx)', data: r.per_request.map(q => q.user_query_tokens_approx), backgroundColor: CHART_COLORS.accent2, stack: 's' },
            { label: 'Retrieved context tokens (approx)', data: r.per_request.map(q => q.context_tokens_approx), backgroundColor: CHART_COLORS.accent4, stack: 's' },
            { label: 'Output tokens (exact)', data: r.per_request.map(q => q.output_tokens), backgroundColor: CHART_COLORS.accent3, stack: 's' },
          ],
        },
        options: { responsive: true, scales: { x: { stacked: true, title: { display: true, text: 'Request index' } }, y: { stacked: true, title: { display: true, text: 'Tokens' } } } },
      });
    };
    select.onchange = draw;
    draw();
  }
}

const TEST_META = {
  test1: { label: 'Test 1', title: 'Hardware / Serving', heading: 'Test 1 : Raw inference performance', render: renderTest1, wire: wireTest1Charts },
  test2: { label: 'Test 2', title: 'Chatbot', heading: 'Test 2 : Concurrent chatbot', render: renderTest2, wire: wireTest2Charts },
  test3: { label: 'Test 3', title: 'RAG', heading: 'Test 3 : RAG', render: renderTest3, wire: wireTest3Charts },
};

let currentPhase = null;
let currentModel = null;
let currentTest = 'test1';
let currentServingMode = null; // sub-tab shared by Test 1/2/3: 1 GPU vs tensor-parallel vs data-parallel

function renderTestPanel() {
  const d = DATA[currentPhase][currentModel];
  destroyCharts();
  const meta = TEST_META[currentTest];
  const runs = d[currentTest];
  const panel = document.getElementById('testPanel');
  panel.setAttribute('data-test', currentTest);
  panel.innerHTML = `<section><h2><span class="test-badge"></span>${meta.heading}</h2>${meta.render(runs)}</section>`;
  if (runs.length) meta.wire(runs);
}

function renderTestTabs() {
  const bar = document.getElementById('testTabs');
  bar.innerHTML = '';
  for (const key of Object.keys(TEST_META)) {
    const meta = TEST_META[key];
    const btn = document.createElement('button');
    btn.className = 'test-tab';
    btn.setAttribute('data-test', key);
    btn.setAttribute('aria-selected', key === currentTest ? 'true' : 'false');
    btn.innerHTML = `<span class="test-tab-label">${meta.label}</span><span class="test-tab-title">${meta.title}</span>`;
    btn.onclick = () => {
      currentTest = key;
      bar.querySelectorAll('.test-tab').forEach(b => b.setAttribute('aria-selected', String(b.getAttribute('data-test') === key)));
      renderTestPanel();
    };
    bar.appendChild(btn);
  }
}

function renderModel(model) {
  currentModel = model;
  currentServingMode = null; // don't carry e.g. "2 GPU" over to a model/phase that has no such split
  renderTestTabs();
  renderTestPanel();
}

function renderModelTabs() {
  const models = Object.keys(DATA[currentPhase]);
  const tabs = document.getElementById('modelTabs');
  tabs.innerHTML = '';
  models.forEach((model, i) => {
    const btn = document.createElement('button');
    btn.className = 'model-tab';
    btn.textContent = model;
    btn.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
    btn.onclick = () => {
      tabs.querySelectorAll('.model-tab').forEach(b => b.setAttribute('aria-selected', 'false'));
      btn.setAttribute('aria-selected', 'true');
      renderModel(model);
    };
    tabs.appendChild(btn);
  });
  if (models.length) renderModel(models[0]);
}

function renderPhase(phase) {
  currentPhase = phase;
  renderModelTabs();
}

function init() {
  const phases = Object.keys(DATA);
  const select = document.getElementById('phaseSelect');
  select.innerHTML = phases.map(phase => `<option value="${phase}">${PHASE_LABELS[phase] || phase}</option>`).join('');
  select.onchange = () => renderPhase(select.value);
  if (phases.length) renderPhase(phases[0]);
}

// Chart.js is loaded from an external <script src> - don't assume it has
// finished loading by the time this inline script runs (seen in practice:
// on a slow connection the external script can still be in flight here,
// leaving `Chart` undefined and every chart silently blank).
if (window.Chart) {
  init();
} else {
  window.addEventListener('load', init);
}
</script>
</body>
</html>
"""


def main():
    test1 = collect_test1_runs()
    test2 = collect_test2_runs()
    test3 = collect_test3_runs()
    grouped = group_by_model(test1, test2, test3)

    DASHBOARD_DIR.mkdir(exist_ok=True)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(grouped))
    html = html.replace("__PHASE_LABELS_JSON__", json.dumps(PHASE_LABELS))
    output_path = DASHBOARD_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"==> Dashboard built: {output_path}")
    for phase, models in grouped.items():
        print(f"    {PHASE_LABELS.get(phase, phase)}:")
        for model, d in models.items():
            print(f"      {model}: test1={len(d['test1'])} test2={len(d['test2'])} test3={len(d['test3'])} runs")


if __name__ == "__main__":
    main()
