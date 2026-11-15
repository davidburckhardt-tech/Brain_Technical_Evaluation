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
            runs.append({
                "run_dir": run_dir.name,
                "model": model,
                "strategy": strategy.get("type_"),
                "concurrency": concurrency,
                "gpu_count": infer_gpu_count(resource_usage),
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
            "users": normalized_users,
            "resource_usage": read_resource_usage(resource_csv),
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
        runs.append({
            "run_file": path.name,
            "model": data.get("model", "unknown"),
            "num_queries": data.get("num_queries"),
            "concurrency": data.get("concurrency"),
            "top_k": data.get("top_k"),
            "wall_duration_s": data.get("wall_duration_s"),
            "error_rate": data.get("error_rate"),
            "avg_embedding_s": avg("embedding_duration_s"),
            "avg_vector_search_s": avg("vector_search_duration_s"),
            "avg_retrieval_s": avg("retrieval_duration_s"),
            "avg_generation_s": avg("generation_duration_s"),
            "avg_input_tokens": avg("input_tokens"),
            "avg_output_tokens": avg("output_tokens"),
            "avg_similarity_score": avg("avg_similarity_score"),
            "expected_chunk_hit_rate": hit_rate,
            "per_request": per_request,
            "resource_usage": read_resource_usage(resource_csv),
        })
    return runs


def group_by_model(test1, test2, test3) -> dict:
    models = set(r["model"] for r in test1) | set(r["model"] for r in test2) | set(r["model"] for r in test3)
    return {
        model: {
            "test1": [r for r in test1 if r["model"] == model],
            "test2": [r for r in test2 if r["model"] == model],
            "test3": [r for r in test3 if r["model"] == model],
        }
        for model in sorted(models)
    }


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
  .model-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .model-tab {
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 999px;
    cursor: pointer; background: transparent; color: var(--fg); font-size: 13px;
  }
  .model-tab[aria-selected="true"] { background: var(--accent); color: white; border-color: var(--accent); }

  /* Test-level tabs: one big, unmistakably color-coded control per test, so
     switching between Test 1/2/3 reads as switching sections, not just
     changing a filter - each test keeps its color everywhere below (the
     active-tab fill, the panel's top border and badge, the section note). */
  .test-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
  .test-tab {
    text-align: left; cursor: pointer; border: 2px solid var(--border); border-radius: 12px;
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

  .test-panel { border-top: 4px solid var(--border); border-radius: 0 0 12px 12px; padding: 20px 0 4px; }
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
    padding: 6px 16px; border-radius: 999px; border: 1.5px solid var(--border);
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
  .bars { display: flex; flex-direction: column; gap: 6px; margin-bottom: 20px; }
  .bar-row { display: flex; align-items: center; gap: 8px; }
  .bar-label { width: 150px; flex-shrink: 0; font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; background: var(--bar-bg); border-radius: 4px; height: 18px; position: relative; min-width: 60px; }
  .bar-fill { background: var(--accent); height: 100%; border-radius: 4px; }
  .bar-value { position: absolute; left: 8px; top: 0; height: 100%; display: flex; align-items: center; font-size: 11px; color: var(--fg); white-space: nowrap; }
  .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
  .chart-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 20px; }
  .chart-card canvas { max-height: 320px; }
  .run-select { margin-bottom: 10px; font-size: 12.5px; }
  select {
    background: var(--bg); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font-size: 12.5px; max-width: 100%;
  }
  .stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }
  .stat-row b { color: var(--fg); }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Brain Demo &mdash; Benchmark Results</h1>
  <p class="subtitle">Generated by scripts/build_dashboard.py</p>

  <div class="model-tabs" id="modelTabs"></div>

  <div class="test-tabs" id="testTabs"></div>
  <div class="test-panel" id="testPanel"></div>
</main>

<script>
const DATA = __DATA_JSON__;
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

function barChart(rows, labelFn, valueFn, unit) {
  const maxVal = Math.max(...rows.map(valueFn).filter(v => v !== null && v !== undefined), 0.001);
  let html = '<div class="bars">';
  for (const row of rows) {
    const v = valueFn(row);
    const pctw = v === null || v === undefined ? 0 : Math.max((v / maxVal) * 100, 2);
    html += `<div class="bar-row">
      <div class="bar-label">${labelFn(row)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pctw}%"></div>
        <div class="bar-value">${fmt(v)}${unit}</div></div>
    </div>`;
  }
  return html + '</div>';
}

// ---------- Test 1: Hardware / Model Serving ----------

function gpuTabsHtml(allRuns) {
  const gpuCounts = [...new Set(allRuns.map(r => r.gpu_count).filter(v => v != null))].sort((a, b) => a - b);
  if (gpuCounts.length === 0) return { gpuCounts, html: '' };
  if (currentGpuMode == null || !gpuCounts.includes(currentGpuMode)) currentGpuMode = gpuCounts[0];
  if (gpuCounts.length < 2) return { gpuCounts, html: '' };
  const html = '<div class="gpu-tabs">' + gpuCounts.map(c =>
    `<button class="gpu-tab" data-gpu="${c}" aria-selected="${c === currentGpuMode}">${c} GPU${c > 1 ? 's' : ''} (tensor-parallel=${c})</button>`
  ).join('') + '</div>';
  return { gpuCounts, html };
}

function renderTest1(allRuns) {
  if (!allRuns.length) return '<p class="empty">No Test 1 data for this model.</p>';
  const { gpuCounts, html: gpuTabs } = gpuTabsHtml(allRuns);
  const runs = gpuCounts.length ? allRuns.filter(r => r.gpu_count === currentGpuMode) : allRuns;
  if (!runs.length) return gpuTabs + '<p class="empty">No Test 1 data for this GPU mode.</p>';
  let html = gpuTabs + '<div class="test-note">Synthetic load, GuideLLM. "Concurrency" = fixed configured streams.</div>';
  html += barChart(runs, r => `${r.strategy}${r.concurrency ? ' @' + r.concurrency : ''}`, r => r.ttft_ms_mean, ' ms TTFT');
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
    html += '<div class="run-select">Run (concurrency level): <select id="t1RunSelect"></select></div>';
    html += '<div class="chart-card"><canvas id="t1LatencyChart"></canvas></div>';
  } else {
    html += '<p class="empty">No per-request detail available for these runs (requires requests_detail.csv).</p>';
  }

  html += '<h3><span class="step-num">2</span>Throughput vs concurrency</h3>';
  html += '<div class="chart-card"><canvas id="t1ThroughputChart"></canvas><div class="chart-note" id="t1SaturationNote"></div></div>';

  html += '<h3><span class="step-num">3</span>Hardware utilization vs time</h3>';
  const runsWithResources = runs.filter(r => r.resource_usage && r.resource_usage.length);
  if (runsWithResources.length) {
    html += '<div class="run-select">Run: <select id="t1HwRunSelect"></select></div>';
    html += '<div class="chart-card"><canvas id="t1HwChart"></canvas></div>';
  } else {
    html += '<p class="empty">No resource_usage.csv found for these runs.</p>';
  }

  return html;
}

function wireTest1Charts(allRuns) {
  document.querySelectorAll('#testPanel .gpu-tab').forEach(btn => {
    btn.onclick = () => {
      currentGpuMode = Number(btn.getAttribute('data-gpu'));
      renderTestPanel();
    };
  });
  const gpuCounts = [...new Set(allRuns.map(r => r.gpu_count).filter(v => v != null))];
  const runs = gpuCounts.length ? allRuns.filter(r => r.gpu_count === currentGpuMode) : allRuns;
  if (!runs.length) return;

  const runsWithDetail = runs.filter(r => r.requests_detail && r.requests_detail.length);
  const runSelect = document.getElementById('t1RunSelect');
  if (runSelect) {
    runSelect.innerHTML = runsWithDetail.map((r, i) =>
      `<option value="${i}">${r.strategy} @${r.concurrency ?? '?'} (${r.run_dir})</option>`).join('');
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
    hwSelect.innerHTML = runsWithResources.map((r, i) =>
      `<option value="${i}">${r.strategy} @${r.concurrency ?? '?'} (${r.run_dir})</option>`).join('');
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

function destroyChartById(canvasId) {
  const idx = activeCharts.findIndex(c => c.canvas && c.canvas.id === canvasId);
  if (idx >= 0) activeCharts.splice(idx, 1)[0].destroy();
}

// ---------- Test 2: Chatbot ----------

function renderTest2(runs) {
  if (!runs.length) return '<p class="empty">No Test 2 data for this model.</p>';
  let html = '<div class="test-note">Simulated concurrent users (general, code, summary, knowledge), each running a conversation of 1+ turns, against the inference server directly.</div>';
  html += barChart(runs, r => parallelismLabel(r), r => r.overall_duration_s, ' s total');
  html += '<table><thead><tr><th>Target</th><th>Parallelism setting</th><th>Users</th><th>Turns/conv.</th><th>Total duration (s)</th><th>Error rate</th></tr></thead><tbody>';
  for (const r of runs) {
    html += `<tr><td>${r.target ?? '-'}</td><td>${parallelismLabel(r)}</td><td>${r.concurrent_users ?? '-'}</td>
      <td>${r.turns_per_conversation ?? 1}</td><td>${fmt(r.overall_duration_s)}</td><td>${pct(r.error_rate)}</td></tr>`;
  }
  html += '</tbody></table>';

  html += '<div class="run-select">Run: <select id="t2RunSelect"></select></div>';
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

function wireTest2Charts(runs) {
  const select = document.getElementById('t2RunSelect');
  if (!select) return;
  select.innerHTML = runs.map((r, i) => `<option value="${i}">${r.run_file} (${r.concurrent_users} users, ${r.turns_per_conversation ?? 1} turns)</option>`).join('');

  const draw = () => {
    const r = runs[Number(select.value)];
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

function renderTest3(runs) {
  if (!runs.length) return '<p class="empty">No Test 3 data for this model.</p>';
  let html = '<div class="test-note">Retrieval (Qdrant + nomic-embed-text) + generation, over the Kubernetes docs corpus. No reranking or keyword/hybrid search stage exists in this pipeline (vector-only) - charts below mark those as n/a rather than fabricating a value.</div>';
  html += barChart(runs, r => `q${r.num_queries} c${r.concurrency}`, r => r.wall_duration_s, ' s wall-clock');
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

  html += '<h3><span class="step-num">2</span>Retrieval performance vs top-K</h3>';
  html += '<div class="chart-card"><canvas id="t3TopkChart"></canvas><div class="chart-note">Keyword and hybrid search: not implemented in this pipeline (vector-only) - shown as n/a.</div></div>';

  html += '<h3><span class="step-num">3</span>Token distribution per RAG request</h3>';
  html += '<div class="run-select">Run: <select id="t3RunSelect"></select></div>';
  html += '<div class="chart-card"><canvas id="t3TokensChart"></canvas></div>';

  return html;
}

function wireTest3Charts(runs) {
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

  const byTopK = {};
  for (const r of runs) {
    const k = r.top_k ?? '?';
    if (!byTopK[k]) byTopK[k] = [];
    if (r.avg_retrieval_s != null) byTopK[k].push(r.avg_retrieval_s);
  }
  const topKs = Object.keys(byTopK).sort((a, b) => Number(a) - Number(b));
  makeChart('t3TopkChart', {
    type: 'bar',
    data: {
      labels: topKs.map(k => `top-${k}`),
      datasets: [
        { label: 'Vector search (retrieval, s)', data: topKs.map(k => { const v = byTopK[k]; return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; }), backgroundColor: CHART_COLORS.accent },
        { label: 'Keyword search (n/a)', data: topKs.map(() => null), backgroundColor: CHART_COLORS.muted },
        { label: 'Hybrid search (n/a)', data: topKs.map(() => null), backgroundColor: CHART_COLORS.border },
      ],
    },
    options: { responsive: true, scales: { x: { title: { display: true, text: 'Top-K' } }, y: { title: { display: true, text: 'Avg retrieval latency (s)' } } } },
  });

  const select = document.getElementById('t3RunSelect');
  if (select) {
    select.innerHTML = runs.map((r, i) => `<option value="${i}">${r.run_file}</option>`).join('');
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
  test1: { label: 'Test 1', title: 'Hardware / Serving', heading: 'Test 1 — Raw inference performance', render: renderTest1, wire: wireTest1Charts },
  test2: { label: 'Test 2', title: 'Chatbot', heading: 'Test 2 — Concurrent chatbot', render: renderTest2, wire: wireTest2Charts },
  test3: { label: 'Test 3', title: 'RAG', heading: 'Test 3 — RAG', render: renderTest3, wire: wireTest3Charts },
};

let currentModel = null;
let currentTest = 'test1';
let currentGpuMode = null; // Test 1 sub-tab: 1 GPU vs 2 GPU (tensor-parallel)

function renderTestPanel() {
  const d = DATA[currentModel];
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
  renderTestTabs();
  renderTestPanel();
}

function init() {
  const models = Object.keys(DATA);
  const tabs = document.getElementById('modelTabs');
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
    output_path = DASHBOARD_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"==> Dashboard built: {output_path}")
    print(f"    Models: {', '.join(grouped.keys())}")
    for model, d in grouped.items():
        print(f"    {model}: test1={len(d['test1'])} test2={len(d['test2'])} test3={len(d['test3'])} runs")


if __name__ == "__main__":
    main()
