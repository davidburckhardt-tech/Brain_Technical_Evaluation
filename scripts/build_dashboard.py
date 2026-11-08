#!/usr/bin/env python3
"""Scans results/ and builds dashboard/index.html - a small, self-contained,
offline HTML page (data embedded inline, no server, no external libraries)
grouping every run by model, so it stays useful once more models (Mistral,
GLM, Qwen, ...) are tested.

Re-run this any time new results land under results/.
"""

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


def collect_test1_runs() -> list[dict]:
    runs = []
    for benchmarks_json in sorted(RESULTS_DIR.glob("*/*/benchmarks.json")):
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

            runs.append({
                "run_dir": benchmarks_json.parent.name,
                "model": model,
                "strategy": strategy.get("type_"),
                "concurrency": strategy.get("max_concurrency") or strategy.get("worker_count"),
                "duration_s": benchmark.get("duration"),
                "ttft_ms_mean": stat("time_to_first_token_ms"),
                "tpot_ms_mean": stat("time_per_output_token_ms"),
                "itl_ms_mean": stat("inter_token_latency_ms"),
                "request_latency_s_mean": stat("request_latency"),
                "output_tokens_per_second_mean": stat("output_tokens_per_second"),
                "error_rate": (
                    metrics.get("request_totals", {}).get("errored", 0) / metrics["request_totals"]["total"]
                    if metrics.get("request_totals", {}).get("total")
                    else None
                ),
            })
    return runs


def collect_test2_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*/*_chatbot*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        num_parallel = data.get("ollama_num_parallel") or KNOWN_NUM_PARALLEL_BY_FILENAME.get(path.name)
        users = data.get("users", [])
        runs.append({
            "run_file": path.name,
            "model": data.get("model", "unknown"),
            "ollama_num_parallel": num_parallel,
            "concurrent_users": data.get("concurrent_users"),
            "overall_duration_s": data.get("overall_duration_s"),
            "error_rate": data.get("error_rate"),
            "users": [
                {
                    "user_id": u["user_id"],
                    "category": u["category"],
                    "ttft_s": u["ttft_s"],
                    "total_latency_s": u["total_latency_s"],
                    "output_tokens_per_second_approx": u["output_tokens_per_second_approx"],
                }
                for u in users
            ],
        })
    return runs


def collect_test3_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*/*_rag_q*_c*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = data.get("queries", [])
        succeeded = [q for q in queries if not q.get("error")]
        avg_retrieval = sum(q["retrieval_duration_s"] for q in succeeded) / len(succeeded) if succeeded else None
        avg_generation = sum(q["generation_duration_s"] for q in succeeded) / len(succeeded) if succeeded else None
        avg_input_tokens = (
            sum(q.get("input_tokens") or 0 for q in succeeded) / len(succeeded) if succeeded else None
        )
        avg_output_tokens = (
            sum(q.get("output_tokens") or 0 for q in succeeded) / len(succeeded) if succeeded else None
        )
        hit_rate = (
            sum(1 for q in succeeded if q["expected_chunk_found_in_top_k"]) / len(succeeded) if succeeded else None
        )
        runs.append({
            "run_file": path.name,
            "model": data.get("model", "unknown"),
            "num_queries": data.get("num_queries"),
            "concurrency": data.get("concurrency"),
            "wall_duration_s": data.get("wall_duration_s"),
            "error_rate": data.get("error_rate"),
            "avg_retrieval_s": avg_retrieval,
            "avg_generation_s": avg_generation,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "expected_chunk_hit_rate": hit_rate,
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
<title>Brain Demo - Resultados</title>
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
    --bar-bg: #eef2fb;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #15181d;
      --fg: #e8e8e8;
      --muted: #9aa0a6;
      --border: #30343a;
      --accent: #6d9bff;
      --bar-bg: #202636;
    }
  }
  :root[data-theme="dark"] {
    --bg: #15181d;
    --fg: #e8e8e8;
    --muted: #9aa0a6;
    --border: #30343a;
    --accent: #6d9bff;
    --bar-bg: #202636;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 24px 16px 48px;
  }
  main { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .subtitle { color: var(--muted); margin-bottom: 28px; font-size: 13px; }
  .model-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
  .model-tab {
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 999px;
    cursor: pointer; background: transparent; color: var(--fg); font-size: 13px;
  }
  .model-tab[aria-selected="true"] { background: var(--accent); color: white; border-color: var(--accent); }
  section { margin-bottom: 36px; }
  h2 { font-size: 16px; margin: 0 0 4px; }
  .test-note { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
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
  [hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Brain Demo &mdash; Resultados de benchmark</h1>
  <p class="subtitle">Generado por scripts/build_dashboard.py &middot; datos de Fase 0 (Mac, no representativos del hardware final)</p>

  <div class="model-tabs" id="modelTabs"></div>

  <div id="modelContent"></div>
</main>

<script>
const DATA = __DATA_JSON__;

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return '&ndash;';
  return Number(n).toFixed(digits);
}

function barChart(rows, labelFn, valueFn, unit) {
  const maxVal = Math.max(...rows.map(valueFn).filter(v => v !== null && v !== undefined), 0.001);
  let html = '<div class="bars">';
  for (const row of rows) {
    const v = valueFn(row);
    const pct = v === null || v === undefined ? 0 : Math.max((v / maxVal) * 100, 2);
    html += `<div class="bar-row">
      <div class="bar-label">${labelFn(row)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div>
        <div class="bar-value">${fmt(v)}${unit}</div></div>
    </div>`;
  }
  return html + '</div>';
}

function pct(v) {
  return (v === null || v === undefined) ? '-' : Math.round(v * 100) + '%';
}

function renderTest1(runs) {
  if (!runs.length) return '<p class="empty">Sin datos de Test 1 para este modelo.</p>';
  let html = '<div class="test-note">Carga sintetica, GuideLLM. "Concurrencia" = streams fijos configurados.</div>';
  html += barChart(runs, r => `${r.strategy}${r.concurrency ? ' @' + r.concurrency : ''}`, r => r.ttft_ms_mean, ' ms TTFT');
  html += '<table><thead><tr><th>Estrategia</th><th>Concurrencia</th><th>Duracion (s)</th><th>TTFT medio (ms)</th><th>TPOT medio (ms)</th><th>Latencia/req (s)</th><th>Output tok/s</th><th>Error rate</th></tr></thead><tbody>';
  for (const r of runs) {
    html += `<tr><td>${r.strategy ?? '-'}</td><td>${r.concurrency ?? '-'}</td><td>${fmt(r.duration_s)}</td>
      <td>${fmt(r.ttft_ms_mean)}</td><td>${fmt(r.tpot_ms_mean)}</td><td>${fmt(r.request_latency_s_mean)}</td>
      <td>${fmt(r.output_tokens_per_second_mean)}</td><td>${pct(r.error_rate)}</td></tr>`;
  }
  return html + '</tbody></table>';
}

function renderTest2(runs) {
  if (!runs.length) return '<p class="empty">Sin datos de Test 2 para este modelo.</p>';
  let html = '<div class="test-note">4 usuarios simulados (general, codigo, resumen, conocimiento) contra Ollama directo.</div>';
  html += barChart(runs, r => `NUM_PARALLEL=${r.ollama_num_parallel ?? '?'}`, r => r.overall_duration_s, ' s total');
  html += '<table><thead><tr><th>OLLAMA_NUM_PARALLEL</th><th>Usuarios</th><th>Duracion total (s)</th><th>Error rate</th><th>TTFT por usuario (s)</th></tr></thead><tbody>';
  for (const r of runs) {
    const ttfts = r.users.map(u => `${u.user_id}: ${fmt(u.ttft_s)}`).join(', ');
    html += `<tr><td>${r.ollama_num_parallel ?? '-'}</td><td>${r.concurrent_users ?? '-'}</td>
      <td>${fmt(r.overall_duration_s)}</td><td>${pct(r.error_rate)}</td><td>${ttfts}</td></tr>`;
  }
  return html + '</tbody></table>';
}

function renderTest3(runs) {
  if (!runs.length) return '<p class="empty">Sin datos de Test 3 para este modelo.</p>';
  let html = '<div class="test-note">Retrieval (Qdrant + nomic-embed-text) + generacion, sobre corpus de docs de Kubernetes.</div>';
  html += barChart(runs, r => `q${r.num_queries} c${r.concurrency}`, r => r.wall_duration_s, ' s wall');
  html += '<table><thead><tr><th>Queries</th><th>Concurrencia</th><th>Wall (s)</th><th>Retrieval medio (s)</th><th>Generacion media (s)</th><th>Tokens in/out</th><th>Error rate</th><th>Hit rate top-k</th></tr></thead><tbody>';
  for (const r of runs) {
    html += `<tr><td>${r.num_queries ?? '-'}</td><td>${r.concurrency ?? '-'}</td><td>${fmt(r.wall_duration_s)}</td>
      <td>${fmt(r.avg_retrieval_s, 3)}</td><td>${fmt(r.avg_generation_s)}</td>
      <td>${fmt(r.avg_input_tokens, 0)} / ${fmt(r.avg_output_tokens, 0)}</td>
      <td>${pct(r.error_rate)}</td>
      <td>${pct(r.expected_chunk_hit_rate)}</td></tr>`;
  }
  return html + '</tbody></table>';
}

function renderModel(model) {
  const d = DATA[model];
  const container = document.getElementById('modelContent');
  container.innerHTML = `
    <section><h2>Test 1 - Raw inference performance</h2>${renderTest1(d.test1)}</section>
    <section><h2>Test 2 - Chatbot concurrente</h2>${renderTest2(d.test2)}</section>
    <section><h2>Test 3 - RAG</h2>${renderTest3(d.test3)}</section>
  `;
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

init();
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
