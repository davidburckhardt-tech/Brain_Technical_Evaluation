#!/usr/bin/env python3
"""Summarize a GuideLLM benchmark run into a single CSV row per strategy, for
cross-run comparison.

Field paths below were confirmed against a real benchmarks.json produced by
guidellm==0.8.1 (Phase 0 smoke test, Ollama + llama3:8b). Each metric lives at
benchmarks[i]["metrics"][<metric_name>]["successful"][<stat_name>]. If a
future GuideLLM version renames these, this script will raise a KeyError with
the metric name, which is the signal to come back and update METRIC_PATHS.
"""

import argparse
import csv
import json
from pathlib import Path

# metric_name -> (JSON key under "metrics", stat to read from the "successful" stats block)
METRIC_PATHS = {
    "ttft_ms_mean": ("time_to_first_token_ms", "mean"),
    "ttft_ms_median": ("time_to_first_token_ms", "median"),
    "itl_ms_mean": ("inter_token_latency_ms", "mean"),
    "tpot_ms_mean": ("time_per_output_token_ms", "mean"),
    "request_latency_s_mean": ("request_latency", "mean"),
    "requests_per_second_mean": ("requests_per_second", "mean"),
    "output_tokens_per_second_mean": ("output_tokens_per_second", "mean"),
}


def summarize_benchmark_entry(benchmark, run_dir):
    config = benchmark.get("config", {})
    row = {
        "run_dir": run_dir,
        "model": config.get("backend", {}).get("model"),
        "strategy": config.get("strategy", {}).get("type_"),
        "duration_s": benchmark.get("duration"),
    }

    metrics = benchmark.get("metrics", {})
    for column_name, (metric_key, stat_key) in METRIC_PATHS.items():
        try:
            row[column_name] = metrics[metric_key]["successful"][stat_key]
        except (KeyError, TypeError):
            row[column_name] = None

    totals = metrics.get("request_totals", {})
    successful = totals.get("successful")
    errored = totals.get("errored")
    total = totals.get("total")
    row["requests_successful"] = successful
    row["requests_errored"] = errored
    row["requests_total"] = total
    row["error_rate"] = (errored / total) if total else None

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to benchmarks.json")
    parser.add_argument("--summary-csv", required=True, help="Path to the cumulative summary CSV")
    parser.add_argument("--run-dir", required=True, help="Path to this run's results directory")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        summarize_benchmark_entry(benchmark, args.run_dir)
        for benchmark in data.get("benchmarks", [])
    ]

    if not rows:
        print("Warning: no benchmarks found in the input file.")
        return

    missing = [name for name, value in rows[0].items() if value is None]
    if missing:
        print(f"Warning: could not find these metrics: {missing}")
        print("Inspect benchmarks.json manually and update METRIC_PATHS in this script.")

    summary_path = Path(args.summary_csv)
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
