#!/usr/bin/env python3
"""Samples vLLM's own Prometheus /metrics endpoint at a fixed interval,
appending rows to a CSV - the only way to get queue length, KV cache
utilization and active/waiting sequence counts, none of which GuideLLM (an
external client) can see or report; vLLM's own scheduler knows them exactly.

Meant to run in the background alongside a benchmark, same pattern as
monitor_resources.py, started/stopped by the run_*.sh wrappers. Not specific
to GuideLLM - any load test hitting the same vLLM server can use this.

Parses the Prometheus text exposition format with a plain regex rather than
a full client library (avoids adding a dependency for 4 numbers), looking
for the "vllm:<metric_name>" line (ignoring any {labels}) for each metric in
TRACKED_METRICS. Different vLLM versions have renamed some of these before
(e.g. num_requests_swapped was dropped when CPU-offload swapping was
removed) - a metric not found in a given server's /metrics output is simply
left blank in the CSV rather than raising, since this script shouldn't be
the reason a benchmark run fails.
"""

import argparse
import csv
import re
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TRACKED_METRICS = [
    "num_requests_running",
    "num_requests_waiting",
    "num_requests_swapped",
    "gpu_cache_usage_perc",
    "cpu_cache_usage_perc",
]

FIELDNAMES = ["timestamp_iso", "elapsed_s", *TRACKED_METRICS]

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def fetch_metrics_text(metrics_url: str) -> str:
    with urllib.request.urlopen(metrics_url, timeout=5) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_metric(text: str, metric_name: str) -> float | None:
    # Matches "vllm:metric_name{...optional labels...} <value>" at line start,
    # taking the last match if the metric is reported per-model/per-label
    # (sums would double-count; last value matches vLLM's own /metrics
    # convention of one active series per process in our single-model setup).
    pattern = re.compile(rf"^vllm:{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", required=True, help="Base URL of the vLLM server, e.g. http://localhost:8000")
    parser.add_argument("--output", required=True, help="CSV file to write samples to")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    metrics_url = args.vllm_url.rstrip("/") + "/metrics"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"==> Monitoring vLLM metrics at {metrics_url} every {args.interval}s -> {args.output}")

    start_time = time.monotonic()
    warned_unreachable = False
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        while not _stop:
            now = datetime.now(timezone.utc).isoformat()
            elapsed = time.monotonic() - start_time
            row = {"timestamp_iso": now, "elapsed_s": round(elapsed, 3)}
            try:
                text = fetch_metrics_text(metrics_url)
                for metric in TRACKED_METRICS:
                    row[metric] = parse_metric(text, metric)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if not warned_unreachable:
                    print(f"Warning: {metrics_url} unreachable ({exc}); this server may not be vLLM, "
                          f"or /metrics may be disabled. Rows will be blank.", flush=True)
                    warned_unreachable = True
                for metric in TRACKED_METRICS:
                    row[metric] = None

            writer.writerow(row)
            f.flush()
            time.sleep(args.interval)

    print(f"==> vLLM metrics monitoring stopped. Samples saved to {output_path}")


if __name__ == "__main__":
    main()
