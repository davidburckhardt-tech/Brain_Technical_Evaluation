#!/usr/bin/env python3
"""Extract per-request detail from a GuideLLM benchmarks.json into a flat CSV.

GuideLLM already measures every one of these per request (confirmed against
a real benchmarks.json, guidellm==0.8.1) - it just never writes them out
anywhere except buried inside benchmarks[i]["requests"][status][j]. Its own
CSV export (benchmarks.csv) is aggregate-only (one row per strategy). This
script makes the per-request data usable for "latency vs request number"
style plots without re-measuring anything.

One row per request, across all three status buckets (successful, errored,
incomplete), numbered in the order they started (request_start_time) so the
request number reflects submission order under the given concurrency, not
completion order.
"""

import argparse
import csv
import json
from pathlib import Path


def extract_rows(benchmark: dict) -> list[dict]:
    config = benchmark.get("config", {})
    model = config.get("backend", {}).get("model")
    strategy = config.get("strategy", {}).get("type_")
    concurrency = config.get("strategy", {}).get("max_concurrency") or config.get("strategy", {}).get("streams")

    requests = benchmark.get("requests", {})
    all_requests = []
    for status in ("successful", "errored", "incomplete"):
        for r in requests.get(status, []):
            all_requests.append((status, r))

    # Order by submission time, not completion time, so request_number reflects
    # the order requests were actually dispatched under this concurrency level.
    all_requests.sort(key=lambda sr: sr[1].get("request_start_time") or 0)

    rows = []
    for i, (status, r) in enumerate(all_requests):
        rows.append({
            "request_number": i + 1,
            "status": status,
            "model": model,
            "strategy": strategy,
            "concurrency": concurrency,
            "request_id": r.get("request_id"),
            "request_start_time": r.get("request_start_time"),
            "request_end_time": r.get("request_end_time"),
            "request_latency_s": r.get("request_latency"),
            "request_dispatch_delay_s": r.get("request_dispatch_delay"),
            "request_scheduled_latency_s": r.get("request_scheduled_latency"),
            "prompt_tokens": r.get("prompt_tokens"),
            "output_tokens": r.get("output_tokens"),
            "total_tokens": r.get("total_tokens"),
            "ttft_ms": r.get("time_to_first_token_ms"),
            "itl_ms": r.get("inter_token_latency_ms"),
            "tpot_ms": r.get("time_per_output_token_ms"),
            "tokens_per_second": r.get("tokens_per_second"),
            "output_tokens_per_second": r.get("output_tokens_per_second"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to benchmarks.json")
    parser.add_argument("--output", required=True, help="Path to write the per-request CSV")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for benchmark in data.get("benchmarks", []):
        rows.extend(extract_rows(benchmark))

    if not rows:
        print("Warning: no per-request data found in the input file.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"==> Wrote {len(rows)} per-request rows to {output_path}")


if __name__ == "__main__":
    main()
