#!/usr/bin/env python3
"""Test 2: concurrent chatbot workload.

Fires one request per entry in datasets/prompts.jsonl at the same time
against the model's OpenAI-compatible chat endpoint, simulating several
users with different kinds of query (general question, code generation,
summarization, general knowledge) hitting the server simultaneously.

Streams tokens live, interleaved and prefixed by user id, so the terminal
output is suitable for recording: this is meant to look like several people
chatting with the model at once, not a single isolated prompt.

Metrics recorded per user: time to first token (TTFT), total latency, output
token count (approximate, via whitespace-split word count), tokens/second.
A JSON summary is written at the end for comparison across runs.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "datasets" / "prompts.jsonl"


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def classify_error(exc: Exception) -> str:
    """Best-effort label for why a request failed, so a failure due to the
    server running out of memory reads differently from a plain timeout or
    connection drop - useful signal when pushing concurrency/model size past
    what the hardware can hold."""
    if isinstance(exc, httpx.ConnectError):
        return "connection_error"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text.lower()
        if "out of memory" in body or "cuda oom" in body or "memory" in body:
            return f"http_{exc.response.status_code}_possible_oom"
        return f"http_{exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


async def run_user(
    client: httpx.AsyncClient, target: str, model: str, entry: dict, print_lock: asyncio.Lock, quiet: bool
):
    user_id = entry["id"]
    category = entry["category"]
    prompt = entry["prompt"]

    start_time = time.monotonic()
    first_token_time: float | None = None
    output_text_parts: list[str] = []

    if not quiet:
        async with print_lock:
            print(f"[{user_id}] ({category}) -> {prompt}", flush=True)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    try:
        async with client.stream("POST", f"{target}/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue

                if first_token_time is None:
                    first_token_time = time.monotonic()

                output_text_parts.append(content)
                if not quiet:
                    async with print_lock:
                        print(f"[{user_id}] {content}", end="", flush=True)
    except Exception as exc:
        error_label = classify_error(exc)
        if not quiet:
            async with print_lock:
                print(f"\n[{user_id}] FAILED: {error_label}\n", flush=True)
        return {
            "user_id": user_id,
            "category": category,
            "prompt": prompt,
            "error": error_label,
            "ttft_s": None,
            "total_latency_s": time.monotonic() - start_time,
            "output_tokens_approx": 0,
            "output_tokens_per_second_approx": None,
            "output_text": "",
        }

    end_time = time.monotonic()
    if not quiet:
        async with print_lock:
            print(f"\n[{user_id}] done.\n", flush=True)

    output_text = "".join(output_text_parts)
    total_latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None
    output_tokens_approx = len(output_text.split())

    return {
        "user_id": user_id,
        "category": category,
        "prompt": prompt,
        "error": None,
        "ttft_s": ttft,
        "total_latency_s": total_latency,
        "output_tokens_approx": output_tokens_approx,
        "output_tokens_per_second_approx": (
            output_tokens_approx / total_latency if total_latency > 0 else None
        ),
        "output_text": output_text,
    }


async def main_async(
    target: str, model: str, prompts_file: Path, output_path: Path, ollama_num_parallel: str | None, quiet: bool
):
    prompts = load_prompts(prompts_file)
    # Auto-silence live per-token/per-user streaming above 16 users - with
    # dozens or hundreds of simulated users, interleaved tokens from every
    # one of them printed live is unreadable noise, not useful for recording.
    quiet = quiet or len(prompts) > 16
    print(f"==> Simulating {len(prompts)} concurrent users against {target} (model: {model})\n")

    print_lock = asyncio.Lock()
    overall_start = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        results = await asyncio.gather(
            *(run_user(client, target, model, entry, print_lock, quiet) for entry in prompts)
        )

    overall_duration = time.monotonic() - overall_start
    errored_count = sum(1 for r in results if r["error"])

    summary = {
        "target": target,
        "model": model,
        "ollama_num_parallel": ollama_num_parallel,
        "concurrent_users": len(prompts),
        "overall_duration_s": overall_duration,
        "error_rate": errored_count / len(results) if results else None,
        "users": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ok_results = [r for r in results if not r["error"]]

    if quiet:
        # Aggregate-only summary - a per-user line per request is noise once
        # there are dozens/hundreds of simulated users.
        ttfts = [r["ttft_s"] for r in ok_results if r["ttft_s"] is not None]
        latencies = [r["total_latency_s"] for r in ok_results]
        tps_values = [
            r["output_tokens_per_second_approx"] for r in ok_results if r["output_tokens_per_second_approx"]
        ]
        print("==> Summary (aggregate):")
        print(f"  Users: {len(prompts)}   Errors: {errored_count} ({summary['error_rate']:.0%})")
        if ttfts:
            print(f"  TTFT:    mean={statistics.mean(ttfts):.2f}s   median={statistics.median(ttfts):.2f}s")
        if latencies:
            print(
                f"  Latency: mean={statistics.mean(latencies):.2f}s   median={statistics.median(latencies):.2f}s"
            )
        if tps_values:
            print(f"  Output tokens/s (approx, per user): mean={statistics.mean(tps_values):.1f}")
    else:
        print("==> Summary:")
        for r in results:
            if r["error"]:
                print(f"  [{r['user_id']}] FAILED: {r['error']}")
                continue
            ttft_str = f"{r['ttft_s']:.2f}s" if r["ttft_s"] is not None else "n/a"
            print(
                f"  [{r['user_id']}] TTFT={ttft_str} "
                f"latency={r['total_latency_s']:.2f}s "
                f"output_tokens~={r['output_tokens_approx']} "
                f"tps~={r['output_tokens_per_second_approx']:.1f}"
            )

    print(f"\n  All {len(prompts)} users finished in {overall_duration:.2f}s (wall clock). Error rate: {summary['error_rate']:.0%}")
    print(f"==> Full results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Base URL of the OpenAI-compatible server (no /v1)")
    parser.add_argument("--model", required=True, help="Model name as known by the server (e.g. Ollama tag)")
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE), help="Path to datasets/prompts.jsonl")
    parser.add_argument("--output", required=True, help="Path to write the JSON summary to")
    parser.add_argument(
        "--ollama-num-parallel",
        default=None,
        help="OLLAMA_NUM_PARALLEL value in effect for this run, recorded in the summary for later comparison (informational only, does not set it).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-user streaming output and the per-user summary table; print only an aggregate "
        "summary at the end. Also applied automatically above 16 users, regardless of this flag.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            main_async(
                args.target,
                args.model,
                Path(args.prompts_file),
                Path(args.output),
                args.ollama_num_parallel,
                args.quiet,
            )
        )
    except httpx.HTTPError as exc:
        print(f"Error: request to the server failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
