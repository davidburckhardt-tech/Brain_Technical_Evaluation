#!/usr/bin/env python3
"""Test 2: concurrent chatbot workload.

Fires one CONVERSATION per entry in datasets/prompts.jsonl at the same time
against the model's OpenAI-compatible chat endpoint, simulating several
users chatting with the model simultaneously. Each conversation can run
several turns (--turns, default 1): turn 1 is the dataset's own prompt,
turns 2+ are generic follow-ups (FOLLOW_UP_TEMPLATES below) sent with the
growing message history, the same way a real multi-turn chat client would -
this is what makes context-growth metrics (cumulative tokens, context
window utilization) possible to measure at all. With --turns 1 (the
default) this is identical in behavior to the original single-shot version.

Streams tokens live, interleaved and prefixed by user id, so the terminal
output is suitable for recording: this is meant to look like several people
chatting with the model at once, not a single isolated prompt.

Metrics recorded per turn: TTFT, generation time, end-to-end latency, real
token counts (input/output, from the API's own `usage` field - not the
previous word-count approximation), cumulative input/output tokens for the
conversation so far, and context window utilization if --max-context-tokens
is given. A JSON summary is written at the end for comparison across runs.
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

# Generic, topic-agnostic follow-ups for turns 2+ - lets any existing prompts
# dataset (prompts.jsonl, prompts_heavy.jsonl, ...) double as a multi-turn
# conversation starter set without having to author per-topic follow-ups.
FOLLOW_UP_TEMPLATES = [
    "Can you elaborate on that with a concrete example?",
    "What's a common misconception people have about this?",
    "How would that change if we needed this to scale to a much larger case?",
    "What are the main trade-offs or risks with the approach you just described?",
    "Can you summarize what you just said in one or two sentences?",
]


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


async def run_turn(
    client: httpx.AsyncClient, target: str, model: str, messages: list[dict],
    user_id: str, turn_id: int, print_lock: asyncio.Lock, quiet: bool,
):
    """Sends one turn's worth of messages (full history so far) and streams
    the reply. Returns (assistant_text, metrics_dict) - the caller appends
    assistant_text to the conversation history for the next turn."""
    start_time = time.monotonic()
    first_token_time: float | None = None
    output_text_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    async with client.stream("POST", f"{target}/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    output_text_parts.append(content)
                    if not quiet:
                        async with print_lock:
                            print(f"[{user_id} t{turn_id}] {content}", end="", flush=True)
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")

    end_time = time.monotonic()
    if not quiet:
        async with print_lock:
            print(f"\n[{user_id} t{turn_id}] done.\n", flush=True)

    output_text = "".join(output_text_parts)
    total_latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None
    generation_duration = (end_time - first_token_time) if first_token_time else None

    metrics = {
        "turn_id": turn_id,
        "error": None,
        "ttft_s": ttft,
        "generation_duration_s": generation_duration,
        "total_latency_s": total_latency,
        # "context tokens" = the full prompt sent this turn (history + new
        # message), as counted by the server itself - this IS the context
        # growth signal: it should rise turn over turn as history piles up.
        "context_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": (output_tokens / total_latency if output_tokens and total_latency > 0 else None),
    }
    return output_text, metrics


async def run_conversation(
    client: httpx.AsyncClient, target: str, model: str, entry: dict, num_turns: int,
    max_context_tokens: int | None, print_lock: asyncio.Lock, quiet: bool,
):
    user_id = entry["id"]
    category = entry["category"]
    first_prompt = entry["prompt"]

    if not quiet:
        async with print_lock:
            print(f"[{user_id}] ({category}) -> {first_prompt}", flush=True)

    messages: list[dict] = []
    turns: list[dict] = []
    conversation_start = time.monotonic()
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0

    try:
        for turn_id in range(1, num_turns + 1):
            user_message = first_prompt if turn_id == 1 else FOLLOW_UP_TEMPLATES[(turn_id - 2) % len(FOLLOW_UP_TEMPLATES)]
            user_input_tokens_approx = len(user_message.split())
            messages.append({"role": "user", "content": user_message})

            assistant_text, turn_metrics = await run_turn(
                client, target, model, messages, user_id, turn_id, print_lock, quiet
            )
            messages.append({"role": "assistant", "content": assistant_text})

            cumulative_output_tokens += turn_metrics["output_tokens"] or 0
            # Cumulative INPUT tokens = sum of each turn's own new user message
            # (not context_tokens, which already includes prior history and
            # would double-count it turn over turn).
            cumulative_input_tokens += user_input_tokens_approx

            turn_metrics["user_input_tokens_approx"] = user_input_tokens_approx
            turn_metrics["cumulative_input_tokens_approx"] = cumulative_input_tokens
            turn_metrics["cumulative_output_tokens"] = cumulative_output_tokens
            turn_metrics["context_window_utilization"] = (
                turn_metrics["context_tokens"] / max_context_tokens
                if turn_metrics["context_tokens"] is not None and max_context_tokens
                else None
            )
            turns.append(turn_metrics)
    except Exception as exc:
        error_label = classify_error(exc)
        if not quiet:
            async with print_lock:
                print(f"\n[{user_id}] FAILED: {error_label}\n", flush=True)
        turns.append({
            "turn_id": len(turns) + 1,
            "error": error_label,
            "ttft_s": None,
            "generation_duration_s": None,
            "total_latency_s": None,
            "context_tokens": None,
            "output_tokens": None,
            "output_tokens_per_second": None,
            "user_input_tokens_approx": None,
            "cumulative_input_tokens_approx": cumulative_input_tokens,
            "cumulative_output_tokens": cumulative_output_tokens,
            "context_window_utilization": None,
        })

    conversation_duration = time.monotonic() - conversation_start
    successful_turns = [t for t in turns if not t["error"]]
    failed_turns = [t for t in turns if t["error"]]

    return {
        "user_id": user_id,
        "category": category,
        "prompt": first_prompt,
        "total_turns": len(turns),
        "successful_turns": len(successful_turns),
        "failed_turns": len(failed_turns),
        "error": failed_turns[0]["error"] if failed_turns else None,
        "conversation_duration_s": conversation_duration,
        # Backward-compatible top-level fields (mirror turn 1 / overall, so
        # existing dashboard code reading ttft_s / total_latency_s etc. at
        # the user level still works unchanged for the --turns 1 case).
        "ttft_s": successful_turns[0]["ttft_s"] if successful_turns else None,
        "total_latency_s": conversation_duration,
        "output_tokens_approx": cumulative_output_tokens,
        "output_tokens_per_second_approx": (
            cumulative_output_tokens / conversation_duration if conversation_duration > 0 else None
        ),
        "output_text": "".join(m["content"] for m in messages if m["role"] == "assistant"),
        "turns": turns,
    }


async def main_async(
    target: str, model: str, prompts_file: Path, output_path: Path, ollama_num_parallel: str | None,
    quiet: bool, num_turns: int, max_context_tokens: int | None,
):
    prompts = load_prompts(prompts_file)
    # Auto-silence live per-token/per-turn streaming above 16 users - with
    # dozens or hundreds of simulated users, interleaved tokens from every
    # one of them printed live is unreadable noise, not useful for recording.
    quiet = quiet or len(prompts) > 16
    print(f"==> Simulating {len(prompts)} concurrent users ({num_turns} turn(s) each) "
          f"against {target} (model: {model})\n")

    print_lock = asyncio.Lock()
    overall_start = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        results = await asyncio.gather(
            *(
                run_conversation(client, target, model, entry, num_turns, max_context_tokens, print_lock, quiet)
                for entry in prompts
            )
        )

    overall_duration = time.monotonic() - overall_start
    errored_count = sum(1 for r in results if r["error"])

    summary = {
        "target": target,
        "model": model,
        "ollama_num_parallel": ollama_num_parallel,
        "concurrent_users": len(prompts),
        "turns_per_conversation": num_turns,
        "max_context_tokens": max_context_tokens,
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
        print(f"  Users: {len(prompts)}   Turns/user: {num_turns}   Errors: {errored_count} ({summary['error_rate']:.0%})")
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
            if r["error"] and r["successful_turns"] == 0:
                print(f"  [{r['user_id']}] FAILED: {r['error']}")
                continue
            ttft_str = f"{r['ttft_s']:.2f}s" if r["ttft_s"] is not None else "n/a"
            print(
                f"  [{r['user_id']}] turns={r['successful_turns']}/{r['total_turns']} TTFT(t1)={ttft_str} "
                f"conv_duration={r['conversation_duration_s']:.2f}s "
                f"output_tokens={r['output_tokens_approx']} "
                f"tps~={fmt_or_na(r['output_tokens_per_second_approx'])}"
            )

    print(f"\n  All {len(prompts)} users finished in {overall_duration:.2f}s (wall clock). Error rate: {summary['error_rate']:.0%}")
    print(f"==> Full results saved to {output_path}")


def fmt_or_na(value):
    return f"{value:.1f}" if value is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Base URL of the OpenAI-compatible server (no /v1)")
    parser.add_argument("--model", required=True, help="Model name as known by the server (e.g. Ollama tag)")
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE), help="Path to datasets/prompts.jsonl")
    parser.add_argument("--output", required=True, help="Path to write the JSON summary to")
    parser.add_argument(
        "--turns", type=int, default=1,
        help="Turns per simulated conversation (default 1 = original single-shot behavior). "
        "Turn 1 uses the dataset's own prompt; turns 2+ use generic follow-ups.",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=None,
        help="Model's context window size, used only to compute context_window_utilization per turn. "
        "Omit to leave that metric null.",
    )
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
                args.turns,
                args.max_context_tokens,
            )
        )
    except httpx.HTTPError as exc:
        print(f"Error: request to the server failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
