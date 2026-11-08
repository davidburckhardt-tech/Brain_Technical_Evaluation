#!/usr/bin/env python3
"""Test 3 (RAG) - step 2: retrieval + generation, performance only (not answer
quality - that is a separate, later concern).

For each of N questions (taken from a QA jsonl file), this:
  1. embeds the question,
  2. retrieves the top-k nearest chunks from Qdrant,
  3. sends question + retrieved context to the LLM and streams the answer.

Retrieval time and generation time are measured SEPARATELY per the plan, so
neither step's cost is attributed to the other. `--concurrency` controls how
many questions run at once (an asyncio semaphore caps in-flight requests);
`--num-queries` controls how many questions (in file order) to run in total.

Both embedding and generation go through the OpenAI-compatible API
(/v1/embeddings, /v1/chat/completions with stream_options.include_usage for
exact token counts) rather than any Ollama-native endpoint, so this script
works unchanged against Ollama OR vLLM - only --embedding-host/--ollama-host
need to point at the right server(s). Embeddings and generation can be two
different servers (--embedding-host vs --ollama-host), since vLLM typically
serves one model per process - see README.md Sec. 7 for the real-hardware setup.
Document extraction/chunking/embedding timing is intentionally NOT broken
out further here; that is measured once, at ingestion time, by ingest.py.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import NamedTuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_FILE = PROJECT_ROOT / "datasets" / "k8s_rag" / "qa_eval_sample.jsonl"

PROMPT_TEMPLATE = """Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:"""


def load_questions(path: Path, num_queries: int) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
            if len(questions) >= num_queries:
                break
    return questions


async def embed_question(client: httpx.AsyncClient, embedding_host: str, model: str, text: str) -> list[float]:
    response = await client.post(
        f"{embedding_host}/v1/embeddings",
        json={"model": model, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


async def retrieve_chunks(
    client: httpx.AsyncClient, qdrant_host: str, collection: str, vector: list[float], top_k: int
) -> list[dict]:
    response = await client.post(
        f"{qdrant_host}/collections/{collection}/points/search",
        json={"vector": vector, "limit": top_k, "with_payload": True},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["result"]


def classify_error(exc: Exception) -> str:
    """Best-effort label for why a request failed - a plain timeout reads
    differently from a server error that looks like it ran out of memory,
    which matters once we're pushing concurrency/model size against real
    hardware limits."""
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


class GenerationResult(NamedTuple):
    text: str
    ttft_s: float | None
    input_tokens: int | None
    output_tokens: int | None


async def generate_answer(client: httpx.AsyncClient, ollama_host: str, model: str, prompt: str) -> GenerationResult:
    """Streams the answer via the OpenAI-compatible API (works against Ollama
    and vLLM unchanged) and returns exact token counts from the final chunk's
    `usage` field, requested via stream_options.include_usage - both Ollama
    and vLLM populate it the same way OpenAI does."""
    start = time.monotonic()
    first_token_time: float | None = None
    parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream(
        "POST", f"{ollama_host}/v1/chat/completions", json=payload, timeout=120.0
    ) as response:
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
                    parts.append(content)
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")

    text = "".join(parts)
    ttft = (first_token_time - start) if first_token_time else None
    return GenerationResult(text, ttft, input_tokens, output_tokens)


async def run_one_query(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    qa: dict,
    index: int,
    print_lock: asyncio.Lock,
) -> dict:
    async with semaphore:
        overall_start = time.monotonic()

        try:
            retrieval_start = time.monotonic()
            query_vector = await embed_question(client, args.embedding_host, args.embedding_model, qa["question"])
            chunks = await retrieve_chunks(client, args.qdrant_host, args.collection, query_vector, args.top_k)
            retrieval_duration = time.monotonic() - retrieval_start

            retrieved_chunk_ids = [c["payload"]["id"] for c in chunks]
            found_expected = qa.get("source_chunk") in retrieved_chunk_ids
            context = "\n\n".join(c["payload"]["text"] for c in chunks)
            prompt = PROMPT_TEMPLATE.format(context=context, question=qa["question"])

            generation_start = time.monotonic()
            generation = await generate_answer(client, args.ollama_host, args.model, prompt)
            generation_duration = time.monotonic() - generation_start
        except Exception as exc:
            error_label = classify_error(exc)
            async with print_lock:
                print(f"[q{index}] FAILED: {error_label} -> {qa['question'][:70]}")
            return {
                "index": index,
                "question_id": qa.get("id"),
                "error": error_label,
                "expected_chunk_found_in_top_k": None,
                "retrieval_duration_s": None,
                "generation_ttft_s": None,
                "generation_duration_s": None,
                "total_duration_s": time.monotonic() - overall_start,
                "input_tokens": None,
                "output_tokens": None,
            }

        total_duration = time.monotonic() - overall_start

        async with print_lock:
            print(f"[q{index}] retrieval={retrieval_duration:.2f}s generation={generation_duration:.2f}s "
                  f"(found_expected_chunk={found_expected}, in={generation.input_tokens}tok "
                  f"out={generation.output_tokens}tok) -> {qa['question'][:70]}")

        return {
            "index": index,
            "question_id": qa.get("id"),
            "error": None,
            "expected_chunk_found_in_top_k": found_expected,
            "retrieval_duration_s": retrieval_duration,
            "generation_ttft_s": generation.ttft_s,
            "generation_duration_s": generation_duration,
            "total_duration_s": total_duration,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
        }


async def main_async(args: argparse.Namespace):
    questions = load_questions(Path(args.qa_file), args.num_queries)
    print(f"==> Running {len(questions)} RAG queries (concurrency={args.concurrency}) "
          f"against model={args.model}, collection={args.collection}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    print_lock = asyncio.Lock()

    wall_start = time.monotonic()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(
                run_one_query(client, semaphore, args, qa, i, print_lock)
                for i, qa in enumerate(questions)
            )
        )
    wall_duration = time.monotonic() - wall_start

    succeeded = [r for r in results if not r["error"]]
    error_rate = (len(results) - len(succeeded)) / len(results) if results else None

    summary = {
        "num_queries": len(questions),
        "concurrency": args.concurrency,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "collection": args.collection,
        "wall_duration_s": wall_duration,
        "error_rate": error_rate,
        "queries": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("==> Summary:")
    print(f"    queries={len(results)} concurrency={args.concurrency} wall_duration={wall_duration:.2f}s "
          f"error_rate={error_rate:.0%}")

    if succeeded:
        avg_retrieval = sum(r["retrieval_duration_s"] for r in succeeded) / len(succeeded)
        avg_generation = sum(r["generation_duration_s"] for r in succeeded) / len(succeeded)
        avg_input_tokens = sum(r["input_tokens"] or 0 for r in succeeded) / len(succeeded)
        avg_output_tokens = sum(r["output_tokens"] or 0 for r in succeeded) / len(succeeded)
        hit_rate = sum(1 for r in succeeded if r["expected_chunk_found_in_top_k"]) / len(succeeded)
        print(f"    avg retrieval time: {avg_retrieval:.3f}s")
        print(f"    avg generation time: {avg_generation:.2f}s")
        print(f"    avg tokens: {avg_input_tokens:.0f} in / {avg_output_tokens:.0f} out")
        print(f"    expected-chunk hit rate (top-{args.top_k}): {hit_rate:.0%}")
    print(f"==> Full results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-file", default=str(DEFAULT_QA_FILE))
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Generation server (Ollama or vLLM), OpenAI-compatible base URL")
    parser.add_argument("--embedding-host", default=None, help="Embedding server, OpenAI-compatible base URL. Defaults to --ollama-host (same server for both).")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--qdrant-host", default="http://localhost:6333")
    parser.add_argument("--collection", default="k8s_docs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.embedding_host is None:
        args.embedding_host = args.ollama_host

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
