#!/usr/bin/env python3
"""Test 3 (RAG) - step 1: ingestion.

Two input modes:
  --documents-dir  Reads .txt/.md documents from a directory and splits them
                    into overlapping chunks (for arbitrary documents).
  --jsonl-file      Reads a JSONL file of ALREADY-chunked passages (one JSON
                    object per line, with at least an "id" and "text" field)
                    and ingests each line as-is, with no further splitting -
                    for pre-chunked corpora such as datasets/k8s_rag/corpus_sample.jsonl.

Either way, each chunk is embedded via the OpenAI-compatible /v1/embeddings
endpoint (works against Ollama or vLLM unchanged) and upserted into a Qdrant
collection.

This only measures ingestion time (useful on its own, as part of the demo's
"how much real AI infrastructure can we run locally" story). Retrieval and
generation are measured separately in query.py, so the RAG pipeline's time
is never attributed entirely to the LLM.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "datasets" / "documents"


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping chunks, measured in whitespace-split words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def embed_text(client: httpx.Client, embedding_host: str, model: str, text: str) -> list[float]:
    response = client.post(
        f"{embedding_host}/v1/embeddings",
        json={"model": model, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def ensure_collection(client: httpx.Client, qdrant_host: str, collection: str, vector_size: int):
    response = client.get(f"{qdrant_host}/collections/{collection}")
    if response.status_code == 200:
        return

    response = client.put(
        f"{qdrant_host}/collections/{collection}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30.0,
    )
    response.raise_for_status()


def load_prechunked_jsonl(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--documents-dir", default=None)
    input_group.add_argument("--jsonl-file", default=None)
    parser.add_argument("--embedding-host", default="http://localhost:11434", help="OpenAI-compatible base URL serving the embedding model (Ollama or vLLM)")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--qdrant-host", default="http://localhost:6333")
    parser.add_argument("--collection", default="brain_demo_docs")
    parser.add_argument("--chunk-size-words", type=int, default=150)
    parser.add_argument("--overlap-words", type=int, default=30)
    args = parser.parse_args()

    all_chunks = []

    if args.jsonl_file:
        jsonl_path = Path(args.jsonl_file)
        all_chunks = load_prechunked_jsonl(jsonl_path)
        print(f"==> Loaded {len(all_chunks)} pre-chunked passages from {jsonl_path}")
    else:
        documents_dir = Path(args.documents_dir or DEFAULT_DOCUMENTS_DIR)
        doc_paths = sorted(
            p for p in documents_dir.glob("*") if p.suffix.lower() in (".txt", ".md")
        )
        if not doc_paths:
            print(f"Error: no .txt/.md files found in {documents_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"==> Found {len(doc_paths)} document(s) in {documents_dir}")
        for doc_path in doc_paths:
            text = doc_path.read_text(encoding="utf-8")
            chunks = chunk_text(text, args.chunk_size_words, args.overlap_words)
            print(f"  {doc_path.name}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                all_chunks.append({"source_file": doc_path.name, "chunk_index": i, "text": chunk})

    with httpx.Client() as client:
        print(f"==> Embedding {len(all_chunks)} chunks with '{args.embedding_model}'...")
        start_time = time.monotonic()

        points = []
        for i, chunk in enumerate(all_chunks):
            vector = embed_text(client, args.embedding_host, args.embedding_model, chunk["text"])
            points.append({"id": i, "vector": vector, "payload": chunk})
            if (i + 1) % 50 == 0 or (i + 1) == len(all_chunks):
                elapsed = time.monotonic() - start_time
                print(f"  embedded {i + 1}/{len(all_chunks)} ({elapsed:.1f}s elapsed)")

        embedding_duration = time.monotonic() - start_time
        vector_size = len(points[0]["vector"])

        print(f"==> Ensuring Qdrant collection '{args.collection}' (vector size {vector_size})...")
        ensure_collection(client, args.qdrant_host, args.collection, vector_size)

        print(f"==> Upserting {len(points)} points into Qdrant...")
        upsert_start = time.monotonic()
        response = client.put(
            f"{args.qdrant_host}/collections/{args.collection}/points",
            json={"points": points},
            timeout=60.0,
        )
        response.raise_for_status()
        upsert_duration = time.monotonic() - upsert_start

    print()
    print("==> Ingestion complete:")
    print(f"    {len(all_chunks)} chunks ingested")
    print(f"    Embedding time: {embedding_duration:.2f}s ({embedding_duration / len(all_chunks):.3f}s/chunk)")
    print(f"    Qdrant upsert time: {upsert_duration:.2f}s")
    print(f"    Collection: '{args.collection}' at {args.qdrant_host}")


if __name__ == "__main__":
    main()
