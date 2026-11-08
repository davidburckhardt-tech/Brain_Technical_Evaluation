#!/usr/bin/env bash
# Test 3 (RAG) - step 1: embed and load a corpus into Qdrant. Run
# scripts/start_vector_db.sh first.
#
# Defaults to the pre-chunked Kubernetes docs sample (datasets/k8s_rag/corpus_sample.jsonl,
# collection "k8s_docs" - what query.py also defaults to). Override RAG_CORPUS_FILE
# and RAG_COLLECTION in .env to load something else; set RAG_CORPUS_FILE to a
# directory instead of switching ingest.py's --jsonl-file for --documents-dir
# chunking of arbitrary .txt/.md files.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

# EMBEDDING_HOST lets embeddings run against a different server than
# generation (e.g. Ollama serving nomic-embed-text while vLLM serves the
# generation model) - defaults to OLLAMA_HOST when unset.
EMBEDDING_HOST="${EMBEDDING_HOST:-${OLLAMA_HOST:-http://localhost:11434}}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
RAG_CORPUS_FILE="${RAG_CORPUS_FILE:-${PROJECT_ROOT}/datasets/k8s_rag/corpus_sample.jsonl}"
RAG_COLLECTION="${RAG_COLLECTION:-k8s_docs}"

if [[ ! -f "${VENV_DIR}/bin/python3" ]]; then
  echo "Error: virtual environment not found. Run scripts/setup.sh first." >&2
  exit 1
fi

if ! curl -sf "http://localhost:6333/readyz" >/dev/null 2>&1; then
  echo "Error: Qdrant is not running. Run scripts/start_vector_db.sh first." >&2
  exit 1
fi

"${VENV_DIR}/bin/python3" "${PROJECT_ROOT}/benchmarks/rag/ingest.py" \
  --jsonl-file "${RAG_CORPUS_FILE}" \
  --collection "${RAG_COLLECTION}" \
  --embedding-host "${EMBEDDING_HOST}" \
  --embedding-model "${EMBEDDING_MODEL}"
