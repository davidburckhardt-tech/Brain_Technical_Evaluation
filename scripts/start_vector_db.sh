#!/usr/bin/env bash
# Starts Qdrant (vector database for Test 3 - RAG) via Docker Compose.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Starting Qdrant via Docker Compose..."
(cd "${PROJECT_ROOT}" && docker compose up -d qdrant)

echo "==> Waiting for Qdrant to become ready..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:6333/readyz" >/dev/null 2>&1; then
    echo "Qdrant is ready."
    break
  fi
  sleep 1
done

echo "==> Qdrant REST API available at: http://localhost:6333"
echo "    Dashboard: http://localhost:6333/dashboard"
