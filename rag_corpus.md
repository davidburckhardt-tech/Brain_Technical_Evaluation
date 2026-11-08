# RAG Corpus — Kubernetes Documentation (Phase 0)

## Source

[evgenypal/k8s-docs-rag-bench](https://hf-p-cfw.fyan.top/datasets/evgenypal/k8s-docs-rag-bench) on Hugging Face — official Kubernetes documentation, in English, **CC-BY-4.0** license.

The full dataset has:
- 956 HTML pages of real technical documentation.
- 7,908 chunks pre-split by the dataset's authors.
- 5,144 verified question/answer pairs.
- 172,700 evaluation labels (judge labels).

## What's used in this phase (a sample, not the full dataset)

Downloading all 7,908 chunks wasn't necessary to validate the pipeline on Mac, so a sample was taken aimed at maximizing coverage of verified questions within a small chunk budget:

- **Selection criteria**: pages with associated evaluation questions, prioritizing pages with more questions, excluding giant pages (>20 chunks, e.g. the full `kubectl` reference) so as not to spend the whole budget on a single page.
- **Result**: 62 pages, 398 chunks, 485 question/answer pairs whose `source_chunk` falls within those 398 chunks (i.e. with full traceability: every question has a known "correct" chunk to validate retrieval against).

Files in `datasets/k8s_rag/`:
- `corpus_sample.jsonl` — the 398 chunks.
- `qa_eval_sample.jsonl` — the 485 verified questions/answers.
- `README.md` — attribution and a link to the full dataset.

## Chunk schema (`corpus_sample.jsonl`)

| Field | Description |
|---|---|
| `id` | Unique chunk id (e.g. `concepts_configuration_secret-0`) |
| `page_id` | Id of the source page |
| `url` | Original URL on kubernetes.io/docs |
| `title` | Page title |
| `section` | Section within the page |
| `text` | The chunk's text (what gets embedded) |
| `chunk_index` | Position of the chunk within the page |
| `page_kind` | Page type (`concept`, `narrative`, etc.) |

## Question schema (`qa_eval_sample.jsonl`)

| Field | Description |
|---|---|
| `id` | Question id |
| `question` | Question, in English |
| `answer` | Reference answer (ground truth) |
| `source_chunk` | Id of the chunk containing the answer — lets you verify whether retrieval found the right chunk |
| `source_page` | Source page |
| `provenance` | How the question was generated (e.g. `manual`) |

## How it's loaded today

- **Embeddings**: `nomic-embed-text` (via Ollama), 768 dimensions.
- **Vector database**: Qdrant, collection `k8s_docs`, 398 points, cosine distance.
- **Measured ingestion time**: ~11s to embed + upload all 398 chunks on this Mac (not a bottleneck).
- Ingestion script: `benchmarks/rag/ingest.py --jsonl-file datasets/k8s_rag/corpus_sample.jsonl --collection k8s_docs`.

## Scaling to the real unit

The full dataset (952 pages / 7,908 chunks) is available at the link above if it makes sense to use the full corpus on the real unit (with a GPU, embedding 7,908 chunks should take seconds). See also the pending Qdrant snapshot/restore section in [PROJECT_PLAN.md](PROJECT_PLAN.md), to avoid having to re-ingest within the 4h window.
