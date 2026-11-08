# Kubernetes Docs RAG corpus (demo sample)

Source: [evgenypal/k8s-docs-rag-bench](https://hf-p-cfw.fyan.top/datasets/evgenypal/k8s-docs-rag-bench) on Hugging Face. Kubernetes documentation, CC-BY-4.0.

This is a **sample** of the full dataset (7,908 chunks / 5,144 QA pairs), cut down to fit a reasonable embedding time on local Mac hardware for the Phase 0 demo:

- `corpus_sample.jsonl` — 398 pre-chunked passages from 62 Kubernetes doc pages (ConfigMaps, DaemonSets, Service Accounts, init containers, etc.), each already split by the original dataset authors (fields: `id`, `page_id`, `url`, `title`, `section`, `text`, `chunk_index`).
- `qa_eval_sample.jsonl` — 485 verified question/answer pairs whose `source_chunk` falls inside this sample, so retrieval can be checked against a known-correct chunk, not just eyeballed.

The full dataset (952 pages) is available at the link above if this needs to scale up on the real Brain hardware, where embedding 7,908 chunks will be much faster.
