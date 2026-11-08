# Results — Phase 0 (Mac smoke test)

**Important**: these results are from a Mac (CPU/Metal, no dedicated GPU), run solely to validate that the benchmark pipeline measures correctly. **They are not representative of Brain unit performance** — that's only measured once there's access to the real hardware (see [PROJECT_PLAN.md](PROJECT_PLAN.md)).

**Environment**: macOS, Ollama (local runtime). Two models tested, from different families: `llama3:8b` (tokenizer `NousResearch/Meta-Llama-3-8B`) and `qwen3:4b-q4_K_M` (tokenizer `Qwen/Qwen3-4B`, Q4_K_M quantization). The second model was added specifically so the cross-model comparison has at least 2 data points — the rest of the candidate list (Mistral, GLM) is still pending on the real unit.

---

## Test 1 — Raw Inference Performance

Benchmark: GuideLLM, synthetic load (~550 input tokens, ~150 output tokens per request).

| Run | Concurrency (`streams`) | Requests | Total duration | Mean TTFT | Median TTFT | Mean ITL | Mean TPOT | Mean latency/request | Requests/sec | Output tokens/sec |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 (sequential) | 5 | 55.6s | 914 ms | 189 ms | 30.1 ms | 32.7 ms | 11.1s | 0.09 | 30.5 |
| 2 | 2 | 10 | 79.9s | 9,759 ms | 10,289 ms | 28.8 ms | 80.1 ms | 15.2s | 0.13 | 23.7 |

**Reading (llama3:8b)**: going from 1 to 2 concurrent streams, TTFT multiplied by ~10x (from ~900ms to ~9.8s) and TPOT nearly doubled (32.7 → 80.1 ms/token). Aggregate throughput (output tokens/sec) dropped slightly instead of rising. This is consistent with hardware that has no free compute headroom: the Mac has no idle capacity to absorb a second request in parallel, so concurrency just splits the same compute across more requests, worsening each one's latency.

Raw data: `results/llama3_8b/20261107_184204_c1/` and `results/llama3_8b/20261107_203726_c2/`.

### Second model: `qwen3:4b-q4_K_M`

| Run | Concurrency | Requests completed | Total duration | Mean TTFT | Mean TPOT | Mean latency/request | Output tokens/sec | Error rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 9/10 | 310.3s | 1,774 ms | 47.9 ms | 64.5s | 39.0 | 0% |

**Reading**: Qwen3 4B (a smaller model, 4B vs. 8B parameters, Q4_K_M quantization) had higher output tokens/sec than Llama 3 8B under the same concurrency condition (39.0 vs. 23.7 tok/s) — expected for a smaller model — but also higher mean latency per request (64.5s vs. 15.2s) and only completed 9 of 10 requests in the allotted time. This isn't a perfectly controlled comparison (different architecture, quantization, size); it serves as a first multi-model reference point, not a definitive conclusion — that requires the full model list on the real unit.

Raw data: `results/qwen3_4b-q4_K_M/20261107_222531_c2/`.

---

## Test 2 — Concurrent chatbot (4 simulated users)

Script: `benchmarks/chatbot/simulate_users.py`, 4 simultaneous users with different kinds of query (general, code generation, summarization, general knowledge), hitting Ollama directly. Control variable: `OLLAMA_NUM_PARALLEL` (how many requests Ollama actually processes in parallel via continuous batching; requires restarting the server to apply).

| `OLLAMA_NUM_PARALLEL` | User | Category | TTFT | Total latency | Output tokens (approx.) | Tokens/sec (approx.) |
|---|---|---|---|---|---|---|
| **1** (default) | general-1 | general | 47.99s | 53.84s | 128 | 2.4 |
| | code-1 | code | 6.08s | 13.07s | 160 | 12.2 |
| | summary-1 | summarization | 13.20s | 30.48s | 451 | 14.8 |
| | knowledge-1 | knowledge | 30.64s | 47.78s | 414 | 8.7 |
| | **Total (4 users)** | | | **53.89s** | | |
| **2** | general-1 | general | 34.94s | 38.61s | 72 | 1.9 |
| | code-1 | code | 23.33s | 37.55s | 168 | 4.5 |
| | summary-1 | summarization | 0.46s | 23.13s | 310 | 13.4 |
| | knowledge-1 | knowledge | 0.47s | 34.68s | 465 | 13.4 |
| | **Total (4 users)** | | | **38.66s** | | |
| **3** (value left in `.env`) | general-1 | general | 13.94s | 19.35s | 81 | 4.2 |
| | code-1 | code | 0.81s | 13.67s | 138 | 10.1 |
| | summary-1 | summarization | 0.81s | 30.19s | 372 | 12.3 |
| | knowledge-1 | knowledge | 0.81s | 33.24s | 485 | 14.6 |
| | **Total (4 users)** | | | **33.31s** | | |

**Reading**:
- With `NUM_PARALLEL=1`, Ollama serializes: each user waits for the previous one to fully finish (that's why the last user to start, `general-1`, has a 48s TTFT).
- With `NUM_PARALLEL=2`, 2 users (`summary-1`, `knowledge-1`) start almost instantly (TTFT ~0.46s); the other 2 wait for a free slot.
- With `NUM_PARALLEL=3`, 3 users (`code-1`, `summary-1`, `knowledge-1`) start almost instantly (TTFT ~0.81s) and **don't wait on each other**; only the 4th (`general-1`) waits for a slot to free up (13.9s).
- Total time for the 4 users dropped from 53.9s → 38.7s → 33.3s as `NUM_PARALLEL` went up.
- Aggregate throughput (tokens/sec summed across users) **didn't improve significantly** between levels — what improves is that more users stop waiting through a full queue, not that the hardware does more work. Same conclusion as Test 1: on this Mac, compute is the limit, not scheduling.

Raw data: `results/llama3_8b/20261107_204825_chatbot.json` (parallel=1), `results/llama3_8b/20261107_205940_chatbot.json` (parallel=2), `results/llama3_8b/20261107_210047_chatbot.json` (parallel=3).

### Second model: `qwen3:4b-q4_K_M` (NUM_PARALLEL=3)

| User | Category | TTFT | Total latency | Output tokens | Tokens/sec |
|---|---|---|---|---|---|
| general-1 | general | 12.85s | 17.09s | 54 | 3.2 |
| code-1 | code | 39.70s | 61.83s | 243 | 3.9 |
| summary-1 | summarization | 72.71s | 94.79s | 337 | 3.6 |
| knowledge-1 | knowledge | 69.60s | 101.96s | 583 | 5.7 |
| **Total (4 users)** | | | **102.04s** | | |

Error rate: 0%. Raw data: `results/qwen3_4b-q4_K_M/20261107_223208_chatbot.json`.

**Reading**: at the same `NUM_PARALLEL=3`, Qwen3 4B took 3x longer than Llama 3 8B to finish all 4 users (102s vs. 33.3s), and its responses were longer on average — the size of the generated response matters as much as model size in this kind of comparison, one more reason not to draw "better/worse model" conclusions yet from a single run each.

---

## Test 3 — RAG (retrieval + generation)

Pipeline: question → embed (`nomic-embed-text`) → retrieve top-3 chunks from Qdrant (`k8s_docs`) → generate the answer (`llama3:8b`). Questions taken from `datasets/k8s_rag/qa_eval_sample.jsonl` (in file order, not randomized). Retrieval and generation are measured separately on purpose, so the vector database's time isn't attributed to the LLM. `OLLAMA_NUM_PARALLEL` was still at 3 (carried over from Test 2).

| Configuration | Queries | Concurrency | Total duration (wall clock) | Mean retrieval | Mean generation | Expected chunk found (top-3) |
|---|---|---|---|---|---|---|
| 1 user | 1 | 1 | 23.17s* | 22.296s* | 0.81s | 100% |
| 2 sequential | 2 | 1 | 3.34s | 0.067s | 1.58s | 100% |
| 3 sequential | 3 | 1 | 4.57s | 0.044s | 1.47s | 67% |
| 2 concurrent (2 threads) | 2 | 2 | 1.50s | 0.173s | 1.18s | 100% |
| 3 concurrent (2 threads) | 3 | 2 | 2.65s | 0.069s | 1.46s | 67% |
| 4 concurrent (2 threads) | 4 | 2 | 6.26s | 0.083s | 2.63s | 75% |

\* The first run (1 user) shows a 22.3s retrieval time because it was the first time the embedding model was used in this Ollama session — that time is the model's cold load into memory, not real retrieval. Subsequent runs, with the model already loaded, show the real retrieval cost: **0.04-0.17s**, practically negligible compared to generation. **For the real unit: it's worth adding an embedding-model warm-up before measuring**, same as is already done for the LLM in `start_inference_ollama.sh`.

**Reading**:
- Retrieval (embedding + Qdrant search) is very fast once warm — it's never the bottleneck in this pipeline, sequential or concurrent. Generation time dominates, as expected.
- At concurrency 2 (within the `OLLAMA_NUM_PARALLEL=3` limit set in Test 2), the queueing degradation seen in Test 2 without parallel slots didn't show up here — a signal that leaving `OLLAMA_NUM_PARALLEL` above 1 also benefits this test.
- The "hit rate" (whether the correct chunk showed up in the top-3 retrieved) varied between 67-100% depending on which questions landed in the sample — not a load metric, it's a retrieval-quality signal that came along for free because the dataset carries the known-correct chunk (`source_chunk`), but it isn't the focus of this round (performance metrics were requested, not quality ones).

Raw data: `results/llama3_8b/*_rag_q*_c*.json` (6 files, one per configuration).

Note: these 6 runs predate the switch to exact token counting (see below) — they show `output_tokens_approx` (word-count approximated), not exact tokens.

### Second model: `qwen3:4b-q4_K_M`

With exact token counting (via the OpenAI-compatible API's `stream_options.include_usage`, not word-count approximation):

| Configuration | Queries | Concurrency | Wall clock | Mean retrieval | Mean generation | Tokens in/out (avg.) | Error rate | Hit rate top-3 |
|---|---|---|---|---|---|---|---|---|
| 1 user | 1 | 1 | 12.67s | 0.942s | 11.68s | 532 / 429 | 0% | 100% |
| 4 concurrent (2 threads) | 4 | 2 | 37.12s | 0.165s | 14.69s | 508 / 357 | 0% | 75% |

Raw data: `results/qwen3_4b-q4_K_M/20261107_223400_rag_q1_c1.json`, `results/qwen3_4b-q4_K_M/20261107_223413_rag_q4_c2.json`.

**Reading**: same pattern as Test 1/2 — Qwen3 4B generated noticeably more output tokens per answer (357-429 vs. what Llama generated on these same questions) and took longer to generate, while retrieval (independent of the LLM, same embedding/Qdrant) stayed equally fast for both models — confirms that separating retrieval from generation was the right call: retrieval doesn't vary by model, generation does.

---

## Conclusion for this phase

All 3 tests (raw inference, concurrent chatbot, RAG) are validated end-to-end on Mac and ready to be ported to vLLM on the real unit. The expectation for the real unit is that, unlike this Mac, there will be actual compute headroom (GPU) for raising concurrency to improve aggregate throughput instead of just splitting the queue — that's exactly what needs to be measured again there.
