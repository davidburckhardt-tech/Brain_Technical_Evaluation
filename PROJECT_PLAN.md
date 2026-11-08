# Brain Demo — Project Plan

## 1. Objective

Produce material (an editable video) that demonstrates, for a technical audience, the capabilities of the "Brain" units:

1. What open source models the hardware can run (from small to large, across several families).
2. What performance it has (tokens/sec, latency, behavior under concurrent load).
3. What real applications it supports (not just an isolated prompt): concurrent chatbot and RAG.

**This is not a product or an end-user UI.** It's a recorded demo, with terminal/metrics on screen, edited afterward.

## 2. Central constraint

Access to the real units: ~4 hours per unit (to be confirmed with Nicolas whether that's per unit or total). This implies:

- Nothing gets improvised during that window. Everything is tested beforehand.
- This weekend's work is a **reproducible lab**, not the final results (the real results depend on hardware we don't have yet).

## 3. Architecture decision

**Important: the focus is on open source models running locally, not on OpenAI APIs.** OpenAI (the company) is not part of the demo at any point — "OpenAI-compatible API" is used only as an *interface format* (the standard every benchmark tool expects), not as a service.

| Phase | Inference engine | Why |
|---|---|---|
| Phase 0 (now, local Mac) | **Ollama** | Already installed/simple on Mac, exposes an OpenAI-compatible endpoint (`/v1/chat/completions`), good for validating that the benchmark harness works end-to-end. |
| Real phase (Brain units, NVIDIA GPU) | **vLLM** | Production server, same API format, but with better real throughput/concurrency and quantization support (AWQ/GPTQ) for large models. |

The benchmark tooling (GuideLLM/AIPerf) doesn't change between phases because both engines speak the same protocol — that's the key point: what's being validated now on Mac with Ollama is the same measurement pipeline that will be used on the real unit, only the backend changes.

### Benchmark tool

- **GuideLLM** (primary): official project from `vllm-project`, hits any OpenAI-compatible server, measures TTFT / ITL (TPOT) / throughput / latency. Lower setup risk, chosen as the pipeline's foundation.
- **AIPerf** (secondary / nice-to-have): NVIDIA's official successor to GenAI-Perf. Adds GPU telemetry (`--gpu-telemetry`) that could be a visual plus for the video once we're on the real unit with an NVIDIA GPU. On Mac (Apple Silicon, no `nvidia-smi`) that part doesn't apply — its value is only evaluated once we have NVIDIA hardware.

**Phase 0 gate:** no moving on to the concurrent chatbot or RAG until `ollama serve` + GuideLLM correctly measuring a small model works repeatably, documented and automated via script.

## 4. Models

### Phase 0 — smoke test (Mac, now)

- **Llama 3 8B** via Ollama. Sole objective: confirm the benchmark software measures correctly, not evaluate real performance.
- **Qwen3 4B** (Q4_K_M quantization) added as a second model, from a different family, so the dashboard's comparison table has at least 2 models — see [RESULTS.md](RESULTS.md). The rest of the candidate list below is still pending testing on the real unit.

### Target list for the real unit (subject to hardware specs, pending)

| Family | Candidates | Status |
|---|---|---|
| Llama | Llama 3.1 8B / 70B | Reference for comparison |
| Mistral | Mistral Small / Ministral | Requested by Nicolas |
| GLM | GLM-4.5-Air | Requested by Nicolas |
| Qwen | Qwen3 8B / 32B | Reference for comparison |
| Inkling-Small (not "Inkling 2") | Thinking Machines Lab, 276B total / 12B active MoE, Apache 2.0, public weights on HF | **Requires ~600GB aggregate VRAM in BF16** — a candidate only if we confirm multi-GPU hardware at that scale. Otherwise it's dropped or an aggressive quantization is evaluated (unclear whether vLLM supports it well for this model). |

The full list isn't run on every unit: 3-4 representative models based on available VRAM, always documenting the exact configuration (quantization, context window, GPU, VRAM used) so comparisons are valid.

## 5. Workloads to demonstrate (in implementation order)

1. **Raw inference performance** (Phase 0 gate): ~1000 input tokens prompt, 500 output tokens, concurrency 1 → 4 → 8 users. Metrics: TTFT, output TPS, ITL/TPOT, request latency, throughput, (GPU util/VRAM where applicable).
2. **Concurrent chatbot**: Open WebUI connected to the model, several simulated users with different kinds of query simultaneously. Same instrumented backend, not just visual.
3. **RAG / document processing**: small document set → embeddings → vector DB (Qdrant) → retrieval → generation. Measure retrieval and generation **separately** so the pipeline's time isn't attributed to the model.

## 6. Proposed repo structure

```
brain-demo/
├── docker-compose.yml
├── .env.example
├── scripts/
│   ├── setup.sh
│   ├── download_models.sh
│   ├── start_inference_ollama.sh      # Phase 0 (Mac)
│   ├── start_inference_vllm.sh        # Real hardware — untested, see §9e
│   ├── run_benchmark.sh        # guidellm (primary) / aiperf (optional)
│   └── collect_metrics.sh
├── benchmarks/
│   ├── inference/
│   ├── chatbot/
│   └── rag/
├── datasets/
│   ├── prompts.jsonl
│   └── documents/
├── configs/
│   ├── models.yaml
│   └── benchmark.yaml
└── results/
```

## 7. Phase plan

- **Phase 0 (now):** Ollama + Llama 3 8B on Mac + GuideLLM. Validate end-to-end that the pipeline measures correctly. **We are here.**
- **Phase 1:** Formalize scripts/configs, standardized prompt dataset, structured reporting in `results/`.
- **Phase 2:** Port the same pipeline to vLLM (once there's access to real NVIDIA hardware or a test cloud GPU).
- **Phase 3:** Run the target model list on the real unit within the 4h window.
- **Phase 4:** Concurrent chatbot and RAG (only once Phases 0-3 are solid).
- **Phase 5:** Video recording and editing.

## 8. Open questions for Nicolas (unresolved)

**Priority 1 — Hardware**
1. Specs of each Brain: GPU (model/count), total VRAM, CPU, RAM, storage.
2. Are all units identical?
3. NVIDIA GPUs? What architecture?
4. Root/sudo access?

**Priority 2 — Access**
5. Do they come with Linux installed?
6. Is there internet during the 4h?
7. Can we download from Hugging Face live, or does it need to be pre-cached?
8. Can we bring a pre-built Docker image?
9. Can we prepare/install before the recording window?
10. Are results kept afterward?

**Priority 3 — Scope**
11. How many units are being tested?
12. 4h per unit, or 4h total?
13. Compare units against each other, or just show individual capabilities?
14. Is there interest in testing giant models even if they don't run at interactive speed?
15. Runtime preference (vLLM/Ollama/SGLang)?
16. Is the goal maximum possible throughput, or a realistic usage experience? (these aren't the same)

## 9a. Test 2 — concurrent chatbot (implemented and validated on Mac)

- `docker-compose.yml` + `scripts/start_chatbot_ui.sh` bring up **Open WebUI** connected to Ollama (`http://localhost:3000`, no login). This is the visual layer for recording.
- `benchmarks/chatbot/simulate_users.py` + `scripts/run_chatbot_test.sh` generate the actual load: 4 simultaneous "users" (general, code, summarization, general knowledge — the 4 personas Nicolas asked for), hitting **Ollama directly** (not through Open WebUI) to get real instrumented metrics. Streams tokens live to the terminal, prefixed by user — good for recording, though it's a bit "noisy" from the per-token prefix; if a cleaner look is wanted for the video, it could be adjusted to prefix only at the start of each message.
- Saves a JSON per run at `results/<timestamp>_chatbot_<model>.json` with TTFT, total latency, and approximate tokens/sec per user.
- **Important finding**: with 4 simultaneous users, TTFT spiked (from ~6s to ~48s depending on the user) because **Ollama serializes generations against the same model by default** instead of truly parallelizing them — what's measured today on Mac is more of a queue than real compute-level concurrency.

**`OLLAMA_NUM_PARALLEL` tested and confirmed** (`scripts/start_inference_ollama.sh` automatically restarts Ollama with this variable, since it only takes effect at process start):

| `OLLAMA_NUM_PARALLEL` | TTFT for users with a free slot | TTFT for queued user(s) | Total duration (4 users) |
|---|---|---|---|
| 1 (real default) | — | up to 48s (the last one waits for the other 3) | 53.9s |
| 2 | ~0.46s (2 users) | 23-35s (2 users) | 38.7s |
| 3 | ~0.81s (3 users) | 13.9s (1 user) | 33.3s |

With `streams=3` it's very clear in the video: 3 of the 4 users start almost instantly and don't wait on each other; only the 4th waits for a slot to free up. Left fixed at `OLLAMA_NUM_PARALLEL=3` in `.env` for this phase. **Important**: aggregate throughput (tokens/sec summed across all) doesn't improve much between levels — what improves is that the first N users no longer wait through a full queue. This is consistent with expectations: on this Mac, compute is the limit, not scheduling. On the real unit with vLLM this number can be raised much further and should scale differently, because there GPU compute headroom exists for aggregate throughput to rise too, not just per-user latency.

## 9b. Test 3 — RAG (in progress)

- **Vector database**: Qdrant via Docker Compose (`scripts/start_vector_db.sh`), no auth for this local demo. Dashboard at `http://localhost:6333/dashboard`.
- **Embeddings**: `nomic-embed-text` via Ollama (768 dims) — deliberately decoupled from the generation LLM: switching the generation LLM (Llama/Mistral/GLM/Qwen) does **not** require re-embedding or touching the collection, because the vector space is defined by the embedding model, not the LLM. If at some point we want to compare *embedding* models against each other, that does require one collection per model (re-ingesting, which takes seconds with this corpus).
- **Corpus**: real Kubernetes documentation (a sample of 398 chunks / 62 pages, with 485 verified questions) — see [rag_corpus.md](rag_corpus.md) for full detail.
- **Ingestion**: `scripts/load_rag_documents.sh` / `benchmarks/rag/ingest.py`, with two modes: `--documents-dir` (own chunking, for loose documents) and `--jsonl-file` (already-chunked corpus, like the Kubernetes one). Validated: 398 chunks embedded + uploaded to Qdrant in ~11s on this Mac.
- **`query.py` implemented and validated** (retrieval + generation, concurrency configurable via an asyncio semaphore). Runs of 1/2/3 sequential queries and 2/3/4 queries at concurrency 2 — results and metrics in [RESULTS.md](RESULTS.md). Finding: retrieval (embedding + Qdrant) is negligible once warm (~0.04-0.17s); the bottleneck is LLM generation, same as the other tests. Important: the session's first run pays a cold-load cost for the embedding model (~22s) — on the real unit it's worth adding an embedding-model warm-up, same as is already done for the LLM.
- **Metrics scoped to what matters**: the decision was to measure only retrieval vs. generation (and now token consumption and error rate) — chunking/embedding aren't broken out as separate stages, because that's measured once at ingestion and isn't part of each query's hot path.
- **Exact token counting, portable to vLLM**: `query.py` and `ingest.py` were updated again to use *only* the OpenAI-compatible API (`/v1/chat/completions` with `stream_options.include_usage`, `/v1/embeddings`) instead of Ollama's native endpoints (`/api/chat`, `/api/embeddings`) — both Ollama and vLLM populate the standard `usage` field with exact `prompt_tokens`/`completion_tokens`, so this removes the last Ollama-specific code path in the pipeline. A new `--embedding-host` (separate from `--ollama-host`) lets embeddings run on a different server than generation, since vLLM typically serves one model per process — see README.md §7.2 for the "100% vLLM" setup (two vLLM instances) vs. the hybrid one (vLLM for generation, Ollama for embeddings).
- **Pending (portability to real hardware)**: use Qdrant's native snapshot (`POST /collections/k8s_docs/snapshots`) to bring the already-loaded collection to the real unit without having to re-ingest within the 4h window. Alternative: a Docker image with the storage volume already populated.

## 9c. Resource monitoring (CPU/GPU) — added to all 3 tests

`scripts/monitor_resources.py` runs automatically in the background during each test (the 3 `run_*.sh` scripts start and stop it on their own) and **auto-detects** what to monitor: if it finds `nvidia-smi` on the PATH, it measures utilization, VRAM, **power draw, and temperature** per GPU (what will happen on the real unit); otherwise (as on this Mac), it falls back to CPU/memory via `psutil` (no power/temperature — there's no portable way to measure those on Mac without tools that require sudo). Same CSV schema either way, so nothing needs to change when moving from Mac to the real unit — it will simply start reporting GPU data because `nvidia-smi` will be available there. Validated across all 3 tests; the file is saved alongside each run (`resource_usage.csv` in Test 1, `..._resources.csv` in Test 2/3).

## 9d. Error rate — added to all 3 tests

Previously, if a single request failed (timeout, connection, insufficient server memory), the whole test would crash. Now Test 2 and Test 3 capture errors per request individually (without bringing down the rest) and classify them heuristically (`connection_error`, `timeout`, `http_<code>_possible_oom` if the response body mentions memory, or other) — specifically useful for when the hardware gets pushed to its limit with large models or heavy concurrency. Test 1 already had this from GuideLLM (`request_totals.errored`); it just needed surfacing, and it's now in `results/summary.csv` and the dashboard.

## 9e. `start_inference_vllm.sh` — script for real hardware (untested)

`start_inference.sh` was split into two scripts, one per runtime: `start_inference_ollama.sh` (the usual one, Phase 0/Mac) and `start_inference_vllm.sh` (new, for once we have the real unit). The vLLM one conceptually does the same thing as the Ollama one — download the model, start the server, warm it up — but it was never possible to test it because **vLLM needs an NVIDIA GPU and this Mac doesn't have one**. It was built anyway, because having it written (even if it needs adjusting on the spot) beats starting from zero within the 4h window.

Substantive differences from the Ollama one, documented in the script itself:
- **Model source**: takes a Hugging Face repo id (`VLLM_MODEL`), not an Ollama tag. The download is an explicit, timed step (`huggingface-cli download`) before starting the server, so download time isn't mixed with startup time inside the 4h window.
- **Separate venv** (`.venv-vllm`): vLLM brings heavy, CUDA-specific dependencies; it was isolated from GuideLLM's venv to avoid conflicts.
- **Config via vLLM's own flags** (`--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--dtype`, `--quantization`), exposed as environment variables (`VLLM_*` in `.env.example`).
- **No direct equivalent to `OLLAMA_NUM_PARALLEL`**: vLLM does continuous batching on its own; `VLLM_MAX_NUM_SEQS` is the closest thing, but it needs to be confirmed on the real unit whether it needs tuning or the default behavior is already good.
- **Reuses vLLM's real health check** (`/health`), unlike Ollama which doesn't have one (hence `validate_backend=false` in GuideLLM) — to be confirmed whether that flag is still needed against vLLM.

**Important**: since it couldn't be tested, it should be treated as a first draft to debug live, not as something validated — the same approach we took with GuideLLM's CLI, which turned out to differ from what its documentation showed.

## 9. Phase 0 — status and lessons from the smoke test (validated)

The Ollama + GuideLLM pipeline was validated on Mac with `llama3:8b`: connection, synthetic load generation, tokenization, and metric computation (TTFT, ITL, TPOT, latency, throughput) work end-to-end. Details relevant for when it's ported to vLLM on the real unit:

- GuideLLM 0.8.1's real CLI is `guidellm run` (not `guidellm benchmark run`, which is what several outdated tutorials/blog posts show).
- `--backend target=...` must be the server's base URL **without** `/v1` — GuideLLM appends `/v1/chat/completions` on its own.
- Ollama doesn't implement `/health` (that's vLLM-specific), so `validate_backend=false` is needed against Ollama. Against vLLM on the real unit, this flag probably won't be needed.
- The `--model` the backend receives is the identifier the server expects (e.g. the Ollama tag), but GuideLLM *also* needs an explicit `--tokenizer kind=huggingface_auto,model=<hf-repo>`, to count tokens, because it tries to resolve the tokenizer as a Hugging Face repo and a tag like `llama3:8b` isn't valid. For each model in the real list, its runtime tag will need to be mapped to the correct HF tokenizer repo.
- The `sweep` profile (intended for the real Test 1 with multiple concurrency levels) failed on Mac because some of its auto-derived rates were too slow to complete even one request within the configured `max_duration` — a slow-hardware problem, not a harness problem. **Fixed for Phase 0**: `sweep` was replaced with `--profile kind=concurrent,streams=N` with a fixed `N` from `BENCHMARK_CONCURRENCY` in `.env` (currently 2). This stops GuideLLM from deriving automatic rates and lets you manually choose the levels to test (1, 2, 4, 8...). Confirmed with `streams=2` on Mac: measured real concurrency 2.0, and TTFT rose from ~900ms (1 user) to ~9.7s (2 users) — Ollama on CPU/Metal doesn't scale well with concurrency, which is exactly the kind of degradation the demo wants to show. For the real unit (vLLM + GPU) this needs to be repeated with higher streams (4, 8) to see whether `sweep` fares better with more throughput available, or whether to keep setting levels manually.
- Metrics confirmed in `benchmarks.json`: `metrics.<name>.successful.<mean|median|...>` for `time_to_first_token_ms`, `inter_token_latency_ms`, `time_per_output_token_ms`, `request_latency`, `requests_per_second`, `output_tokens_per_second`.

## 10. What "Inkling 2" is (clarified)

The correct name is **Inkling-Small** (not "Inkling 2"), Thinking Machines Lab's second model. 276B total parameters / 12B active (MoE), public weights on Hugging Face under Apache 2.0, released ~July 2026. The real blocker isn't weight availability (they're public) but size: ~600GB of aggregate VRAM in BF16. It stays as a conditional candidate pending confirmation of hardware at that scale.
