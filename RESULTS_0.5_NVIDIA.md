# Results — Phase 0.5 (real NVIDIA GPU test server)

**Status**: Test 1 and Test 2 complete. Test 3 (RAG) pending.

This is **not the final Brain hardware** — it's a test server with real NVIDIA GPUs, used to validate the vLLM pipeline on actual GPU hardware before the 4h window with the real unit. Phase 0 (Mac, Ollama) validated the *pipeline*; this phase validates *real GPU behavior*: concurrency scaling, multi-GPU tensor parallelism, and where the hardware actually saturates.

## Setup

| | |
|---|---|
| Hardware | 2x NVIDIA RTX 4070 SUPER (12GB VRAM each), driver 560.35.03 |
| OS | Ubuntu 24.04.1 |
| Inference runtime | vLLM 0.19.1 (pinned — see "Fixes" below) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (small model, chosen to validate the pipeline quickly — not representative of a production-size model) |
| Embeddings (Test 3) | Ollama + `nomic-embed-text`, hybrid setup (generation on vLLM, embeddings on Ollama — see README §7.2) |
| Benchmark tool (Test 1) | GuideLLM 0.8.1, same as Phase 0 |

---

## Test 1 — Raw inference performance (GuideLLM)

Synthetic load, ~550 input tokens / 150 output tokens per request, 100 requests per concurrency level (300 for the final concurrency=128 point, so it genuinely reaches that many requests in flight — see "Bugs found" below).

### 1 GPU (`VLLM_TENSOR_PARALLEL_SIZE=1`)

| Concurrency | Duration | Req/s | Output tok/s | TTFT (median) | GPU0 util / VRAM at steady state |
|---|---|---|---|---|---|
| 1 | 114.2s | 0.88 | 131.3 | 35ms | — |
| 2 | 59.1s | 1.69 | 253.7 | 22ms | — |
| 4 | 30.2s | 3.32 | 497.4 | 23ms | — |
| 8 | 19.4s | 5.15 | 772.3 | 30ms | — |
| 16 | 9.1s | 11.02 | 1652.5 | 38ms | — |
| 32 | 5.8s | 17.39 | 2608.3 | 48ms | — |
| 64 | 3.7s | 25.85 | **4038.1** | 77ms | **100% util, 11.7/12.3GB VRAM** |

At concurrency 64, GPU0 hit **100% utilization and ~95% of VRAM** — this is the real ceiling for this model on 1 GPU. GPU1 was completely idle (0%) the whole time, since 1-GPU mode doesn't touch it.

### 2 GPUs (`VLLM_TENSOR_PARALLEL_SIZE=2`)

| Concurrency | Duration | Req/s | Output tok/s | TTFT (median) | vs. 1-GPU output tok/s |
|---|---|---|---|---|---|
| 1 | 84.3s | 1.19 | 178.0 | 49ms | **+35.6%** |
| 2 | 43.0s | 2.33 | 348.8 | 20ms | **+37.5%** |
| 4 | 22.5s | 4.45 | 667.8 | 22ms | **+34.3%** |
| 8 | 16.2s | 6.16 | 924.3 | 28ms | **+19.7%** |
| 16 | 7.9s | 12.71 | 1907.2 | 39ms | **+15.4%** |
| 32 | 5.7s | 17.60 | 2640.0 | 49ms | **+1.2%** |
| 64 | 4.4s | 22.88 | 3431.5 | 82ms | **-15.0%** (worse than 1 GPU) |
| 128 (300 req) | 10.5s | 28.57 | **4300.5** | 125ms | +6.5% vs. 1-GPU's best (64), but needed *double* the concurrency and *both* GPUs to get there |

Both GPUs reached 99-100% utilization at concurrency 128.

### Reading — this is the most important finding of this phase

**Tensor parallelism (2 GPUs) only pays off at low-to-medium concurrency, for a model this small.** Below ~16 concurrent requests, splitting the model across 2 GPUs gives a real 15-38% throughput boost — each individual request's matmuls get split and computed in parallel, and the per-token NCCL synchronization cost between GPUs is small relative to the compute each request needs. But once the 1-GPU setup is already compute-saturated via batching (concurrency ≥32), the picture flips: 2-GPU mode plateaus or even gets **worse** (-15% at concurrency 64), because every token generated for every request in the batch now pays a fixed NCCL all-gather cost between the 2 GPUs, and for a 1.5B model that communication overhead is no longer negligible next to the (now parallelized, so individually smaller) compute per GPU.

**Practical implication for the real Brain unit**: tensor parallelism across multiple GPUs is a lever worth using when the model doesn't fit on a single GPU (which is the normal reason to use it), or when per-request latency at low concurrency matters more than aggregate throughput. It is **not automatically a throughput win** once the workload is batched and compute-saturated — this needs to be re-measured with whatever model size actually ships on the Brain unit, since a bigger model would shift where this crossover happens (more compute per GPU per token relative to the fixed communication cost).

---

## Test 2 — Concurrent chatbot (simulated users, direct to vLLM)

Fixed-size prompts (~general/code/summarization/knowledge categories), all fired simultaneously, same vLLM server (2-GPU/TP2 config, same one used for Test 1's 128-concurrency point).

| Users | Duration (wall clock) | TTFT (mean / median) | Latency (mean / median) | Output tok/s (mean, per user, approx) | Errors |
|---|---|---|---|---|---|
| 4 | 3.25s | 0.04s / 0.04s | 1.98s / 1.97s | 123.8 | 0% |
| 32 | 5.48s | 0.09s / 0.09s | 3.09s / 3.24s | 78.8 | 0% |
| 128 | 12.07s | 0.94s / 0.16s | 7.28s / 7.24s | 34.5 | 0% |
| 256 | 21.90s | 5.74s / 5.14s | 12.74s / 13.35s | 22.5 | 0% |

GPU utilization during the 256-user run: both GPUs at 99-100% almost the entire time (same saturation signature as Test 1's high-concurrency runs).

**Reading**: 0% error rate at every level tested — vLLM's continuous batching queues gracefully instead of failing outright, even at 256 simultaneous users on a 1.5B model + 2 small GPUs. But TTFT degrades sharply past ~64-128 users (0.09s at 32 users → 5.74s mean at 256) as the scheduler's queue grows — this is the real-world version of what Test 1's concurrency sweep showed synthetically. For the real Brain unit, whatever concurrent-user target Nicolás cares about should be checked against this kind of curve, not just "does it error or not."

---

## Test 3 — RAG (retrieval + generation)

**Pending** — not yet run on this server. Will be added once executed (embedding via Ollama/`nomic-embed-text`, generation via the same vLLM server, retrieval via Qdrant with the same k8s-docs corpus from Phase 0).

---

## Bugs found and fixed during this phase (all pushed to the repo)

Beyond the vLLM/CUDA driver fixes already validated in Phase 0.5's first pass (see `scripts/start_inference_vllm.sh` header and `PROJECT_PLAN.md` §9e — vLLM pinned to 0.19.1, `CUDA_VISIBLE_DEVICES` UUID normalization), this load-testing pass found:

1. **`run_benchmark.sh` silently ignored concurrency overrides.** Running `BENCHMARK_CONCURRENCY=8 ./scripts/run_benchmark.sh` to sweep levels didn't work — the script's own `source .env` ran *after* the shell export and unconditionally overwrote it back to `.env`'s literal value. Every level in a sweep silently ran at the `.env` default instead of the intended one. Fixed by preserving any pre-set override before sourcing `.env` and restoring it after.
2. **Multi-GPU (`tensor-parallel-size=2`) needs more VRAM headroom than single-GPU.** `VLLM_GPU_MEMORY_UTILIZATION=0.90` (fine for 1 GPU) OOM'd during warm-up with 2 GPUs — each rank's own CUDA context/NCCL overhead pushed it past the card's physical VRAM. `0.80` worked.
3. **A killed vLLM process can leave orphaned GPU-resident workers.** `pkill -f "vllm serve"` only kills the parent process; the actual GPU memory holders (`VLLM::EngineCore`, `VLLM::Worker_TPx`) can survive and keep VRAM allocated, causing the *next* start attempt to OOM immediately even though `nvidia-smi` looked clean moments before. Needs `pkill -9 -f "VLLM::"` too.
4. **GuideLLM's `streams=N` concurrency isn't real if `max_requests` is close to or below N.** With only 100 total requests requested and `streams=128`, GuideLLM can't actually sustain 128 in flight — the reported "Concurrency Mean" came back as ~99, not 128. Needed `BENCHMARK_REQUEST_COUNT` well above the concurrency level being tested (300 for streams=128) to get a real reading.
5. **`simulate_users.py`'s live per-token streaming output is unusable noise past a handful of users.** Fine for the original 4-user "recording" use case, unreadable once pushed to 32+ simulated users (interleaved tokens from dozens of users at once). Now auto-switches to an aggregate-only summary (error rate, mean/median TTFT and latency) above 16 users.

None of these are specific to this test server — they're general script/tooling bugs that would have hit the exact same way on the real Brain hardware, so finding them now (rather than inside the 4h window) was the point of this pass.
