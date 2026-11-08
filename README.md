# Brain Demo — Benchmark Harness

Pipeline to benchmark LLMs running locally (Ollama today on Mac, vLLM on the
real Brain hardware later) across 3 workloads: raw inference, concurrent
chatbot, and RAG. Built to be scripted end-to-end so that, on the real
hardware, the only manual work left is running these commands in order and
recording results — see [§7](#7-moving-to-the-real-brain-hardware) for what
changes there.

Context and decisions: [PROJECT_PLAN.md](PROJECT_PLAN.md). Metrics measured
so far: [RESULTS.md](RESULTS.md) / `dashboard/index.html`. Corpus used for
RAG: [rag_corpus.md](rag_corpus.md).

## Contents

1. [Prerequisites](#1-prerequisites)
2. [One-time setup](#2-one-time-setup)
3. [Test 1 — raw inference performance](#3-test-1--raw-inference-performance)
4. [Test 2 — concurrent chatbot](#4-test-2--concurrent-chatbot)
5. [Test 3 — RAG](#5-test-3--rag)
5a. [Resource monitoring (CPU/GPU)](#5a-resource-monitoring-cpugpu)
5b. [Error handling](#5b-error-handling)
6. [Dashboard](#6-dashboard)
7. [Moving to the real Brain hardware](#7-moving-to-the-real-brain-hardware)
8. [`.env` reference](#8-env-reference)
9. [Project layout](#9-project-layout)
10. [Known quirks / troubleshooting](#10-known-quirks--troubleshooting)

---

## 1. Prerequisites

- **Ollama** installed and on `PATH` (`ollama --version`). [ollama.com](https://ollama.com)
- **Docker** (Desktop or Engine) running (`docker ps` works without error) — used for Open WebUI and Qdrant.
- **Python 3.11+** (`python3 --version`).
- Internet access for the one-time model/corpus downloads.

### Install commands

**macOS:**
```bash
brew install ollama
brew install --cask docker   # then open Docker.app once so the daemon starts
brew install python@3.11
```

**Linux** (the real Brain hardware is expected to run Linux):
```bash
curl -fsSL https://ollama.com/install.sh | sh

curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # log out/in (or `newgrp docker`) for this to take effect

sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv   # Debian/Ubuntu
```

Verify everything is in place before moving on:
```bash
ollama --version
docker ps
python3 --version
```

## 2. One-time setup

Run once per machine (including the first time on the real hardware):

```bash
cp .env.example .env           # review/adjust values - see §8 for what each one does
./scripts/setup.sh             # creates .venv, installs GuideLLM into it
./scripts/download_models.sh   # ollama pull <MODEL_NAME> (default: llama3:8b)
ollama pull nomic-embed-text   # embedding model, needed for Test 3
```

Then start the inference server (needed before any test):

```bash
./scripts/start_inference_ollama.sh   # starts/warms up Ollama. Restarts it automatically
                                # if OLLAMA_NUM_PARALLEL is set in .env (see §8, §10)
```

At this point Test 1 and Test 2 can run. Test 3 needs its own one-time setup — see [§5](#5-test-3--rag).

## 3. Test 1 — raw inference performance

Synthetic load (~550 input tokens / ~150 output tokens per request) via GuideLLM, against Ollama's OpenAI-compatible API directly.

```bash
./scripts/run_benchmark.sh     # results -> results/<timestamp>_<model>_c<N>/benchmarks.json
./scripts/collect_metrics.sh   # appends the key metrics to results/summary.csv
```

Controlled by `.env`: `BENCHMARK_REQUEST_COUNT` (total requests), `BENCHMARK_CONCURRENCY` (fixed concurrent streams — **not** a "sweep", see [§10](#10-known-quirks--troubleshooting) for why). To test another concurrency level, change `BENCHMARK_CONCURRENCY` and re-run both commands; each run gets its own timestamped folder, so nothing gets overwritten.

Metrics captured: TTFT, TPOT, ITL, request latency, requests/sec, output tokens/sec.

## 4. Test 2 — concurrent chatbot

Fires 4 simulated users (general question, code generation, summarization, general knowledge — from `datasets/prompts.jsonl`) at Ollama **simultaneously**, streaming tokens live to the terminal (useful for recording). Open WebUI is a separate, purely visual layer — it is not what gets measured.

```bash
./scripts/start_chatbot_ui.sh   # starts Open WebUI (Docker) at http://localhost:3000, connected to Ollama
./scripts/run_chatbot_test.sh   # the actual measured run -> results/<timestamp>_chatbot_<model>.json
```

`OLLAMA_NUM_PARALLEL` in `.env` controls how many of those 4 users Ollama actually processes at once (vs. queueing). **Changing it requires restarting Ollama** — re-run `./scripts/start_inference_ollama.sh` after changing it, then `./scripts/run_chatbot_test.sh` again. See [RESULTS.md](RESULTS.md) for what each level (1/2/3) looked like on Mac.

## 5. Test 3 — RAG

Pipeline: question → embed (`nomic-embed-text`) → retrieve top-k chunks from Qdrant → generate answer (the model in `MODEL_NAME`) → retrieval time and generation time are recorded **separately**.

**One-time setup** (per machine):

```bash
./scripts/start_vector_db.sh      # starts Qdrant (Docker) at http://localhost:6333, dashboard at /dashboard
./scripts/load_rag_documents.sh   # embeds + loads datasets/k8s_rag/corpus_sample.jsonl into Qdrant (~11s for 398 chunks on Mac)
```

**Running queries** (repeat with different arguments to vary load):

```bash
./scripts/run_rag_test.sh <num_queries> <concurrency>
# e.g.:
./scripts/run_rag_test.sh 1 1    # 1 user
./scripts/run_rag_test.sh 3 1    # 3 sequential
./scripts/run_rag_test.sh 4 2    # 4 queries, 2 concurrent at a time
```

Results: `results/<timestamp>_rag_q<N>_c<C>.json` — per-query retrieval time, generation TTFT/duration, and whether the known-correct chunk (from the QA dataset) was retrieved in the top-k (a quality signal that comes along for free, not the focus of these runs).

**First query of a session will look slow** — the embedding model's first call pays a cold-load cost (~22s measured on Mac), separate from actual retrieval. See [§10](#10-known-quirks--troubleshooting).

The corpus itself (what it is, how it was sampled, schema) is documented in [rag_corpus.md](rag_corpus.md).

## 5a. Resource monitoring (CPU/GPU)

Every `run_*.sh` script (Test 1, 2, 3) automatically starts `scripts/monitor_resources.py` in the background for the duration of the test, and stops it when the test finishes. It **auto-detects** what to monitor:

- If `nvidia-smi` is found on `PATH` (real Brain hardware, NVIDIA GPUs): samples GPU utilization %, VRAM used/total, power draw, and temperature, per GPU.
- Otherwise (this Mac today): falls back to CPU utilization % and system memory via `psutil` (no power/temperature — not portably available without sudo-gated tools).

Same CSV schema either way (`timestamp_iso, elapsed_s, mode, device, util_percent, memory_used_mb, memory_total_mb, power_draw_w, temperature_c`), saved alongside each run's results — `resource_usage.csv` inside Test 1's run folder, `<timestamp>_..._resources.csv` next to Test 2/3's JSON. Nothing to configure; it just works on whichever hardware it's run on.

## 5b. Error handling

Test 2 and Test 3 catch failures per-request instead of letting one failure crash the whole run: a failed request is recorded with a best-effort reason (`connection_error`, `timeout`, `http_<code>_possible_oom` if the response body mentions memory, etc.), and the run continues with the rest. Each summary JSON includes an `error_rate`. Test 1 gets this from GuideLLM directly (`request_totals`), surfaced in `results/summary.csv` and the dashboard.

## 6. Dashboard

```bash
./scripts/build_dashboard.py   # rebuilds dashboard/index.html from everything under results/
```

Self-contained HTML (data embedded inline, no server, nothing sent anywhere — these are potentially sensitive benchmark numbers) with one tab per model tested, each showing all 3 tests' results as tables + simple bar charts. Re-run any time after new results land; open `dashboard/index.html` directly in a browser, no server needed.

### Viewing it from another device (e.g. the real Brain unit, terminal-only)

```bash
./scripts/share_dashboard.sh   # rebuilds the dashboard, serves it locally, and tunnels it with ngrok
```

Prints a public `https://...ngrok-free.dev` URL that can be opened from a phone, tablet, or any other device on a different network — useful for recording the results on screen when the machine running the tests has no GUI/browser of its own (the terminal-only scenario on the real hardware). Needs `ngrok` installed and authenticated once (`brew install ngrok`, then `ngrok config add-authtoken <token>` from a free [ngrok.com](https://ngrok.com) account). Ctrl+C stops both the tunnel and the local server; the URL changes every time it's run (ngrok's free tier doesn't keep a fixed subdomain) — **anyone with that URL can view the dashboard while it's running**, since this is a convenience tunnel with no auth of its own, so don't leave it running longer than needed and don't post the link publicly. Override the local port with `DASHBOARD_PORT` in `.env` (default `8080`).

The same `ngrok http <port>` approach works for Open WebUI (`ngrok http 3000`) if the chatbot UI itself also needs to be viewed remotely for recording.

## 7. Moving to the real Brain hardware

The whole pipeline (GuideLLM, the chatbot simulator, and RAG's `ingest.py`/`query.py`) talks only the OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`) — no Ollama-native endpoint is used anywhere in the code anymore. That means switching from Ollama to vLLM is, in principle, just pointing the same scripts at a different URL. This section gives the actual commands for that; §7.3 covers what's genuinely different about vLLM (not just the URL) and what's still unvalidated.

**Everything in this section is UNTESTED** — written without access to an NVIDIA GPU (vLLM doesn't run on this Mac). Treat it as a first draft to debug on the real hardware, not a validated path, the same way GuideLLM's own CLI turned out to differ from its docs (see [PROJECT_PLAN.md](PROJECT_PLAN.md) §9).

### 7.1. One-time setup (vLLM)

```bash
# In .env: set the model as a Hugging Face repo id, NOT an Ollama tag
#   VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
# For gated models (e.g. Llama), also set HF_TOKEN=<your token> in .env.

./scripts/start_inference_vllm.sh   # creates .venv-vllm, installs vllm, downloads
                                     # the model from HF, starts `vllm serve`, warms it up
```

This installs vLLM into its own virtualenv (`.venv-vllm`, separate from `.venv`) the first time it runs, since vLLM's CUDA-specific dependencies can conflict with GuideLLM's. See `scripts/start_inference_vllm.sh` and the `VLLM_*` variables in `.env.example` for all the knobs (tensor parallel size, GPU memory fraction, max model length, quantization, extra flags).

### 7.2. Running the 3 tests against vLLM

Same commands as the rest of this README (§3-§5) — the only thing that changes is what `OLLAMA_HOST`/`BENCHMARK_TARGET` point at in `.env`:

```bash
# In .env, point generation at vLLM instead of Ollama:
#   OLLAMA_HOST=http://localhost:8000      (vLLM's default port)
#   BENCHMARK_TARGET=http://localhost:8000
#   MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct   (same HF repo id as VLLM_MODEL)
#   TOKENIZER_MODEL=meta-llama/Llama-3.1-8B-Instruct  (same repo - no tag mismatch
#                                                       problem like with Ollama tags)

./scripts/run_benchmark.sh && ./scripts/collect_metrics.sh   # Test 1, unchanged
./scripts/run_chatbot_test.sh                                 # Test 2, unchanged
```

**Test 3 (RAG) needs a decision about embeddings**, because vLLM typically serves one model per process — it can't serve both the generation model and an embedding model from the same `vllm serve` instance. Two options:

- **Hybrid (simplest)**: keep Ollama running just for embeddings (`ollama serve` + `nomic-embed-text`, as today), and point generation at vLLM. This is not "100% vLLM", but it's the least new moving parts:
  ```bash
  # .env: generation via vLLM, embeddings stay on Ollama
  #   OLLAMA_HOST=http://localhost:8000       (vLLM, used for generation by query.py)
  #   EMBEDDING_HOST=http://localhost:11434   (Ollama, used for embeddings)
  ./scripts/start_vector_db.sh && ./scripts/load_rag_documents.sh
  ./scripts/run_rag_test.sh 4 2
  ```
- **100% vLLM**: run a *second* `vllm serve` process, on a different port, loaded with an embedding model (vLLM supports embedding models via the same `/v1/embeddings` endpoint):
  ```bash
  # A second instance of start_inference_vllm.sh, different port/model/venv-safe:
  VLLM_MODEL=nomic-ai/nomic-embed-text-v1.5 VLLM_PORT=8001 ./scripts/start_inference_vllm.sh

  # .env:
  #   OLLAMA_HOST=http://localhost:8000      (generation vLLM)
  #   EMBEDDING_HOST=http://localhost:8001   (embedding vLLM)
  #   EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
  ./scripts/start_vector_db.sh && ./scripts/load_rag_documents.sh
  ./scripts/run_rag_test.sh 4 2
  ```
  This needs two GPUs (or enough VRAM on one to hold both models) and has not been tried at all — the embedding model's exact HF repo id, and whether vLLM's embedding support behaves the way this assumes, both need to be verified on the real hardware.

### 7.3. What's genuinely different about vLLM (not just the URL)

- **Model source**: vLLM loads from Hugging Face repo ids (safetensors), not Ollama tags/GGUF. Every model in the candidate list needs its actual HF repo id before testing.
- **No `OLLAMA_NUM_PARALLEL` equivalent**: vLLM does continuous batching on its own; `VLLM_MAX_NUM_SEQS` is the closest manual knob, but it's unconfirmed whether it needs tuning at all, unlike Ollama where it made a dramatic difference (§4).
- **`validate_backend=false`** (needed for GuideLLM against Ollama, which has no `/health`) probably isn't needed against vLLM, which does have one — worth trying without it first.
- **`sweep` profile for Test 1**: untested whether it works better on real hardware — it failed on Mac because some auto-derived rates were too slow to complete a request in time (see [PROJECT_PLAN.md](PROJECT_PLAN.md) §9). With more throughput available it might just work; if not, keep using fixed `BENCHMARK_CONCURRENCY` levels (1, 2, 4, 8...).
- **Qdrant / RAG corpus**: portable as-is once ingested — see [PROJECT_PLAN.md](PROJECT_PLAN.md) §9b for the snapshot plan to avoid re-ingesting inside the 4h window. Switching *embedding* models (not generation models) means re-ingesting (fast — ~11s for this corpus) because that's what defines the vector space.
- **Models to test**: see [PROJECT_PLAN.md](PROJECT_PLAN.md) §4 for the candidate list (Llama, Mistral, GLM, Qwen; Inkling-Small conditional on VRAM).
- **16 open questions about the hardware itself** (specs, access, scope) are listed in [PROJECT_PLAN.md](PROJECT_PLAN.md) §8 — still unanswered as of this writing; answering them first will likely change some of the above, including which of the two RAG options in §7.2 is even feasible (depends on GPU count/VRAM).

## 8. `.env` reference

Copy `.env.example` to `.env` and adjust. All variables:

| Variable | Used by | Meaning |
|---|---|---|
| `OLLAMA_HOST` | all scripts | Base URL of the **generation** server, OpenAI-compatible (Ollama today; vLLM's URL once switched — see §7) |
| `MODEL_NAME` | Test 1, 2 | Model id as the server knows it (an Ollama tag today; a Hugging Face repo id once on vLLM) |
| `BENCHMARK_TARGET` | Test 1 | Same as `OLLAMA_HOST` conceptually — GuideLLM's backend target (no `/v1` suffix) |
| `TOKENIZER_MODEL` | Test 1 | HF repo id GuideLLM uses to count tokens — **must match the model family**, since `MODEL_NAME` itself usually isn't a valid HF repo id (unless already on vLLM, where it already is one) |
| `BENCHMARK_REQUEST_COUNT` | Test 1 | Total requests per run |
| `BENCHMARK_CONCURRENCY` | Test 1 | Fixed concurrent streams (not a sweep — see §10) |
| `EMBEDDING_MODEL` | Test 3 | Embedding model id, served by `EMBEDDING_HOST` |
| `EMBEDDING_HOST` | Test 3 | OpenAI-compatible server for embeddings. Defaults to `OLLAMA_HOST` when empty — set it separately when generation and embeddings run on different servers (e.g. vLLM + vLLM, or vLLM + Ollama — see §7.2) |
| `RAG_CORPUS_FILE` | Test 3 | Corpus to load (jsonl of pre-chunked passages, or a `--documents-dir` directory) |
| `RAG_COLLECTION` | Test 3 | Qdrant collection name to load the corpus into / query against |
| `OLLAMA_NUM_PARALLEL` | Test 2 (and helps Test 3) | How many requests Ollama processes in parallel. Restarts Ollama when set (see §10). Ollama-only — no effect on vLLM. |
| `VLLM_MODEL` / `VLLM_*` | setup (vLLM) | vLLM server config — see `.env.example` and §7.1 |
| `HF_TOKEN` | setup (vLLM) | Hugging Face token, needed to download gated models (e.g. Llama) |

## 9. Project layout

```
.
├── PROJECT_PLAN.md          # decisions, architecture, open questions, phase roadmap
├── RESULTS.md                # metrics measured so far, by test
├── rag_corpus.md             # RAG corpus: source, sampling, schema
├── docker-compose.yml        # Open WebUI + Qdrant, both pointed at the host's Ollama
├── .env.example               # see §8
├── configs/
│   ├── models.yaml           # model catalog: smoke-test model + target candidates
│   └── benchmark.yaml        # GuideLLM/AIPerf profile notes
├── datasets/
│   ├── prompts.jsonl          # Test 2's 4 simulated-user prompts
│   └── k8s_rag/                # Test 3's corpus + QA eval sample (see rag_corpus.md)
├── benchmarks/
│   ├── chatbot/simulate_users.py   # Test 2 implementation
│   └── rag/{ingest,query}.py        # Test 3 implementation
├── scripts/                   # see table below
├── dashboard/index.html       # generated, see §6
└── results/                   # benchmark output, one subfolder/file per run (gitignored)
```

| Script | Test | Purpose |
|---|---|---|
| `setup.sh` | setup | venv + GuideLLM |
| `download_models.sh` | setup | `ollama pull` the model in `.env` |
| `start_inference_ollama.sh` | setup | start/restart Ollama, warm up the model (Phase 0, Mac) |
| `start_inference_vllm.sh` | setup | download the model from HF, start vLLM, warm up (real hardware — **untested**, see §7) |
| `run_benchmark.sh` / `collect_metrics.sh` | 1 | run + aggregate raw inference benchmark |
| `start_chatbot_ui.sh` / `run_chatbot_test.sh` | 2 | Open WebUI / the actual measured concurrent chat run |
| `start_vector_db.sh` / `load_rag_documents.sh` / `run_rag_test.sh` | 3 | Qdrant / corpus ingestion / retrieval+generation run |
| `build_dashboard.py` | — | rebuild `dashboard/index.html` |
| `share_dashboard.sh` | — | rebuild + serve the dashboard over a temporary public ngrok tunnel |
| `summarize_benchmark.py` | 1 | called by `collect_metrics.sh`, not run directly |
| `monitor_resources.py` | 1, 2, 3 | auto-started/stopped by each `run_*.sh`, not run directly |

## 10. Known quirks / troubleshooting

- **GuideLLM's CLI** (validated against `guidellm==0.8.1`) differs from most tutorials/blog posts: the command is `guidellm run`, not `guidellm benchmark run`; the backend `target` is the server's base URL **without** `/v1` (GuideLLM appends `/v1/chat/completions` itself); and it needs `validate_backend=false` against Ollama, which has no `/health` endpoint (that's vLLM-specific — may not be needed once on vLLM). Full list: [PROJECT_PLAN.md](PROJECT_PLAN.md) §9.
- **Tokenizer vs. model tag**: GuideLLM tries to resolve `--model`'s value as a Hugging Face repo id to load a tokenizer. An Ollama tag like `llama3:8b` isn't one, hence the separate `TOKENIZER_MODEL` setting.
- **`sweep` profile fails on slow hardware**: some of its auto-derived low-rate strategies can't complete a single request within their time budget. Test 1 uses a **fixed** `concurrent,streams=N` profile instead (`BENCHMARK_CONCURRENCY`).
- **`OLLAMA_NUM_PARALLEL` only takes effect at process start.** `start_inference_ollama.sh` handles this by killing and restarting Ollama (including the menu-bar app, if that's how it's running) whenever this variable is set in `.env`.
- **Embedding model cold start**: the first embedding call in a fresh Ollama session pays a one-time model-load cost (~22s measured on Mac) that shows up as "retrieval time" in Test 3's first query. Not a bug — just don't read the very first query's retrieval number as representative. A warm-up step (like the one `start_inference_ollama.sh` already does for the main model) is a good idea to add before trusting Test 3 numbers on new hardware.
- **Qdrant point ids**: `benchmarks/rag/ingest.py` uses sequential integers as Qdrant point ids regardless of input format; the original chunk id (e.g. from the k8s dataset) is preserved in the payload for traceability.
- **Embeddings vs. generation LLM are independent**: switching the generation model (Llama/Mistral/GLM/Qwen) does **not** require re-ingesting the RAG corpus — only switching the *embedding* model does, since that's what defines the vector space.
