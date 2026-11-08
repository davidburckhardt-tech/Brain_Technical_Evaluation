# Resultados — Fase 0 (smoke test en Mac)

**Importante**: estos resultados son de una Mac (CPU/Metal, sin GPU dedicada), corridos únicamente para validar que el pipeline de benchmark mide correctamente. **No son representativos del rendimiento de las unidades Brain** — eso se mide recién cuando haya acceso al hardware real (ver [PROJECT_PLAN.md](PROJECT_PLAN.md)).

**Entorno**: macOS, Ollama (runtime local). Dos modelos probados, de familias distintas: `llama3:8b` (tokenizer `NousResearch/Meta-Llama-3-8B`) y `qwen3:4b-q4_K_M` (tokenizer `Qwen/Qwen3-4B`, cuantización Q4_K_M). El segundo modelo se agregó específicamente para que la comparación entre modelos tenga al menos 2 puntos de datos — el resto de la lista candidata (Mistral, GLM) queda pendiente de probar en la unidad real.

---

## Test 1 — Raw Inference Performance

Benchmark: GuideLLM, carga sintética (~550 tokens de entrada, ~150 de salida por request).

| Run | Concurrencia (`streams`) | Requests | Duración total | TTFT medio | TTFT mediana | ITL medio | TPOT medio | Latencia media/request | Requests/seg | Output tokens/seg |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 (secuencial) | 5 | 55.6s | 914 ms | 189 ms | 30.1 ms | 32.7 ms | 11.1s | 0.09 | 30.5 |
| 2 | 2 | 10 | 79.9s | 9,759 ms | 10,289 ms | 28.8 ms | 80.1 ms | 15.2s | 0.13 | 23.7 |

**Lectura (llama3:8b)**: al pasar de 1 a 2 streams concurrentes, el TTFT se multiplicó por ~10x (de ~900ms a ~9.8s) y el TPOT casi se duplicó (32.7 → 80.1 ms/token). El throughput agregado (output tokens/seg) bajó levemente en vez de subir. Esto es consistente con un hardware sin margen de cómputo libre: la Mac no tiene capacidad ociosa para absorber una segunda request en paralelo, así que la concurrencia solo reparte el mismo cómputo entre más requests, empeorando la latencia de cada una.

Datos crudos: `results/llama3_8b/20261107_184204_c1/` y `results/llama3_8b/20261107_203726_c2/`.

### Segundo modelo: `qwen3:4b-q4_K_M`

| Run | Concurrencia | Requests completados | Duración total | TTFT medio | TPOT medio | Latencia media/request | Output tokens/seg | Error rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 9/10 | 310.3s | 1,774 ms | 47.9 ms | 64.5s | 39.0 | 0% |

**Lectura**: Qwen3 4B (modelo más chico, 4B vs 8B parámetros, cuantización Q4_K_M) tuvo mayor output tokens/seg que Llama 3 8B en la misma condición de concurrencia (39.0 vs 23.7 tok/s) — esperable por ser un modelo más chico — pero también mayor latencia media por request (64.5s vs 15.2s) y solo completó 9 de 10 requests en el tiempo asignado. No es una comparación perfectamente controlada (distinta arquitectura, cuantización, tamaño); sirve como primer punto de referencia multi-modelo, no como conclusión definitiva — eso requiere la lista completa de modelos en la unidad real.

Datos crudos: `results/qwen3_4b-q4_K_M/20261107_222531_c2/`.

---

## Test 2 — Chatbot concurrente (4 usuarios simulados)

Script: `benchmarks/chatbot/simulate_users.py`, 4 usuarios simultáneos con distinto tipo de consulta (general, generación de código, resumen, conocimiento general), pegándole directo a Ollama. Variable de control: `OLLAMA_NUM_PARALLEL` (cantidad de requests que Ollama procesa en paralelo real vía continuous batching; requiere reiniciar el server para aplicarse).

| `OLLAMA_NUM_PARALLEL` | Usuario | Categoría | TTFT | Latencia total | Output tokens (aprox.) | Tokens/seg (aprox.) |
|---|---|---|---|---|---|---|
| **1** (default) | general-1 | general | 47.99s | 53.84s | 128 | 2.4 |
| | code-1 | código | 6.08s | 13.07s | 160 | 12.2 |
| | summary-1 | resumen | 13.20s | 30.48s | 451 | 14.8 |
| | knowledge-1 | conocimiento | 30.64s | 47.78s | 414 | 8.7 |
| | **Total (4 usuarios)** | | | **53.89s** | | |
| **2** | general-1 | general | 34.94s | 38.61s | 72 | 1.9 |
| | code-1 | código | 23.33s | 37.55s | 168 | 4.5 |
| | summary-1 | resumen | 0.46s | 23.13s | 310 | 13.4 |
| | knowledge-1 | conocimiento | 0.47s | 34.68s | 465 | 13.4 |
| | **Total (4 usuarios)** | | | **38.66s** | | |
| **3** (valor dejado en `.env`) | general-1 | general | 13.94s | 19.35s | 81 | 4.2 |
| | code-1 | código | 0.81s | 13.67s | 138 | 10.1 |
| | summary-1 | resumen | 0.81s | 30.19s | 372 | 12.3 |
| | knowledge-1 | conocimiento | 0.81s | 33.24s | 485 | 14.6 |
| | **Total (4 usuarios)** | | | **33.31s** | | |

**Lectura**:
- Con `NUM_PARALLEL=1`, Ollama serializa: cada usuario espera a que el anterior termine por completo (por eso el último usuario en entrar, `general-1`, tiene TTFT de 48s).
- Con `NUM_PARALLEL=2`, 2 usuarios (`summary-1`, `knowledge-1`) entran casi al instante (TTFT ~0.46s); los otros 2 esperan un slot libre.
- Con `NUM_PARALLEL=3`, 3 usuarios (`code-1`, `summary-1`, `knowledge-1`) entran casi al instante (TTFT ~0.81s) y **no se esperan entre sí**; solo el 4to (`general-1`) espera que se libere un slot (13.9s).
- El tiempo total para los 4 usuarios bajó de 53.9s → 38.7s → 33.3s a medida que subió `NUM_PARALLEL`.
- El throughput agregado (tokens/seg sumando usuarios) **no mejoró significativamente** entre niveles — lo que mejora es que más usuarios dejan de hacer cola completa, no que el hardware rinda más. Es la misma conclusión que el Test 1: en esta Mac el límite es el cómputo, no el scheduling.

Datos crudos: `results/llama3_8b/20261107_204825_chatbot.json` (parallel=1), `results/llama3_8b/20261107_205940_chatbot.json` (parallel=2), `results/llama3_8b/20261107_210047_chatbot.json` (parallel=3).

### Segundo modelo: `qwen3:4b-q4_K_M` (NUM_PARALLEL=3)

| Usuario | Categoría | TTFT | Latencia total | Output tokens | Tokens/seg |
|---|---|---|---|---|---|
| general-1 | general | 12.85s | 17.09s | 54 | 3.2 |
| code-1 | código | 39.70s | 61.83s | 243 | 3.9 |
| summary-1 | resumen | 72.71s | 94.79s | 337 | 3.6 |
| knowledge-1 | conocimiento | 69.60s | 101.96s | 583 | 5.7 |
| **Total (4 usuarios)** | | | **102.04s** | | |

Error rate: 0%. Datos crudos: `results/qwen3_4b-q4_K_M/20261107_223208_chatbot.json`.

**Lectura**: a igual `NUM_PARALLEL=3`, Qwen3 4B tardó 3x más que Llama 3 8B en completar a los 4 usuarios (102s vs 33.3s) y sus respuestas fueron más largas en promedio — el tamaño de la respuesta generada pesa tanto como el tamaño del modelo en este tipo de comparación, otra razón para no sacar conclusiones de "mejor/peor modelo" todavía con una sola corrida cada uno.

---

## Test 3 — RAG (retrieval + generación)

Pipeline: pregunta → embeber (`nomic-embed-text`) → recuperar top-3 chunks de Qdrant (`k8s_docs`) → generar respuesta (`llama3:8b`). Preguntas tomadas de `datasets/k8s_rag/qa_eval_sample.jsonl` (en orden del archivo, no aleatorio). Retrieval y generación se miden por separado a propósito, para no atribuirle al LLM el tiempo de la base vectorial. `OLLAMA_NUM_PARALLEL` seguía en 3 (heredado del Test 2).

| Configuración | Queries | Concurrencia | Duración total (wall clock) | Retrieval medio | Generación media | Chunk esperado encontrado (top-3) |
|---|---|---|---|---|---|---|
| 1 usuario | 1 | 1 | 23.17s* | 22.296s* | 0.81s | 100% |
| 2 secuenciales | 2 | 1 | 3.34s | 0.067s | 1.58s | 100% |
| 3 secuenciales | 3 | 1 | 4.57s | 0.044s | 1.47s | 67% |
| 2 concurrentes (2 hilos) | 2 | 2 | 1.50s | 0.173s | 1.18s | 100% |
| 3 concurrentes (2 hilos) | 3 | 2 | 2.65s | 0.069s | 1.46s | 67% |
| 4 concurrentes (2 hilos) | 4 | 2 | 6.26s | 0.083s | 2.63s | 75% |

\* La primera corrida (1 usuario) tiene un retrieval de 22.3s porque fue la primera vez que se usó el modelo de embeddings en esta sesión de Ollama — ese tiempo es carga en frío del modelo en memoria, no retrieval real. Las corridas siguientes, con el modelo ya cargado, muestran el retrieval real: **0.04-0.17s**, prácticamente insignificante comparado con la generación. **Para la unidad real: conviene hacer un "warm-up" del modelo de embeddings antes de medir**, igual que ya hacemos con el LLM en `start_inference_ollama.sh`.

**Lectura**:
- El retrieval (embeber + buscar en Qdrant) es rapidísimo una vez caliente — nunca es el cuello de botella de este pipeline, ni secuencial ni concurrente. El tiempo lo domina la generación del LLM, como es esperable.
- Con concurrencia 2 (dentro del límite de `OLLAMA_NUM_PARALLEL=3` configurado en el Test 2), no se vio la degradación de cola que sí aparecía en el Test 2 sin slots paralelos — señal de que dejar `OLLAMA_NUM_PARALLEL` en un valor >1 beneficia también a este test.
- El "hit rate" (si el chunk correcto apareció entre los top-3 recuperados) varió entre 67-100% según qué preguntas entraron en la muestra — no es una métrica de carga, es una señal de calidad de retrieval que quedó registrada de paso porque el dataset trae el chunk correcto conocido (`source_chunk`), pero no es el foco de esta ronda (se pidieron métricas de performance, no de calidad).

Datos crudos: `results/llama3_8b/*_rag_q*_c*.json` (6 archivos, uno por configuración).

Nota: estas 6 corridas son de antes de cambiar la medición de tokens a conteo exacto (ver más abajo) — muestran `output_tokens_approx` (aproximado por palabras), no tokens exactos.

### Segundo modelo: `qwen3:4b-q4_K_M`

Con conteo de tokens exacto (`prompt_eval_count`/`eval_count` de la API nativa de Ollama, no aproximado por palabras):

| Configuración | Queries | Concurrencia | Wall clock | Retrieval medio | Generación media | Tokens in/out (prom.) | Error rate | Hit rate top-3 |
|---|---|---|---|---|---|---|---|---|
| 1 usuario | 1 | 1 | 12.67s | 0.942s | 11.68s | 532 / 429 | 0% | 100% |
| 4 concurrentes (2 hilos) | 4 | 2 | 37.12s | 0.165s | 14.69s | 508 / 357 | 0% | 75% |

Datos crudos: `results/qwen3_4b-q4_K_M/20261107_223400_rag_q1_c1.json`, `results/qwen3_4b-q4_K_M/20261107_223413_rag_q4_c2.json`.

**Lectura**: mismo patrón que en Test 1/2 — Qwen3 4B generó bastante más tokens de salida por respuesta (357-429 vs. lo que generó Llama en estas mismas preguntas) y tardó más en generación, mientras que el retrieval (independiente del LLM, mismo embedding/Qdrant) se mantuvo igual de rápido en ambos modelos — confirma que separar retrieval de generación fue la decisión correcta: el retrieval no varía por modelo, la generación sí.

---

## Conclusión de esta fase

Los 3 tests (raw inference, chatbot concurrente, RAG) están validados end-to-end en Mac y listos para portarse a vLLM en la unidad real. La expectativa para la unidad real es que, a diferencia de esta Mac, sí haya margen de cómputo (GPU) para que subir la concurrencia mejore el throughput agregado en vez de solo repartir la cola — eso es justamente lo que hay que volver a medir ahí.
