# Corpus RAG — Kubernetes Documentation (Fase 0)

## Fuente

[evgenypal/k8s-docs-rag-bench](https://hf-p-cfw.fyan.top/datasets/evgenypal/k8s-docs-rag-bench) en Hugging Face — documentación oficial de Kubernetes, en inglés, licencia **CC-BY-4.0**.

El dataset completo tiene:
- 956 páginas HTML de documentación técnica real.
- 7.908 chunks pre-divididos por los autores del dataset.
- 5.144 pares de pregunta/respuesta verificados.
- 172.700 etiquetas de evaluación (judge labels).

## Qué se usa en esta fase (muestra, no el dataset completo)

Bajar los 7.908 chunks completos no era necesario para validar el pipeline en Mac, así que se tomó una muestra dirigida a maximizar cobertura de preguntas verificadas dentro de un presupuesto chico de chunks:

- **Criterio de selección**: páginas con preguntas de evaluación asociadas, priorizando las páginas con más preguntas, excluyendo páginas gigantes (>20 chunks, ej. la referencia completa de `kubectl`) para no gastar presupuesto en una sola página.
- **Resultado**: 62 páginas, 398 chunks, 485 pares de pregunta/respuesta cuyo `source_chunk` cae dentro de esos 398 chunks (o sea, con trazabilidad completa: cada pregunta tiene un chunk "correcto" conocido para validar retrieval).

Archivos en `datasets/k8s_rag/`:
- `corpus_sample.jsonl` — los 398 chunks.
- `qa_eval_sample.jsonl` — las 485 preguntas/respuestas verificadas.
- `README.md` — atribución y link al dataset completo.

## Esquema de los chunks (`corpus_sample.jsonl`)

| Campo | Descripción |
|---|---|
| `id` | Id único del chunk (ej. `concepts_configuration_secret-0`) |
| `page_id` | Id de la página de origen |
| `url` | URL original en kubernetes.io/docs |
| `title` | Título de la página |
| `section` | Sección dentro de la página |
| `text` | El texto del chunk (lo que se embebe) |
| `chunk_index` | Posición del chunk dentro de la página |
| `page_kind` | Tipo de página (`concept`, `narrative`, etc.) |

## Esquema de las preguntas (`qa_eval_sample.jsonl`)

| Campo | Descripción |
|---|---|
| `id` | Id de la pregunta |
| `question` | Pregunta en inglés |
| `answer` | Respuesta de referencia (ground truth) |
| `source_chunk` | Id del chunk que contiene la respuesta — permite verificar si el retrieval encontró el chunk correcto |
| `source_page` | Página de origen |
| `provenance` | Cómo se generó la pregunta (ej. `manual`) |

## Cómo está cargado hoy

- **Embeddings**: `nomic-embed-text` (vía Ollama), 768 dimensiones.
- **Base vectorial**: Qdrant, colección `k8s_docs`, 398 puntos, distancia coseno.
- **Tiempo de ingesta medido**: ~11s para embeber + subir los 398 chunks en esta Mac (no es cuello de botella).
- Script de ingesta: `benchmarks/rag/ingest.py --jsonl-file datasets/k8s_rag/corpus_sample.jsonl --collection k8s_docs`.

## Escalar a la unidad real

El dataset completo (952 páginas / 7.908 chunks) está disponible en el link de arriba si conviene usar el corpus completo en la unidad real (con GPU, embeber 7.908 chunks debería tardar segundos). Ver también la sección de snapshot/restore de Qdrant pendiente en [PROJECT_PLAN.md](PROJECT_PLAN.md) para no tener que re-ingestar en la ventana de 4h.
