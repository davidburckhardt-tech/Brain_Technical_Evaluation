#!/usr/bin/env bash
# Shared model short-name -> Hugging Face repo id lookup, sourced by every
# test script's --model flag, so "qwen-2.5-1.5b-instruct" and
# "llama-3.2-1b-instruct" mean the same repo everywhere (vLLM startup,
# GuideLLM's tokenizer, the chatbot/RAG scripts' MODEL_NAME) instead of each
# script's .env needing VLLM_MODEL/MODEL_NAME/TOKENIZER_MODEL kept in sync by
# hand - exactly the kind of drift (mismatched tokenizer, wrong model
# targeted) this project already hit once with .env edits.
#
# Add a new model by adding one line to resolve_model_repo() - no other
# script needs to change.

resolve_model_repo() {
  case "$1" in
    qwen-2.5-1.5b-instruct) echo "Qwen/Qwen2.5-1.5B-Instruct" ;;
    llama-3.2-1b-instruct) echo "meta-llama/Llama-3.2-1B-Instruct" ;;
    *)
      echo "Error: unknown --model '$1'. Supported: qwen-2.5-1.5b-instruct, llama-3.2-1b-instruct" >&2
      return 1
      ;;
  esac
}

# Extracts "--model <short-name>" from the given arguments (if present),
# resolving it to a HF repo id in RESOLVED_MODEL_REPO (empty if --model was
# not passed) and leaving every other argument, in order, in the PARSED_ARGS
# array - the caller re-applies them with `set -- "${PARSED_ARGS[@]}"`.
# Must be called directly (not via `$(...)`), so these globals are set in
# the caller's own shell rather than a throwaway subshell.
parse_model_flag() {
  RESOLVED_MODEL_REPO=""
  PARSED_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        RESOLVED_MODEL_REPO="$(resolve_model_repo "$2")" || exit 1
        shift 2
        ;;
      *)
        PARSED_ARGS+=("$1")
        shift
        ;;
    esac
  done
}
