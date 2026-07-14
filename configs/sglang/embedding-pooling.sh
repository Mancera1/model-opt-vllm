#!/usr/bin/env bash
###############################################################################
# sglang/embedding-pooling.sh — Qwen3-VL-Embedding-8B embedding server on SGLang
#                               (--is-embedding; pooler LAST+normalize) — Variant C
# -----------------------------------------------------------------------------
# Ruling : Gate-3 D6; Gate-1 D5 quoted (preprocessing is part of the model; parity is a hard
#          gate vs the production precomputed index — embedding is NEVER quantized, D2).
# Pooling: sglang's Qwen3VLForConditionalGeneration hardcodes Pooler(LAST, normalize=True)
#          (gate3 C9) — the same semantics the vLLM profile pins via --runner pooling.
#          --is-embedding flips THIS WHOLE SERVER to embedding mode (exclusive; one model
#          per container process, same as the vLLM layout).
# CAVEATS (gate3 §1a, carried into D6 — resolve BEFORE prod cutover):
#   * chat-template: sglang's multimodal-embedding docs use a --chat-template; NO
#     Qwen3-VL-Embedding-specific template was found. UNSET here. Resolve at profile
#     authoring vs upstream guidance, verify at first GPU smoke. (gate3 C13)
#   * mm preprocessing parity pins: the vLLM profile pins --mm-processor-kwargs
#     {"fps":2.0,"max_frames":64}; the sglang analogue is UNVERIFIED and therefore NOT set —
#     the D5/R8 parity procedure (handoff/parity_embedding.py, --save-reference on the
#     CURRENT image first) is the hard gate before any index-serving cutover.
# IMAGE (env): variant-c serve-sglang (sglang 0.5.15.post1).
#
# >>> BLOCKED-protocol header <<<
#   GPU REQUIRED (NVIDIA Hopper sm90 / H100|H200). No GPU in the authoring environment ->
#   BLOCKED here. COMPLETE and runnable as-written on a Hopper host with the NVIDIA Container
#   Toolkit. Preflight refuses to launch without a visible GPU (SKIP_GPU_PREFLIGHT=1 overrides).
###############################################################################
set -euo pipefail

# --- required / tunable env --------------------------------------------------
: "${IMAGE:?Set IMAGE=<serve image>  (variant-c serve-sglang)}"
PORT="${PORT:-30001}"                # host port; container listens on sglang default 30000
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="Qwen/Qwen3-VL-Embedding-8B"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         sglang/embedding-pooling requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:30000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (embedding mode; BF16; chat-template deliberately unset — see header) ---
serve_args=( serve --model-path "${MODEL}"
             --host 0.0.0.0 --port 30000
             --trust-remote-code
             --dtype bfloat16
             --is-embedding )

echo "[sglang/embedding-pooling] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT}"
echo "[sglang/embedding-pooling] embedding mode (pooler LAST+normalize in-model) | BF16 | chat-template UNSET (GPU-gate item) | /v1/embeddings"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
