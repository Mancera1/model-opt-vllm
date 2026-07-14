#!/usr/bin/env bash
###############################################################################
# sglang/omni-audio.sh — Qwen3-Omni-30B-A3B-Instruct (thinker, audio input) on SGLang
#                        — Variant C
# -----------------------------------------------------------------------------
# Ruling : Gate-3 D6; Gate-1 D2 quoted (omni serves BF16 by default; FP8-thinker recipe is an
#          offline, eval-gated script — unchanged by this profile).
# Arch   : sglang serves the THINKER only (its Qwen3OmniMoeForConditionalGeneration hardcodes
#          enable_talker=False — gate3 C6). Matches production call type A (audio -> features).
# AUDIO API CAVEAT (gate3 §1a): audio arrives via sglang's `audio_url` chat content part.
#          The OpenAI-official `input_audio` content type is NOT accepted at this tag — callers
#          using the OpenAI SDK audio shape need a client-side shim. Reconcile with the
#          vllm-omni request shape at the parity gate (handoff/omni_audio_check.py).
# Kernel : first launch seeds the sgl-flash-attn3 kernel from the image's baked snapshot into
#          the mounted HF cache (R14 entrypoint shim) — no network fetch expected.
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
PORT="${PORT:-30002}"                # host port; container listens on sglang default 30000
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         sglang/omni-audio requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:30000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (BF16 thinker; audio via chat `audio_url` parts) --------------
serve_args=( serve --model-path "${MODEL}"
             --host 0.0.0.0 --port 30000
             --trust-remote-code
             --dtype bfloat16 )

echo "[sglang/omni-audio] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT} dtype=bfloat16"
echo "[sglang/omni-audio] thinker-only | audio via sglang 'audio_url' content part (OpenAI 'input_audio' NOT accepted — client shim; gate3 §1a)"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
