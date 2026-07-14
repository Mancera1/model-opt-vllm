#!/usr/bin/env bash
###############################################################################
# sglang/qwen36-default.sh — Qwen3.6-35B-A3B DEFAULT generation profile on SGLang
#                            (BF16, no spec decode) — Variant C
# -----------------------------------------------------------------------------
# Ruling : Gate-3 D6 (variant-c profiles mirror the Gate-1 D3 posture); D3 quoted:
#          prefix caching ON, spec decode OFF. Engine mapping per gate3-adjudication §3.
# Pins / levers (SGLang equivalents):
#     BF16                          -> --dtype bfloat16
#     prefix caching ON             -> RadixAttention DEFAULT-ON (no flag needed); hybrid-GDN
#                                      routed via mamba radix cache, strategy auto (C26)
#     FP8 KV cache                  -> **OFF (R21)**: sglang has NO --calculate-kv-scales
#                                      analogue for BF16 targets; scales would default to 1.0
#                                      with a documented accuracy warning. Enable only after
#                                      the GPU gate validates a --quantization-param-path route.
#     spec decode                   -> OFF (no --speculative-algorithm)  (D3)
#     CUDA graphs                   -> DEFAULTS (on; decode=full, prefill=breakable — COR-13:
#                                      torch.compile stays OFF/experimental on sglang)
# Parity : cookbook suggests --reasoning-parser qwen3; OMITTED here — output-shape parity with
#          the vLLM qwen36-default profile is adjudicated first (Gate-2-style reconcile item).
# sm90   : GDN decode runs triton with FP32 SSM state on Hopper (R20 — bench at GPU gate).
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
PORT="${PORT:-30000}"                # host port; container listens on sglang default 30000
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="Qwen/Qwen3.6-35B-A3B"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         sglang/qwen36-default requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:30000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (BF16; radix default-on; NO fp8 KV per R21; NO spec decode) ---
serve_args=( serve --model-path "${MODEL}"
             --host 0.0.0.0 --port 30000
             --trust-remote-code
             --dtype bfloat16 )

echo "[sglang/qwen36-default] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT}"
echo "[sglang/qwen36-default] BF16 | radix cache ON (engine default) | kv dtype auto (R21: no fp8 on BF16 target) | spec decode OFF"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
