#!/usr/bin/env bash
###############################################################################
# qwen36-default.sh  —  Qwen3.6-35B-A3B DEFAULT generation profile (BF16, no spec decode)
# -----------------------------------------------------------------------------
# Ruling : Gate-1 D3 (default posture: prefix caching ON, spec decode OFF), D4 (levers),
#          COR-1 (Qwen3.6 is natively MULTIMODAL — deployment uses TEXT-GEN traffic only, but
#          VRAM sizing MUST include the BF16 vision tower).
# Pins / levers:
#     BF16                          -> --dtype bfloat16
#     prefix caching ON             -> --enable-prefix-caching       (D3 default profile)
#     FP8 KV cache (D4 lever 3)     -> --kv-cache-dtype fp8_e4m3
#     BF16 target computes KV scale -> --calculate-kv-scales          (D4; NVFP4 ckpt already carries them)
#     spec decode                   -> OFF (no --speculative-config)  (D3)
#     torch.compile + CUDA graphs   -> DEFAULTS (D4 lever 1: never --enforce-eager in prod)
# Risk R4: fp8_e4m3 KV depends on the FA3-active-on-cu13 path (UNVERIFIED, gated at Gate-2 GPU
#     smoke test); the FlashInfer backend also supports fp8 KV as a fallback.
# IMAGE (env): variant-a serve-main (vLLM 0.25.0)  OR  variant-b serve-unified (vLLM 0.24.0).
#
# >>> BLOCKED-protocol header <<<
#   GPU REQUIRED (NVIDIA Hopper sm90 / H100|H200). No GPU in the authoring environment ->
#   BLOCKED here. COMPLETE and runnable as-written on a Hopper host with the NVIDIA Container
#   Toolkit. Preflight refuses to launch without a visible GPU (SKIP_GPU_PREFLIGHT=1 overrides).
###############################################################################
set -euo pipefail

# --- required / tunable env --------------------------------------------------
: "${IMAGE:?Set IMAGE=<serve image>  (variant-a serve-main OR variant-b serve-unified)}"
PORT="${PORT:-8000}"
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="Qwen/Qwen3.6-35B-A3B"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         qwen36-default requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:8000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache
               --entrypoint vllm )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (BF16; prefix caching ON; fp8_e4m3 KV + calculate scales; NO spec decode) ---
serve_args=( serve "${MODEL}"
             --host 0.0.0.0 --port 8000
             --trust-remote-code
             --dtype bfloat16
             --enable-prefix-caching
             --kv-cache-dtype fp8_e4m3
             --calculate-kv-scales )

echo "[qwen36-default] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT}"
echo "[qwen36-default] BF16 | prefix-caching ON | kv fp8_e4m3 + calculate-kv-scales | spec decode OFF"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
