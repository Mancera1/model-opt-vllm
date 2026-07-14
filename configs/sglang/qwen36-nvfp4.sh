#!/usr/bin/env bash
###############################################################################
# sglang/qwen36-nvfp4.sh — nvidia/Qwen3.6-35B-A3B-NVFP4 serving profile on SGLang
#                          (ModelOpt MIXED_PRECISION, Marlin-on-sm90) — Variant C
# -----------------------------------------------------------------------------
# Ruling : Gate-3 D6; Gate-1 D2 quoted (NVIDIA pre-quant served as-is; promotion gate).
# Quant  : NO --quantization flag — sglang AUTO-DETECTS hf_quant_config.json and selects
#          quant_method `modelopt_mixed` (the same method name vLLM selects; gate3 C14).
#          On sm90 the 161 W4A16_NVFP4 expert layers run the Marlin A16 path; the NVFP4
#          FusedMoE auto-falls back to Marlin for capability in [8.0, 10.0) (C15/C17) —
#          no --moe-backend flag needed (contrast: vLLM profile pins --moe-backend marlin).
# KV     : --kv-cache-dtype fp8_e4m3 — THIS checkpoint ships FP8 KV scales; sglang loads them
#          via ModelOptFp8KVCacheMethod (C19). (The BF16 profile has no such scales — R21.)
# Floor  : sglang >=0.5.15 required for this artifact (W4A16_NVFP4 handling; gate3 C21).
# GATE   : **PROMOTION-GATED** exactly like the vLLM profile — re-run handoff/scale_scan.py on
#          YOUR OWN downloaded checkpoint copy before prod cutover (upstream copy already
#          certified 30,841/30,841 clean, 2026-07-12 — R2 closed-pass for that copy).
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
MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         sglang/qwen36-nvfp4 requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:30000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (quant auto-detected; fp8_e4m3 KV: checkpoint ships scales) ---
serve_args=( serve --model-path "${MODEL}"
             --host 0.0.0.0 --port 30000
             --trust-remote-code
             --kv-cache-dtype fp8_e4m3 )

echo "[sglang/qwen36-nvfp4] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT}"
echo "[sglang/qwen36-nvfp4] modelopt_mixed auto-detect | Marlin W4A16 on sm90 | kv fp8_e4m3 (checkpoint scales) | PROMOTION-GATED on scale_scan (R2)"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
