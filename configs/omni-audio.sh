#!/usr/bin/env bash
###############################################################################
# omni-audio.sh  —  Qwen3-Omni-30B-A3B-Instruct omni serving profile (call type A)
# -----------------------------------------------------------------------------
# Ruling : Gate-1 D2 (Omni BF16 default), D5 (parity pins), brief §3 call type A
#          (raw audio in one inference call -> audio feature extraction, routed to omni).
# Engine : vLLM-Omni (`vllm-omni serve ... --omni`), per the vLLM-Omni user guide.
# Parity pins (LOAD-BEARING; brief §2/§3, D5):
#     BF16                          -> --dtype bfloat16
#     --mm-processor-kwargs         -> pinned (see OMNI_MM_KWARGS below)
# AUDIO-KWARGS RECONCILE (honesty): the Gate-1 corpus pins the VISION/frame kwargs
#     ({"fps":2.0,"max_frames":64}) fleet-wide but specifies NO exact audio-processor numeric
#     (e.g. sampling rate) for Qwen3-Omni. OMNI_MM_KWARGS therefore defaults to the fleet
#     vision pins; the audio path uses the model's own processor config. Confirm any
#     audio-specific kwarg against the vLLM-Omni processor config at prod integration — do NOT
#     invent one. Override via OMNI_MM_KWARGS if the omni processor rejects these keys.
# IMAGE (env): variant-a serve-omni  OR  variant-b serve-unified (both ship `vllm-omni`).
#
# >>> BLOCKED-protocol header <<<
#   GPU REQUIRED (NVIDIA Hopper sm90 / H100|H200). No GPU in the authoring environment ->
#   BLOCKED here. COMPLETE and runnable as-written on a Hopper host with the NVIDIA Container
#   Toolkit. Preflight refuses to launch without a visible GPU (SKIP_GPU_PREFLIGHT=1 overrides).
###############################################################################
set -euo pipefail

# --- required / tunable env --------------------------------------------------
: "${IMAGE:?Set IMAGE=<serve image>  (variant-a serve-omni OR variant-b serve-unified)}"
PORT="${PORT:-8000}"
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
OMNI_MM_KWARGS="${OMNI_MM_KWARGS:-{\"fps\":2.0,\"max_frames\":64}}"   # vision-frame parity pins; audio uses processor default
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         omni-audio requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:8000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache
               --entrypoint vllm-omni )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (BF16 default; --omni; pinned mm kwargs) --------------------
serve_args=( serve "${MODEL}"
             --omni
             --host 0.0.0.0 --port 8000
             --trust-remote-code
             --dtype bfloat16
             --mm-processor-kwargs "${OMNI_MM_KWARGS}" )

echo "[omni-audio] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT} dtype=bfloat16"
echo "[omni-audio] mm_kwargs=${OMNI_MM_KWARGS} (vision pins; audio uses processor default — reconcile at prod)"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
