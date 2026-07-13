#!/usr/bin/env bash
###############################################################################
# embedding-pooling.sh  —  Qwen3-VL-Embedding-8B multimodal pooling profile
# -----------------------------------------------------------------------------
# Ruling : Gate-1 D5 (parity pins), S7 (pooler + mm-kwargs), COR (VL-Embedding kept BF16 —
#          D2: "skip quantization"; dense 8B fits, tight cosine gate, #33204 already fragile).
# Parity pins (LOAD-BEARING — feature-parity is a hard gate; brief §2/§3, D5):
#     --pooler-config '{"pooling_type":"LAST","normalize":true}'
#     --mm-processor-kwargs '{"fps":2.0,"max_frames":64}'
#     EVS / video_pruning_rate  ->  UNSET (D4/D5: EVS alters tokens -> parity violation)
# Known upstream issue: vLLM #33204 (pooling-runner video embeddings diverge from
#     qwen_vl_utils, ~0.92 cosine) — documented, not worked around here.
# IMAGE (env): variant-a serve-main (vLLM 0.25.0)  OR  variant-b serve-unified (vLLM 0.24.0).
#
# >>> BLOCKED-protocol header <<<
#   GPU REQUIRED (NVIDIA Hopper sm90 / H100|H200). There is NO GPU in the authoring
#   environment, so this profile is BLOCKED from execution here. It is COMPLETE and runnable
#   as-written on a Hopper host with the NVIDIA Container Toolkit. The preflight below refuses
#   to launch without a visible GPU rather than fabricate a run
#   (export SKIP_GPU_PREFLIGHT=1 to override on a validated host).
###############################################################################
set -euo pipefail

# --- required / tunable env --------------------------------------------------
: "${IMAGE:?Set IMAGE=<serve image>  (variant-a serve-main OR variant-b serve-unified)}"
PORT="${PORT:-8000}"
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"   # host HF cache, mounted read-write
MM_KWARGS="${MM_KWARGS:-{\"fps\":2.0,\"max_frames\":64}}"   # parity pin; override only with cause
POOLER_CONFIG="${POOLER_CONFIG:-{\"pooling_type\":\"LAST\",\"normalize\":true}}"
MODEL="Qwen/Qwen3-VL-Embedding-8B"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         embedding-pooling requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

# --- docker run flags (HF_TOKEN passed through if set; NEVER baked into an image) -----------
docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:8000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache
               --entrypoint vllm )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (BF16; pooling runner; parity pins; EVS/video_pruning intentionally UNSET) --
serve_args=( serve "${MODEL}"
             --host 0.0.0.0 --port 8000
             --trust-remote-code
             --runner pooling
             --pooler-config "${POOLER_CONFIG}"
             --mm-processor-kwargs "${MM_KWARGS}" )

echo "[embedding-pooling] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT}"
echo "[embedding-pooling] pooler=${POOLER_CONFIG} mm_kwargs=${MM_KWARGS} (EVS/video_pruning UNSET)"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
