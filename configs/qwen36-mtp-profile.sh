#!/usr/bin/env bash
###############################################################################
# qwen36-mtp-profile.sh  —  Qwen3.6-35B-A3B MTP speculative-decode profile (OPT-IN)
# -----------------------------------------------------------------------------
# Ruling : Gate-1 D3 (MTP selected over DFlash; DISABLED by default; explicit opt-in only),
#          COR-2 (MTP arch floor is vLLM v0.17.0 -> BOTH the 0.25.0 and 0.24.0 engines qualify),
#          R3 / R11 (prefix-caching x MTP correctness collapse; low-batch net-slowdown reports).
# Mandatory pins:
#     --speculative-config '{"method":"mtp","num_speculative_tokens":2}'   (D3)
#       (nvfp4 target default adds ,"moe_backend":"triton" — NVIDIA serve recipe: MTP experts
#        on triton while main experts stay Marlin; matches specdec_grid.py / canary_repros.md)
#     --no-enable-prefix-caching   <-- MANDATORY (D3): prefix-caching + MTP is an OPEN upstream
#          correctness collapse on THIS exact model (#43559 tool-calling 90%->50%, #47194, #47087;
#          5 fix PRs unmerged). This flag is NON-NEGOTIABLE for the MTP profile.
# Target (SPEC_TARGET env): "bf16" (default) or "nvfp4".
#     bf16  -> Qwen/Qwen3.6-35B-A3B                --dtype bfloat16
#     nvfp4 -> nvidia/Qwen3.6-35B-A3B-NVFP4        --quantization modelopt --kv-cache-dtype fp8 --moe-backend marlin
#              (nvfp4 target is itself promotion-gated on handoff/scale_scan.py — see qwen36-nvfp4.sh)
#
# >>> BENCHMARK GATE (D3, binding) — OFF BY DEFAULT <<<
#   MTP enablement is BENCHMARK-GATED. Ship the composed-benchmark grid (quant x spec x caching,
#   acceptance rate, loop/degeneracy scan, batch-size crossover) + the 7 canary repros as the GPU
#   handoff, and only enable MTP in production after that grid passes on sm90. The default serving
#   profile is qwen36-default.sh (prefix caching ON, MTP OFF).
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
SPEC_TARGET="${SPEC_TARGET:-bf16}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"
SPEC_CONFIG_USER="${SPEC_CONFIG:-}"

# --- resolve target (bf16 | nvfp4) -------------------------------------------
case "${SPEC_TARGET}" in
  bf16)
    MODEL="Qwen/Qwen3.6-35B-A3B"
    target_args=( --dtype bfloat16 )
    ;;
  nvfp4)
    MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"
    target_args=( --quantization modelopt --kv-cache-dtype fp8 --moe-backend marlin )
    echo "[qwen36-mtp] nvfp4 target is PROMOTION-GATED on handoff/scale_scan.py (R2)." >&2
    ;;
  *)
    echo "ERROR: SPEC_TARGET must be 'bf16' or 'nvfp4' (got '${SPEC_TARGET}')." >&2
    exit 2
    ;;
esac

# --- spec config default is TARGET-DEPENDENT (review fix, D3 cross-file consistency) --------
# NVFP4 target: NVIDIA's serve recipe runs the MTP experts on the triton MoE backend while the
# main experts stay on Marlin (--moe-backend marlin above). handoff/specdec_grid.py and
# handoff/canary_repros.md validate exactly that composed config, so the shipped profile must
# match it — otherwise the D3 benchmark gate never covers what actually deploys.
if [ -n "${SPEC_CONFIG_USER}" ]; then
  SPEC_CONFIG="${SPEC_CONFIG_USER}"
elif [ "${SPEC_TARGET}" = "nvfp4" ]; then
  SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"moe_backend\":\"triton\"}"
else
  SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
fi

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         qwen36-mtp-profile requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi

echo "[qwen36-mtp] BENCHMARK-GATED profile (OFF by default). Enable only after the composed" >&2
echo "[qwen36-mtp]   benchmark grid + 7 canary repros pass on sm90 (D3)." >&2

mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:8000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache
               --entrypoint vllm )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args: target flags + MTP spec config + MANDATORY prefix-caching disable ----------
serve_args=( serve "${MODEL}"
             --host 0.0.0.0 --port 8000
             --trust-remote-code
             "${target_args[@]}"
             --speculative-config "${SPEC_CONFIG}"
             --no-enable-prefix-caching )

echo "[qwen36-mtp] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT} target=${SPEC_TARGET}"
echo "[qwen36-mtp] spec=${SPEC_CONFIG} | --no-enable-prefix-caching (MANDATORY per D3)"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
