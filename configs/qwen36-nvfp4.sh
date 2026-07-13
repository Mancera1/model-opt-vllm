#!/usr/bin/env bash
###############################################################################
# qwen36-nvfp4.sh  —  Qwen3.6-35B-A3B NVFP4 quantized generation profile
# -----------------------------------------------------------------------------
# Ruling : Gate-1 D2 (use nvidia/Qwen3.6-35B-A3B-NVFP4 as-is; MIXED_PRECISION), D4 (KV),
#          COR-6 (`--quantization modelopt` auto-selects modelopt_mixed for this checkpoint).
# Pins / levers:
#     model             -> nvidia/Qwen3.6-35B-A3B-NVFP4 (161 W4A16-NVFP4 experts + FP8 attn + FP8 KV)
#     --quantization modelopt   (COR-6: routes to modelopt_mixed / MIXED_PRECISION)
#     --kv-cache-dtype fp8      (checkpoint already carries KV scales -> no --calculate-kv-scales)
#     --moe-backend marlin      (D2: sm90 Marlin weight-only path; min_capability 75, Hopper-servable)
#
# >>> PROMOTION GATE (R2) — FULL SCAN EXECUTED: PASS (2026-07-12) <<<
#   The full handoff/scale_scan.py scan of the UPSTREAM checkpoint has been executed by the
#   orchestrator: 30,841/30,841 F8_E4M3 weight_scale tensors (2,052.9 MB), ZERO NaN (0x7F/0xFF)
#   or negative bytes — evidence: handoff/results/scale_scan_result.2026-07-12.json.
#   (Context: the checkpoint was produced with modelopt 0.44.0, whose NaN-scale export bug is
#   not marked fixed in 0.45's changelog — hence the scan.) Before PROD cutover still:
#   (1) re-run scale_scan.py on YOUR OWN downloaded copy (guards against a divergent/partial
#   download), and (2) run the degenerate-output detector on a GPU smoke run.
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
MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         qwen36-nvfp4 requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi

# --- promotion-gate reminder (non-blocking; the gate is enforced by process, not this script) --
echo "[qwen36-nvfp4] Scale-scan gate (R2): upstream checkpoint FULL-SCAN PASS 2026-07-12" >&2
echo "[qwen36-nvfp4]   (handoff/results/scale_scan_result.2026-07-12.json, 30,841/30,841 clean)." >&2
echo "[qwen36-nvfp4]   Before PROD: re-run scale_scan.py on YOUR downloaded copy + degenerate" >&2
echo "[qwen36-nvfp4]   detector on a GPU smoke run." >&2

mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:8000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache
               --entrypoint vllm )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (NVFP4 via modelopt; fp8 KV from ckpt scales; Marlin MoE backend) -----------
serve_args=( serve "${MODEL}"
             --host 0.0.0.0 --port 8000
             --trust-remote-code
             --quantization modelopt
             --kv-cache-dtype fp8
             --moe-backend marlin )

echo "[qwen36-nvfp4] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT}"
echo "[qwen36-nvfp4] quant=modelopt(modelopt_mixed) | kv fp8 (ckpt scales) | moe-backend marlin"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
