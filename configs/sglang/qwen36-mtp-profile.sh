#!/usr/bin/env bash
###############################################################################
# sglang/qwen36-mtp-profile.sh — Qwen3.6-35B-A3B MTP speculative-decoding profile on SGLang
#                                (OPT-IN; BENCHMARK-GATED) — Variant C
# -----------------------------------------------------------------------------
# Ruling : Gate-3 D6; Gate-1 D3 quoted (MTP is the tested candidate, OFF by default, enabled
#          only after the composed benchmark grid + canaries pass on sm90).
# Engine mapping (gate3 §1a/§3 — differs from the vLLM profile in three ways):
#   1. Algorithm: --speculative-algorithm EAGLE (sglang's NEXTN is an alias that resolves to
#      EAGLE). The draft consumes the checkpoint's NATIVE MTP head: sglang rewrites the draft
#      arch Qwen3_5MoeForConditionalGeneration -> Qwen3_5ForCausalLMMTP (gate3 C4).
#   2. Draft path: this arch is NOT in sglang's auto-draft-path list -> the draft model path
#      MUST be passed explicitly; it is the SAME repo id (the MTP head lives inside the
#      checkpoint; for the NVFP4 target the MTP head loads unquantized — gate3 C5).
#   3. Prefix cache: vLLM's D3 mandate (--no-enable-prefix-caching) came from OPEN vLLM BUGS
#      (#43559-class, R3) that do NOT transfer; sglang's mamba extra_buffer radix strategy is
#      designed to coexist with spec decode (gate3 C26). Radix stays at engine default here.
#      >>> The handoff/canary_repros.md suite MUST be re-run against THIS endpoint before any
#      enablement decision — treat sglang MTPxRadix as unvalidated until then. <<<
# Spec params: cookbook-published values for this model (gate3 C28):
#      --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
# Targets: SPEC_TARGET=bf16 (default) or SPEC_TARGET=nvfp4 (adds fp8_e4m3 KV — checkpoint
#      ships scales; PROMOTION-GATED on scale_scan like the base nvfp4 profile).
# IMAGE (env): variant-c serve-sglang (sglang 0.5.15.post1).
#
# >>> BLOCKED-protocol header <<<
#   GPU REQUIRED (NVIDIA Hopper sm90 / H100|H200) AND the benchmark grid + canaries must pass
#   BEFORE this profile is used for real traffic (D3). No GPU in the authoring environment ->
#   BLOCKED here. Preflight refuses to launch without a visible GPU (SKIP_GPU_PREFLIGHT=1).
###############################################################################
set -euo pipefail

# --- required / tunable env --------------------------------------------------
: "${IMAGE:?Set IMAGE=<serve image>  (variant-c serve-sglang)}"
PORT="${PORT:-30003}"                # host port; container listens on sglang default 30000
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
SPEC_TARGET="${SPEC_TARGET:-bf16}"

case "${SPEC_TARGET}" in
  bf16)
    MODEL="Qwen/Qwen3.6-35B-A3B"
    extra_target_args=( --dtype bfloat16 )
    ;;
  nvfp4)
    MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"
    # quant auto-detects (modelopt_mixed); MTP head loads unquantized (gate3 C5);
    # fp8_e4m3 KV: this checkpoint ships KV scales (gate3 C19). PROMOTION-GATED (R2).
    extra_target_args=( --kv-cache-dtype fp8_e4m3 )
    ;;
  *)
    echo "ERROR: SPEC_TARGET must be 'bf16' or 'nvfp4' (got: ${SPEC_TARGET})" >&2
    exit 2
    ;;
esac

# --- GPU preflight (BLOCKED-protocol) ----------------------------------------
if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "BLOCKED: no NVIDIA GPU visible (nvidia-smi not found)." >&2
    echo "         sglang/qwen36-mtp requires an sm90 Hopper host + NVIDIA Container Toolkit." >&2
    echo "         This profile is additionally BENCHMARK-GATED (D3): specdec grid + canaries first." >&2
    echo "         Set SKIP_GPU_PREFLIGHT=1 to override on a validated host." >&2
    exit 3
  fi
fi
mkdir -p "${HF_HOME_HOST}"

docker_flags=( --rm --gpus all --ipc=host
               -p "${PORT}:30000"
               -v "${HF_HOME_HOST}:/hf-cache" -e HF_HOME=/hf-cache )
[ -n "${HF_TOKEN:-}" ] && docker_flags+=( -e "HF_TOKEN=${HF_TOKEN}" )

# --- serve args (EAGLE on the native MTP head; explicit draft path = same repo) ---
serve_args=( serve --model-path "${MODEL}"
             --host 0.0.0.0 --port 30000
             --trust-remote-code
             --speculative-algorithm EAGLE
             --speculative-draft-model-path "${MODEL}"
             --speculative-num-steps 3
             --speculative-eagle-topk 1
             --speculative-num-draft-tokens 4
             "${extra_target_args[@]}" )

echo "[sglang/qwen36-mtp] IMAGE=${IMAGE} MODEL=${MODEL} PORT=${PORT} target=${SPEC_TARGET}"
echo "[sglang/qwen36-mtp] EAGLE on native MTP head | radix cache at engine default (canary-gated — see header) | BENCHMARK-GATED (D3)"
exec docker run "${docker_flags[@]}" "${IMAGE}" "${serve_args[@]}" "$@"
