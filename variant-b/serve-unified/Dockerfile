# syntax=docker/dockerfile:1
###############################################################################
# Variant B / serve-unified  —  single serving image for ALL THREE models
# -----------------------------------------------------------------------------
# DEPLOYMENT MODEL: one image, one container PER model (vLLM serves exactly one model per
# process). This single image can launch any of:
#   * Qwen/Qwen3.6-35B-A3B         via `vllm serve ...`        (MoE generation, BF16/NVFP4)
#   * Qwen/Qwen3-VL-Embedding-8B   via `vllm serve ...`        (multimodal pooling runner)
#   * Qwen/Qwen3-Omni-30B-A3B-Instruct via `vllm-omni serve ... --omni`  (omni)
# The config launchers (configs/*.sh) select the engine with `docker run --entrypoint {vllm|
# vllm-omni}`, so the SAME image works for every profile.
#
# Base : vllm/vllm-openai:v0.24.0   (Matrix IMAGE 2; config-blob-verified CUDA 13.0.2 /
#        torch 2.11.0 / flashinfer 0.6.12). The base's vLLM 0.24.0 already serves generation +
#        embedding; vllm-omni adds the omni capability on top.
# Omni : vllm-omni @ commit b9e9d236c3f78afd405119a5b686ebebeeb53984 (v0.24.1 line; RMSNorm fix).
# Rulings: Human Amendment 2026-07-12 (build BOTH topology variants), Gate-1 D1 fallback
#          ("pin everything to vllm 0.24.0 — MTP floor v0.17.0 permits it"), COR-2, COR-8, R6.
#
# ---------------------------------------------------------------------------
# >>> PARITY CAVEAT (Human Amendment 2026-07-12, binding) <<<
#   Variant B runs the vLLM-0.24.0 engine, which ships a DIFFERENT server-side video-IO stack
#   than Variant A's serve-main (vLLM 0.25.0):
#       * vLLM 0.24.0  = OpenCV only (opencv-python-headless>=4.13.0);
#                        NO torchcodec, NO PyNvVideoCodec.
#       * vLLM 0.25.0  = added torchcodec>=0.14 + PyNvVideoCodec==2.0.4 on top of OpenCV.
#   Both facts verified at the tags. Server-side video-FILE decode is OFF our critical path
#   (production frames arrive client-extracted — brief call type B), so this does not affect
#   our workloads. But anyone sending RAW VIDEO FILES to a Variant-B endpoint decodes them
#   through a different backend than Variant A -> treat as a feature-parity reconcile item.
#
#   Variant B also FORFEITS vLLM 0.25.0's newest generation + embedding fixes. See README
#   "Variant A vs B trade-off" table.
# ---------------------------------------------------------------------------
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     [--build-arg PIP_INDEX_URL=<prod-mirror-url>] \
#     -t model-opt-vllm/serve-unified:vllm-0.24.0-omni .
#
# Notes: no :latest tags; NO runtime pip; PIP_INDEX_URL is a prod-mirror hook — NO creds baked.
###############################################################################

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

FROM vllm/vllm-openai:v0.24.0

# Re-declare with defaults: a pre-FROM ARG default does not propagate into the build stage.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="model-opt-vllm serve-unified (Variant B)" \
      org.opencontainers.image.description="Single vLLM-0.24.0 + vllm-omni image hosting all three models (one container per model) on sm90 Hopper" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.24.0" \
      org.opencontainers.image.version="vllm-0.24.0-omni-0.24.1+b9e9d23" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="model-opt-vllm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.model-opt-vllm.variant="B" \
      com.model-opt-vllm.role="serve-unified" \
      com.model-opt-vllm.engine="vllm==0.24.0 + vllm-omni@b9e9d23" \
      com.model-opt-vllm.hosts="Qwen3.6-35B-A3B; Qwen3-VL-Embedding-8B; Qwen3-Omni-30B-A3B-Instruct" \
      com.model-opt-vllm.parity-caveat="vLLM 0.24.0 video IO = OpenCV only (no torchcodec/PyNvVideoCodec)" \
      com.model-opt-vllm.rulings="Human Amendment 2026-07-12; Gate-1: D1 fallback, COR-2, COR-8, R6"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_BUILD_DATE=${BUILD_DATE} \
    HF_HOME=/hf-cache

# --- system deps (same as serve-omni: git for the pinned commit + omni media codecs) --------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      libsndfile1 \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY constraints.txt /etc/constraints.txt

# --- GUARD 1/2: snapshot core pins BEFORE the vllm-omni install --------------
RUN python3 -m pip freeze | grep -E "^(vllm|torch|transformers|flashinfer)" | LC_ALL=C sort > /etc/pins_before.txt \
 && echo "=== core pins BEFORE vllm-omni install ===" \
 && cat /etc/pins_before.txt

# --- install vllm-omni at the pinned commit, under constraints ---------------
# Same isolation as serve-omni: the six core packages are frozen; omni's own deps install freely.
RUN python3 -m pip install --no-cache-dir -c /etc/constraints.txt \
      "vllm-omni @ git+https://github.com/vllm-project/vllm-omni@b9e9d236c3f78afd405119a5b686ebebeeb53984"

# --- GUARD 2/2: fail the build if the omni install MOVED any core pin --------
# Decision via `comm -23` (moves/removals only); the naive `diff` is shown for humans but does
# NOT gate the build, because vllm-omni legitimately ADDS new torch*/vllm* packages (torchsde,
# vllm-omni itself) that the grep also matches — additions are not moves. See serve-omni for the
# full rationale (Gate-1 requirement 2: "build MUST FAIL if the omni install moved any of them").
RUN set -eu; export LC_ALL=C; \
    python3 -m pip freeze | grep -E "^(vllm|torch|transformers|flashinfer)" | sort > /etc/pins_after.txt; \
    echo "=== core pins AFTER vllm-omni install ==="; cat /etc/pins_after.txt; \
    echo "=== informational diff ('+' lines are new, allowed omni deps) ==="; \
    diff -u /etc/pins_before.txt /etc/pins_after.txt || true; \
    moved="$(comm -23 /etc/pins_before.txt /etc/pins_after.txt || true)"; \
    if [ -n "${moved}" ]; then \
        echo "GUARD FAILED: vllm-omni install MOVED/removed core pin(s) — build aborted:"; \
        echo "${moved}"; \
        exit 1; \
    fi; \
    echo "GUARD OK: vllm / torch / transformers / flashinfer unchanged by vllm-omni install"

# --- provenance manifest -----------------------------------------------------
RUN python3 - <<'PY'
import json, os, subprocess
pins = []
with open("/etc/constraints.txt") as fh:
    for line in fh:
        s = line.split("#", 1)[0].strip()
        if s and "==" in s:
            pins.append(s)
def apt_versions(names):
    out = {}
    for n in names:
        try:
            out[n] = subprocess.check_output(
                ["dpkg-query", "-W", "-f=${Version}", n],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out[n] = None
    return out
manifest = {
    "image": "model-opt-vllm/serve-unified",
    "variant": "B",
    "role": "single image hosting all three models (one container per model)",
    "hosts": [
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3-VL-Embedding-8B",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    ],
    "base_image": "vllm/vllm-openai:v0.24.0",
    "engine": "vllm==0.24.0 + vllm-omni",
    "vllm_omni_commit": "b9e9d236c3f78afd405119a5b686ebebeeb53984",
    "parity_caveat": "vLLM 0.24.0 server-side video IO = OpenCV only (no torchcodec / no PyNvVideoCodec); off critical path (frames arrive client-extracted)",
    "platform": "linux/amd64",
    "python": "3.12",
    "cuda_lineage": "13.0.2",
    "build_date": os.environ.get("IMAGE_BUILD_DATE", "unknown"),
    "frozen_direct_pins": sorted(pins),
    "apt_packages": apt_versions(["git", "ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0"]),
    "rulings": "Human Amendment 2026-07-12; Gate-1 adjudications: D1 fallback, COR-2, COR-8, R6",
}
with open("/etc/image-manifest.json", "w") as o:
    json.dump(manifest, o, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

# Multi-engine image: NO ENTRYPOINT is baked. The config launchers set `docker run
# --entrypoint {vllm|vllm-omni}` per profile so one image serves generation, embedding, or omni.
