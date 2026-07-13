# syntax=docker/dockerfile:1
###############################################################################
# Variant A / serve-main  —  vLLM 0.25.0 serving image  (linux/amd64, sm90 Hopper)
# -----------------------------------------------------------------------------
# Hosts (one model per container process):
#   * Qwen/Qwen3.6-35B-A3B          — MoE generation, BF16 or NVFP4 profiles
#   * Qwen/Qwen3-VL-Embedding-8B    — multimodal pooling runner (BF16)
#
# Base   : nvidia/cuda:13.0.2-devel-ubuntu24.04   (verl-proven cu130 anchor — Matrix IMAGE 1)
# Anchor : linux/amd64 · py3.12 · CUDA 13.0.2 · torch 2.11.0 (default PyPI wheel IS cu13) ·
#          sm90 first-class · NVIDIA driver >=535 family required (cu13; Risk R10).
# Rulings: Gate-1 adjudications — Matrix IMAGE 1, D1 (three-image topology), D4 (levers).
#          NO ray / NO nvidia-modelopt / NO flash-attn (asserted at build; Matrix IMAGE 1).
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     [--build-arg PIP_INDEX_URL=<prod-mirror-url>] \
#     -t model-opt-vllm/serve-main:vllm-0.25.0 .
#
# Notes: no :latest tags; NO runtime pip (all installs at build); PIP_INDEX_URL is a
#        prod-mirror hook only — NO credentials are ever baked into the image.
###############################################################################

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# ARGs must be re-declared after FROM to be visible in this stage. Defaults are repeated here:
# a pre-FROM ARG default does NOT propagate into the build stage, so without these an unset
# --build-arg would leave PIP_INDEX_URL empty and break pip.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="model-opt-vllm serve-main (Variant A)" \
      org.opencontainers.image.description="vLLM 0.25.0 serving: Qwen3.6-35B-A3B generation + Qwen3-VL-Embedding-8B pooling on sm90 Hopper" \
      org.opencontainers.image.base.name="nvidia/cuda:13.0.2-devel-ubuntu24.04" \
      org.opencontainers.image.version="vllm-0.25.0" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="model-opt-vllm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.model-opt-vllm.variant="A" \
      com.model-opt-vllm.role="serve-main" \
      com.model-opt-vllm.engine="vllm==0.25.0" \
      com.model-opt-vllm.rulings="Gate-1: Matrix IMAGE 1, D1, D4"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_BUILD_DATE=${BUILD_DATE} \
    HF_HOME=/hf-cache \
    PATH=/opt/venv/bin:${PATH}

# --- system deps -------------------------------------------------------------
# python3.12 (+venv) and the media runtime for torchcodec / opencv-python-headless:
#   ffmpeg (decode), libgl1 + libglib2.0-0 (opencv shared libs).
# apt versions are NOT hard-pinned (Ubuntu archive drift removes old versions and would break
# deterministic rebuilds) — the AS-BUILT versions are recorded in /etc/image-manifest.json.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Isolated venv: sidesteps PEP-668 (externally-managed-environment on Ubuntu 24.04) and keeps
# the vLLM install self-contained. The venv bin is first on PATH (see ENV above).
RUN python3.12 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade "pip>=25.0" "setuptools>=77.0.3,<81.0.0" wheel

# --- exact pin graph ---------------------------------------------------------
COPY constraints.txt /etc/constraints.txt

# --- install vLLM 0.25.0 under constraints -----------------------------------
# vLLM 0.25.0's wheel requires_dist carries the whole cu13 graph (torch/flashinfer/cutlass-dsl/
# quack/humming/tokenspeed/torchcodec/opencv/...) as == or >= specs; constraints.txt pins every
# floor EXACTLY. The quack-kernels==0.5.0 pin is load-bearing: without it the resolver picks
# quack 0.6.1 -> cutlass-dsl 4.6.0, which collides with vLLM's own cutlass-dsl==4.5.2 and the
# install fails outright (see constraints.txt).
RUN pip install --no-cache-dir -c /etc/constraints.txt "vllm==0.25.0"

# --- build-time assertion: forbidden packages --------------------------------
# Matrix IMAGE 1: the vLLM 0.25.0 wheel depends on none of these (verified in requires_dist).
# ray is not a dependency; nvidia-modelopt is quantize-time-only (lives in quant-offline);
# flash-attn is embedded inside vLLM via a CMake external project (no PyPI flash-attn wheel).
RUN python - <<'PY'
import importlib.util, sys
banned = {"ray": "ray", "modelopt": "nvidia-modelopt", "flash_attn": "flash-attn"}
present = [dist for mod, dist in banned.items() if importlib.util.find_spec(mod) is not None]
if present:
    sys.exit("BUILD FAILED — forbidden package(s) present in serve-main: " + ", ".join(present))
print("ASSERT OK: no ray / no nvidia-modelopt / no flash-attn in serve-main")
PY

# --- provenance manifest -----------------------------------------------------
# /etc/image-manifest.json = image identity + full direct-pin list (parsed from constraints.txt) +
# base image + BUILD_DATE + as-built apt versions. This is the artifact the Gate-2 lock step
# reconciles against the post-build `pip freeze`.
RUN python - <<'PY'
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
    "image": "model-opt-vllm/serve-main",
    "variant": "A",
    "role": "Qwen3.6-35B-A3B generation + Qwen3-VL-Embedding-8B pooling",
    "base_image": "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    "engine": "vllm==0.25.0",
    "platform": "linux/amd64",
    "python": "3.12",
    "cuda_lineage": "13.0.2",
    "build_date": os.environ.get("IMAGE_BUILD_DATE", "unknown"),
    "direct_pins": sorted(pins),
    "apt_packages": apt_versions(["ffmpeg", "libgl1", "libglib2.0-0", "python3.12"]),
    "forbidden_absent": ["ray", "nvidia-modelopt", "flash-attn"],
    "rulings": "Gate-1 adjudications: Matrix IMAGE 1, D1, D4",
}
with open("/etc/image-manifest.json", "w") as o:
    json.dump(manifest, o, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

# The config launchers (configs/*.sh) invoke `docker run --entrypoint vllm ...`; the baked
# ENTRYPOINT below just makes `docker run <image> serve <model> ...` work standalone too.
ENTRYPOINT ["vllm"]
