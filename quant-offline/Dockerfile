# syntax=docker/dockerfile:1
###############################################################################
# quant-offline  —  NVIDIA ModelOpt offline quantization job  (shared by BOTH variants)
# -----------------------------------------------------------------------------
# This is a JOB runtime, NOT a serving image. It runs the PTQ pipeline
# (mtq.quantize -> calibration -> export_hf_checkpoint); the exported checkpoint is then served
# back through the serving images. Quantization scripts (e.g. handoff/scale_scan.py, an FP8
# thinker-only recipe) are VOLUME-MOUNTED at runtime — they are deliberately NOT COPY'd here so
# the image stays script-agnostic and the scripts can iterate without rebuilds.
#
# Base   : nvidia/cuda:13.0.2-devel-ubuntu24.04  (devel = nvcc/toolkit for modelopt CUDA exts;
#          same CUDA/torch anchor as the serving images — Matrix IMAGE 3)
# Anchor : linux/amd64 · py3.12 · CUDA 13.0.2 · torch 2.11.0 · NVIDIA driver >=535 (cu13; R10)
# Rulings: Gate-1 Matrix IMAGE 3, D2 (quantization). transformers is decoupled from the serving
#          line on purpose: nvidia-modelopt[hf] caps transformers <5.10 (verified requires_dist).
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     [--build-arg PIP_INDEX_URL=<prod-mirror-url>] \
#     -t model-opt-vllm/quant-offline:modelopt-0.45.0 .
#
# Run (scripts + weights mounted at runtime; requires an sm90 GPU host):
#   docker run --rm --gpus all --ipc=host \
#     -v "$PWD/handoff":/work -v "$HF_HOME":/hf-cache -e HF_HOME=/hf-cache \
#     model-opt-vllm/quant-offline:modelopt-0.45.0 \
#     python /work/scale_scan.py ...
#
# Notes: no :latest tags; NO runtime pip; PIP_INDEX_URL is a prod-mirror hook — NO creds baked.
###############################################################################

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# Re-declare with defaults: a pre-FROM ARG default does not propagate into the build stage.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="model-opt-vllm quant-offline" \
      org.opencontainers.image.description="NVIDIA ModelOpt 0.45.0 offline PTQ job runtime (shared by Variant A and B); scripts mounted at runtime" \
      org.opencontainers.image.base.name="nvidia/cuda:13.0.2-devel-ubuntu24.04" \
      org.opencontainers.image.version="modelopt-0.45.0" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="model-opt-vllm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.model-opt-vllm.variant="shared" \
      com.model-opt-vllm.role="quant-offline" \
      com.model-opt-vllm.engine="nvidia-modelopt[hf]==0.45.0" \
      com.model-opt-vllm.rulings="Gate-1: Matrix IMAGE 3, D2"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_BUILD_DATE=${BUILD_DATE} \
    HF_HOME=/hf-cache \
    PATH=/opt/venv/bin:${PATH}

# --- system deps: python3.12 (+venv), git (VCS installs / HF), ca-certificates -------------
# apt versions unpinned (archive drift) -> recorded in /etc/image-manifest.json.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      git \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade "pip>=25.0" "setuptools>=77.0.3,<81.0.0" wheel

COPY constraints.txt /etc/constraints.txt

# --- install the PTQ toolchain under constraints -----------------------------
# The [hf] extra pulls the HuggingFace integration (transformers loader/export). transformers is
# pinned 5.8.1 (newest 5.8.x): modelopt[hf] caps <5.10 and 5.9.x is avoided (loader break).
RUN pip install --no-cache-dir -c /etc/constraints.txt \
      torch==2.11.0 \
      "nvidia-modelopt[hf]==0.45.0" \
      transformers==5.8.1 \
      datasets==5.0.0 \
      accelerate==1.14.0

# --- sanity assertion: the toolchain imports and versions are exactly as pinned ------------
# Versions come from importlib.metadata (authoritative for the installed distribution) rather
# than module __version__ attributes (which may be absent). Light top-level imports confirm the
# packages actually load; the PTQ API (modelopt.torch.quantization) is exercised at RUNTIME on a
# GPU host, not probed at build.
RUN python - <<'PY'
import sys
from importlib.metadata import version as dist_version
import torch, transformers, datasets, accelerate, modelopt  # confirm the toolchain imports
expect = {
    "torch": "2.11.0",
    "transformers": "5.8.1",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "nvidia-modelopt": "0.45.0",
}
got = {pkg: dist_version(pkg) for pkg in expect}
bad = {pkg: got[pkg] for pkg, want in expect.items() if not got[pkg].startswith(want)}
if bad:
    sys.exit("BUILD FAILED — quant toolchain version mismatch: " + repr(bad))
print("ASSERT OK: quant toolchain pinned as expected:", got)
PY

# --- provenance manifest -----------------------------------------------------
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
    "image": "model-opt-vllm/quant-offline",
    "variant": "shared",
    "role": "NVIDIA ModelOpt offline PTQ job (scripts volume-mounted at runtime)",
    "base_image": "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    "engine": "nvidia-modelopt[hf]==0.45.0",
    "platform": "linux/amd64",
    "python": "3.12",
    "cuda_lineage": "13.0.2",
    "build_date": os.environ.get("IMAGE_BUILD_DATE", "unknown"),
    "direct_pins": sorted(pins),
    "install_extras": ["nvidia-modelopt[hf]"],
    "apt_packages": apt_versions(["python3.12", "git"]),
    "rulings": "Gate-1 adjudications: Matrix IMAGE 3, D2",
}
with open("/etc/image-manifest.json", "w") as o:
    json.dump(manifest, o, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

WORKDIR /work
# No baked script/ENTRYPOINT: PTQ scripts are mounted into /work at runtime and invoked as
#   `docker run ... model-opt-vllm/quant-offline:modelopt-0.45.0 python /work/<script>.py ...`
