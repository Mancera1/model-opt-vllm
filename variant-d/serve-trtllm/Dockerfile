# syntax=docker/dockerfile:1
###############################################################################
# Variant D / serve-trtllm — TensorRT-LLM 1.2.1 ENGINE-INFRASTRUCTURE image
#                            (linux/amd64, sm90 Hopper) — D8(a): hosts NO hosted model
# -----------------------------------------------------------------------------
# *** EXPLICIT EXCLUSIONS (Gate-3 ruling D8(a), 2026-07-14 — docs/gate3-adjudication.md §1b) ***
#   ✗ Qwen/Qwen3.6-35B-A3B            — architecture absent from TensorRT-LLM 1.2.1
#   ✗ nvidia/Qwen3.6-35B-A3B-NVFP4    — W4A16_NVFP4 not in QuantAlgo enum; NVFP4 GEMM sm100+-only
#   ✗ Qwen/Qwen3-VL-Embedding-8B      — no /v1/embeddings route / pooling runner in trtllm-serve
#   ✗ Qwen/Qwen3-Omni-30B-A3B-Instruct — architecture absent
# This image exists as a pinned, audited TRT-LLM environment: (1) a pre-validated on-ramp for
# the day upstream ships support (re-activation trigger = risk R19), (2) the TRT-LLM half of
# GPU-day engine comparisons on models TRT-LLM DOES support (see handoff/trtllm_smoke.md).
# There are NO configs/trtllm model profiles, by design — silent scope-narrowing would be a
# defect; loud exclusion is the design.
#
# Base   : nvidia/cuda:13.0.2-devel-ubuntu24.04   (repo anchor base)
# Backend: PyTorch backend + trtllm-serve (D8; classic trtllm-build engines require a GPU at
#          build time and are impossible this cycle — gate3 §4).
# D9 deviations: torch 2.9.1+cu130 (NOT the 2.11.0 anchor; TRT-LLM's own window), transformers
#          4.57.3, numpy<2. Two ADDITIONAL build-time indexes (no credentials, mirror-hookable):
#          download.pytorch.org/whl/cu130 (cu13 torch flavor — plain-PyPI torch 2.9.1 is
#          cu12.8-lineage and is asserted ABSENT below) and pypi.nvidia.com (the 2.51 GB
#          tensorrt_llm wheel; bare linux_x86_64 tag installs under pip — R17).
# Rulings: Gate-3 D8(a), D9, COR-9, R17, R18, R19.
# Note   : single-engine image -> no multi-engine pin-move guard (invariant applies to images
#          installing more than one engine).
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     [--build-arg PIP_INDEX_URL=<prod-mirror-url>] \
#     [--build-arg PYTORCH_CU130_INDEX_URL=<mirror>/whl/cu130] \
#     [--build-arg NVIDIA_PYPI_INDEX_URL=<mirror-of-pypi.nvidia.com>] \
#     -t model-opt-vllm/serve-trtllm:trtllm-1.2.1 .
#
# Notes: no :latest tags; NO runtime pip; no credentials baked into any index URL.
###############################################################################

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_CU130_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG NVIDIA_PYPI_INDEX_URL=https://pypi.nvidia.com
ARG BUILD_DATE=unknown

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# Re-declare ARGs post-FROM (pre-FROM defaults do not propagate into the stage).
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_CU130_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG NVIDIA_PYPI_INDEX_URL=https://pypi.nvidia.com
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="model-opt-vllm serve-trtllm (Variant D)" \
      org.opencontainers.image.description="TensorRT-LLM 1.2.1 engine infrastructure (D8a): hosts NONE of the three hosted models — see exclusions in /etc/image-manifest.json" \
      org.opencontainers.image.base.name="nvidia/cuda:13.0.2-devel-ubuntu24.04" \
      org.opencontainers.image.version="trtllm-1.2.1" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="model-opt-vllm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.model-opt-vllm.variant="D" \
      com.model-opt-vllm.role="engine-infrastructure (hosts NONE of the three hosted models — D8a)" \
      com.model-opt-vllm.engine="tensorrt-llm==1.2.1" \
      com.model-opt-vllm.rulings="Gate-3: D8(a), D9, COR-9, R17, R18, R19"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_BUILD_DATE=${BUILD_DATE} \
    HF_HOME=/hf-cache \
    PATH=/opt/venv/bin:${PATH}

# --- system deps -------------------------------------------------------------
# OpenMPI: tensorrt-llm imports mpi4py at module load; runtime libs + dev headers (dev covers a
# potential mpi4py sdist fallback). libsndfile1 for soundfile. apt versions unpinned (R13
# pattern) — AS-BUILT versions recorded in /etc/image-manifest.json.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      openmpi-bin \
      libopenmpi-dev \
      libsndfile1 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade "pip>=25.0" "setuptools>=77.0.3,<81.0.0" wheel

# --- exact pin graph ---------------------------------------------------------
COPY constraints.txt /etc/constraints.txt

# --- install TensorRT-LLM under constraints, with the two extra indexes -------
# Index roles: PIP_INDEX_URL = primary (pypi.org or prod mirror); PYTORCH_CU130_INDEX_URL
# supplies torch/torchvision/torchao +cu130 local-version wheels (constraints pin the +cu130
# flavor EXACTLY — a plain-PyPI torch 2.9.1 would be cu12.8-lineage, R18-adjacent trap);
# NVIDIA_PYPI_INDEX_URL supplies the tensorrt_llm wheel (~2.51 GB) and tensorrt-cu13 libs.
RUN pip install --no-cache-dir -c /etc/constraints.txt \
      --extra-index-url "${PYTORCH_CU130_INDEX_URL}" \
      --extra-index-url "${NVIDIA_PYPI_INDEX_URL}" \
      "tensorrt-llm==1.2.1"

# --- build-time assertion matrix ----------------------------------------------
RUN pip check \
 && mkdir -p /tmp/cuda-stubs \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/tmp/cuda-stubs:/usr/local/cuda/lib64/stubs python - <<'PY'
from importlib.metadata import version
pins = {
    "tensorrt-llm": "1.2.1",
    "torch": "2.9.1+cu130",            # THE flavor guard (R18): plain 2.9.1 = cu12.8 lineage = FAIL
    "torchvision": "0.24.1+cu130",
    "transformers": "4.57.3",
    "tensorrt": "10.14.1.48.post1",
    "flashinfer-python": "0.6.4",
    "triton": "3.5.1",
    "xgrammar": "0.1.32",
    "nvidia-cutlass-dsl": "4.3.4",
    "nvidia-modelopt": "0.37.0",
    "nvidia-nccl-cu13": "2.27.7",
    "numpy": "1.26.4",
    "fastapi": "0.121.3",
}
bad = {n: (version(n), want) for n, want in pins.items() if version(n) != want}
assert not bad, f"BUILD FAILED — pin drift: {bad}"
# cu12 contamination guard: the cu13-flavored torch must NOT have pulled any -cu12 NVIDIA libs.
from importlib.metadata import distributions
cu12 = sorted({d.metadata["Name"] for d in distributions()
               if d.metadata["Name"] and "cu12" in d.metadata["Name"].lower()})
assert not cu12, f"BUILD FAILED — cu12-lineage packages present (R18): {cu12}"
import tensorrt      # bindings import (no GPU needed under driver stubs)
import tensorrt_llm  # engine import (loads MPI + TRT libs)
print("ASSERT OK: 13 core pins exact; zero cu12 packages; tensorrt",
      tensorrt.__version__, "| tensorrt_llm", tensorrt_llm.__version__)
PY
RUN rm -rf /tmp/cuda-stubs

# --- provenance manifest (exclusions are FIRST-CLASS here — D8a) ---------------
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
    "image": "model-opt-vllm/serve-trtllm",
    "variant": "D",
    "role": "engine-infrastructure (D8a): pinned TensorRT-LLM environment; hosts NONE of the three hosted models",
    "excluded_models": {
        "Qwen/Qwen3.6-35B-A3B": "architecture absent from TensorRT-LLM 1.2.1 (gate3-adjudication §1b)",
        "nvidia/Qwen3.6-35B-A3B-NVFP4": "W4A16_NVFP4 not in QuantAlgo enum; NVFP4 GEMM sm100+-only, no Hopper fallback",
        "Qwen/Qwen3-VL-Embedding-8B": "no embedding serving exists in trtllm-serve (no /v1/embeddings, no pooling runner)",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct": "architecture absent from TensorRT-LLM 1.2.1",
    },
    "reactivation_trigger": "R19: TRT-LLM stable gains Qwen3.5/3.6-multimodal support or an embedding path",
    "base_image": "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    "engine": "tensorrt-llm==1.2.1 (PyTorch backend + trtllm-serve; D8)",
    "platform": "linux/amd64",
    "python": "3.12",
    "cuda_lineage": "13.x (base 13.0.2; torch 2.9.1+cu130; tensorrt-cu13 10.14.1.48.post1)",
    "build_date": os.environ.get("IMAGE_BUILD_DATE", "unknown"),
    "direct_pins": sorted(pins),
    "apt_packages": apt_versions(["openmpi-bin", "libopenmpi-dev", "libsndfile1", "python3.12"]),
    "rulings": "Gate-3 adjudications: D8(a), D9, COR-9, R17, R18, R19",
}
with open("/etc/image-manifest.json", "w") as o:
    json.dump(manifest, o, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
print("\n*** D8(a): this image hosts NONE of the three hosted models — see excluded_models above ***")
PY

# trtllm-serve is the OpenAI-compatible front-end (completions/chat/responses; NO /v1/embeddings).
# No model profiles exist for this image (D8a) — see handoff/trtllm_smoke.md for the BLOCKED
# import/serve smoke runbook and the R19 re-activation checklist.
ENTRYPOINT ["trtllm-serve"]
