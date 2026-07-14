# syntax=docker/dockerfile:1
###############################################################################
# Variant C / serve-sglang  —  SGLang 0.5.15.post1 serving image  (linux/amd64, sm90 Hopper)
# -----------------------------------------------------------------------------
# Hosts (one model per container process — same operating model as Variant A/B):
#   * Qwen/Qwen3.6-35B-A3B            — MoE generation (BF16), hybrid-GDN radix cache
#   * nvidia/Qwen3.6-35B-A3B-NVFP4    — ModelOpt MIXED_PRECISION via modelopt_mixed (Marlin on sm90)
#   * Qwen/Qwen3-VL-Embedding-8B      — embedding server (--is-embedding; pooler LAST+normalize)
#   * Qwen/Qwen3-Omni-30B-A3B-Instruct — omni thinker, audio via sglang `audio_url` content part
#
# Base   : nvidia/cuda:13.0.2-devel-ubuntu24.04   (repo anchor base — same as serve-main)
# Anchor : linux/amd64 · py3.12 · CUDA 13.0.2 · torch 2.11.0 (sglang's own exact pin equals the
#          Gate-1 anchor; default PyPI wheel IS cu13) · sm90 first-class · driver per R10 family.
# Rulings: Gate-3 adjudication (docs/gate3-adjudication.md, RULED 2026-07-14) — D6 (this image),
#          D9 (anchor held; no deviation needed for sglang), COR-10 (sglang-kernel rename),
#          R14 (sgl-flash-attn3 HF kernel PRE-BAKED below — no runtime downloads),
#          R15 (flash-attn-4==4.0.0b15 is a beta pin inside stable sglang),
#          R20 (sm90 GDN decode = triton/FP32-state — GPU-gated bench item),
#          R21 (BF16-target FP8-KV lever ships OFF — no --calculate-kv-scales analogue).
# Note   : NO multi-engine pin-move guard in this image — the comm -23 guard pattern applies to
#          images installing more than one engine (serve-omni/serve-unified); this image installs
#          exactly one engine under a fully exact constraint set.
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     [--build-arg PIP_INDEX_URL=<prod-mirror-url>] \
#     -t model-opt-vllm/serve-sglang:sglang-0.5.15.post1 .
#
# Notes: no :latest tags; NO runtime pip (all installs at build); PIP_INDEX_URL is a
#        prod-mirror hook only — NO credentials are ever baked into the image.
###############################################################################

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# ARGs must be re-declared after FROM to be visible in this stage (a pre-FROM ARG default does
# NOT propagate into the build stage).
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="model-opt-vllm serve-sglang (Variant C)" \
      org.opencontainers.image.description="SGLang 0.5.15.post1 serving: Qwen3.6-35B-A3B (BF16+NVFP4), Qwen3-VL-Embedding-8B, Qwen3-Omni-30B-A3B-Instruct on sm90 Hopper" \
      org.opencontainers.image.base.name="nvidia/cuda:13.0.2-devel-ubuntu24.04" \
      org.opencontainers.image.version="sglang-0.5.15.post1" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="model-opt-vllm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.model-opt-vllm.variant="C" \
      com.model-opt-vllm.role="serve-sglang" \
      com.model-opt-vllm.engine="sglang==0.5.15.post1" \
      com.model-opt-vllm.rulings="Gate-3: D6, D9, COR-10, R14, R15, R20, R21"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_BUILD_DATE=${BUILD_DATE} \
    HF_HOME=/hf-cache \
    PATH=/opt/venv/bin:${PATH}

# --- system deps -------------------------------------------------------------
# python3.12 (+venv); ffmpeg for torchcodec video decode; libsndfile1 for soundfile (omni audio
# input); libgl1 + libglib2.0-0 for image-processing shared libs. apt versions are NOT hard-pinned
# (Ubuntu archive drift) — AS-BUILT versions are recorded in /etc/image-manifest.json (R13 pattern).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      ffmpeg \
      libsndfile1 \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Isolated venv: sidesteps PEP-668 on Ubuntu 24.04; venv bin is first on PATH (ENV above).
RUN python3.12 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade "pip>=25.0" "setuptools>=77.0.3,<81.0.0" wheel

# --- exact pin graph ---------------------------------------------------------
COPY constraints.txt /etc/constraints.txt

# --- install SGLang under constraints ----------------------------------------
# sglang's wheel metadata pins its stack exactly (torch/flashinfer/transformers/kernel/...);
# constraints.txt restates those pins and freezes the few ranged surfaces (torchvision,
# llguidance, quack-kernels, kernels, torch-memory-saver, mistral-common). flash-attn-4 is
# named DIRECTLY because it is an exact PRERELEASE pin (==4.0.0b15, R15): pip permits a
# prerelease when the requirement itself names one. quack-kernels==0.5.0 is load-bearing
# exactly as in serve-main: 0.5.1+ requires cutlass-dsl 4.6.x vs sglang's pinned 4.5.2.
RUN pip install --no-cache-dir -c /etc/constraints.txt \
      "sglang==0.5.15.post1" "flash-attn-4==4.0.0b15"

# --- R14: PRE-BAKE the sgl-flash-attn3 HF kernel ------------------------------
# sglang declares `[tool.kernels.dependencies] "kernels-community/sgl-flash-attn3" = 1` in its
# pyproject: the `kernels` library fetches that repo from the HF Hub AT FIRST USE — a runtime
# network download this repo's invariants forbid. We snapshot it into the image here and seed the
# runtime cache from it in the entrypoint (the HF cache itself is a runtime mount).
# Version-map caveat (R14 residual): [tool.kernels] version "1" is resolved to a git revision by
# the kernels library at runtime; we bake revision "v1" if it exists, else the default branch,
# and record what was baked in /etc/sgl-kernel-bake.json for reconciliation at the GPU gate.
RUN HF_HOME=/opt/hf-kernels-bake python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
repo = "kernels-community/sgl-flash-attn3"
baked_rev = None
path = None
for rev in ("v1", None):
    try:
        path = snapshot_download(repo, revision=rev)
        baked_rev = rev or "default-branch"
        break
    except Exception as e:
        print(f"snapshot_download({repo!r}, revision={rev!r}) failed: {e}")
if path is None:
    raise SystemExit("BUILD FAILED — could not pre-bake sgl-flash-attn3 (R14)")
nfiles = sum(len(fs) for _, _, fs in os.walk(path))
if nfiles == 0:
    raise SystemExit("BUILD FAILED — sgl-flash-attn3 snapshot is empty (R14)")
info = {"repo": repo, "requested_revision": baked_rev,
        "snapshot_path": path, "file_count": nfiles}
with open("/etc/sgl-kernel-bake.json", "w") as fh:
    json.dump(info, fh, indent=2, sort_keys=True)
print("R14 PRE-BAKE OK:", json.dumps(info))
PY

# --- build-time assertion matrix ----------------------------------------------
# (1) pip's own resolver consistency; (2) every core pin at its exact constraint version via
# importlib.metadata (no CUDA needed); (3) torch imports (CPU-safe); (4) sglang package imports
# under the CUDA driver STUBS shipped in the -devel base (no GPU on the builder — the stub
# provides libcuda.so.1 for link-time import only; real driver comes from the runtime host).
RUN pip check \
 && mkdir -p /tmp/cuda-stubs \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/tmp/cuda-stubs:/usr/local/cuda/lib64/stubs python - <<'PY'
from importlib.metadata import version
pins = {
    "sglang": "0.5.15.post1", "torch": "2.11.0", "torchvision": "0.26.0",
    "torchaudio": "2.11.0", "transformers": "5.12.1",
    "flashinfer-python": "0.6.12", "flashinfer-cubin": "0.6.12",
    "sglang-kernel": "0.4.4", "flash-attn-4": "4.0.0b15",
    "torchao": "0.17.0", "xgrammar": "0.2.1", "torchcodec": "0.11.1",
    "llguidance": "0.7.30", "quack-kernels": "0.5.0", "kernels": "0.14.1",
    "nvidia-cutlass-dsl": "4.5.2",
}
bad = {n: (version(n), want) for n, want in pins.items() if version(n) != want}
assert not bad, f"BUILD FAILED — pin drift: {bad}"
import torch  # CPU-safe import
import sglang  # top-level import under CUDA stubs
import sglang.srt.server_args  # server-args module (flag surface the profiles rely on)
print("ASSERT OK: 16 core pins exact; torch/sglang import under stubs;",
      "torch", torch.__version__, "| sglang", sglang.__version__)
PY
RUN rm -rf /tmp/cuda-stubs

# --- entrypoint wrapper: seed the runtime HF cache with the baked kernel (R14) -
# HF_HOME is a runtime mount (repo invariant), so build-time cache content is shadowed at run
# time; this shim copies the baked snapshot in (no-clobber, idempotent) then execs sglang.
COPY <<'EOF' /usr/local/bin/sglang-entry
#!/usr/bin/env bash
set -euo pipefail
if [ -d /opt/hf-kernels-bake/hub ]; then
  if mkdir -p "${HF_HOME:-/hf-cache}/hub" 2>/dev/null \
     && cp -an /opt/hf-kernels-bake/hub/. "${HF_HOME:-/hf-cache}/hub/" 2>/dev/null; then
    :
  else
    echo "WARN (R14): could not seed sgl-flash-attn3 kernel into ${HF_HOME:-/hf-cache} —" \
         "first use may attempt a network fetch" >&2
  fi
fi
exec sglang "$@"
EOF
RUN chmod +x /usr/local/bin/sglang-entry

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
kernel_bake = None
try:
    kernel_bake = json.load(open("/etc/sgl-kernel-bake.json"))
except Exception:
    pass
manifest = {
    "image": "model-opt-vllm/serve-sglang",
    "variant": "C",
    "role": "SGLang serving of all three hosted models (one model per container process)",
    "base_image": "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    "engine": "sglang==0.5.15.post1",
    "platform": "linux/amd64",
    "python": "3.12",
    "cuda_lineage": "13.0.2",
    "build_date": os.environ.get("IMAGE_BUILD_DATE", "unknown"),
    "direct_pins": sorted(pins),
    "apt_packages": apt_versions(["ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0", "python3.12"]),
    "sgl_flash_attn3_bake": kernel_bake,
    "rulings": "Gate-3 adjudications: D6, D9, COR-10, R14, R15, R20, R21",
}
with open("/etc/image-manifest.json", "w") as o:
    json.dump(manifest, o, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

# The config launchers (configs/sglang/*.sh) run `docker run <image> serve --model-path ...`;
# the wrapper seeds the kernel cache (R14) and execs the `sglang` CLI with the given args.
ENTRYPOINT ["/usr/local/bin/sglang-entry"]
