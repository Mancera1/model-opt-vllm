# syntax=docker/dockerfile:1
###############################################################################
# Variant A / serve-omni  —  vLLM-Omni serving image  (linux/amd64, sm90 Hopper)
# -----------------------------------------------------------------------------
# Hosts: Qwen/Qwen3-Omni-30B-A3B-Instruct  (omni; production call type A = raw audio -> features)
#
# Base : vllm/vllm-openai:v0.24.0   (vLLM-Omni's own upstream base — Matrix IMAGE 2;
#        config-blob-verified inside: CUDA 13.0.2 / torch 2.11.0 / flashinfer 0.6.12)
# Omni : vllm-omni @ commit b9e9d236c3f78afd405119a5b686ebebeeb53984  (the v0.24.1 line —
#        carries an RMSNorm default-op numerics fix absent from PyPI 0.24.0; pinned by commit)
# Rulings: Gate-1 D1 (topology), COR-8 (vllm-omni has NO python-level vllm/torch pin — coupling
#          is via the Docker base), R6 (vllm-omni x vllm-0.25.0 untested upstream -> omni isolated
#          on the 0.24.x line). BF16 default (D2); FP8-thinker quant is an offline, eval-gated recipe.
#
# Build:
#   docker build --platform linux/amd64 \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     [--build-arg PIP_INDEX_URL=<prod-mirror-url>] \
#     -t model-opt-vllm/serve-omni:vllm-omni-0.24.1 .
#
# Notes: no :latest tags; NO runtime pip; PIP_INDEX_URL is a prod-mirror hook — NO creds baked.
###############################################################################

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

FROM vllm/vllm-openai:v0.24.0

# Re-declare with defaults: a pre-FROM ARG default does not propagate into the build stage.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="model-opt-vllm serve-omni (Variant A)" \
      org.opencontainers.image.description="vLLM-Omni serving of Qwen3-Omni-30B-A3B-Instruct on the vllm/vllm-openai:v0.24.0 base (sm90 Hopper)" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.24.0" \
      org.opencontainers.image.version="vllm-omni-0.24.1+b9e9d23" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.vendor="model-opt-vllm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.model-opt-vllm.variant="A" \
      com.model-opt-vllm.role="serve-omni" \
      com.model-opt-vllm.engine="vllm==0.24.0 + vllm-omni@b9e9d23" \
      com.model-opt-vllm.rulings="Gate-1: Matrix IMAGE 2, D1, COR-8, R6"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_BUILD_DATE=${BUILD_DATE} \
    HF_HOME=/hf-cache

# --- system deps -------------------------------------------------------------
# git: required for the pinned-commit pip install of vllm-omni.
# ffmpeg + libsndfile1 + libgl1 + libglib2.0-0: omni audio/video decode (av, soundfile,
#   openai-whisper, opencv are omni deps). apt versions unpinned (archive drift) — recorded
#   in the manifest below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      libsndfile1 \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- freeze constraints for the six core packages ---------------------------
COPY constraints.txt /etc/constraints.txt

# --- GUARD 1/2: snapshot the core pins BEFORE the vllm-omni install ----------
# `pip freeze | grep` + sort produces a stable, comparable snapshot. See GUARD 2/2 for why the
# decision uses `comm -23` rather than a naive `diff`.
RUN python3 -m pip freeze | grep -E "^(vllm|torch|transformers|flashinfer)" | LC_ALL=C sort > /etc/pins_before.txt \
 && echo "=== core pins BEFORE vllm-omni install ===" \
 && cat /etc/pins_before.txt

# --- install vllm-omni at the pinned commit, under constraints ---------------
# constraints.txt freezes vllm/torch/torchvision/torchaudio/flashinfer-python/flashinfer-cubin.
# vllm-omni brings its own deps (diffusers, av, soundfile, whisper, ...) which install freely;
# only the six frozen packages are held. A constraint CONFLICT here (omni needing something
# incompatible with a frozen pin) fails the build immediately — a desirable fail-fast.
RUN python3 -m pip install --no-cache-dir -c /etc/constraints.txt \
      "vllm-omni @ git+https://github.com/vllm-project/vllm-omni@b9e9d236c3f78afd405119a5b686ebebeeb53984"

# --- GUARD 2/2: fail the build if the omni install MOVED any core pin --------
# Requirement (Gate-1 item 2): "build MUST FAIL if the omni install moved any of them."
# We print the full before/after diff for visibility, but DECIDE with `comm -23` (lines in
# BEFORE not in AFTER = a pre-existing core pin whose exact `name==ver` changed or was removed).
# A naive `diff` would false-fail here because vllm-omni LEGITIMATELY ADDS new torch*/vllm*
# packages (e.g. torchsde, and vllm-omni itself) that the `^(vllm|torch|...)` grep also matches;
# additions are not "moves" and must not abort a correct build. comm -23 flags moves only.
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
    "image": "model-opt-vllm/serve-omni",
    "variant": "A",
    "role": "Qwen3-Omni-30B-A3B-Instruct serving (call type A: raw audio -> features)",
    "base_image": "vllm/vllm-openai:v0.24.0",
    "engine": "vllm==0.24.0 + vllm-omni",
    "vllm_omni_commit": "b9e9d236c3f78afd405119a5b686ebebeeb53984",
    "platform": "linux/amd64",
    "python": "3.12",
    "cuda_lineage": "13.0.2",
    "build_date": os.environ.get("IMAGE_BUILD_DATE", "unknown"),
    "frozen_direct_pins": sorted(pins),
    "apt_packages": apt_versions(["git", "ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0"]),
    "rulings": "Gate-1 adjudications: Matrix IMAGE 2, D1, COR-8, R6",
}
with open("/etc/image-manifest.json", "w") as o:
    json.dump(manifest, o, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

# The config launcher (configs/omni-audio.sh) invokes `docker run --entrypoint vllm-omni ...`;
# the baked ENTRYPOINT below makes `docker run <image> serve <model> --omni ...` work standalone.
ENTRYPOINT ["vllm-omni"]
