# model-opt-vllm — Inference Container Modernization

Reproducible Docker images that replace the legacy "base image + post-hoc upgrade script"
pattern with the **newest mutually compatible** vLLM serving stack for **NVIDIA Hopper (sm90,
H100/H200)**. Two serving **topology variants** are built side-by-side (Human Amendment,
2026-07-12) plus one shared offline-quantization image. The **Gate-3 cycle (ruled 2026-07-14,
`docs/gate3-adjudication.md`)** adds two engine standalones: **variant-c/serve-sglang** (SGLang
0.5.15.post1 — hosts all three models) and **variant-d/serve-trtllm** (TensorRT-LLM 1.2.1 —
**engine infrastructure only**: it can serve *none* of the three hosted models; ruling D8(a)).
A single-environment coexistence image was **proven impossible** at every version level and is
not built (ruling D7; evidence in the adjudication doc §5).

**Anchor (every image):** `linux/amd64` · Python 3.12 · CUDA 13.0.2 lineage · `torch==2.11.0`
(the default PyPI wheel is cu13) · sm90 first-class · **NVIDIA driver must be in one of the
enumerated branches 535 / 550 / 565 / 570 / 575** (the images' `NVIDIA_REQUIRE_CUDA` lists those
branches exactly — a 545.x or 560.x driver fails; Risk R10, see `docs/migration-notes.md`).
*Gate-3 exception (ruling D9):* `serve-sglang` keeps this anchor exactly; `serve-trtllm`
deviates by engine necessity to **torch 2.9.1+cu130** (TensorRT-LLM's own window) on the same
CUDA-13 lineage and base image — per-image driver notes in `docs/gate3-adjudication.md` §2.

**Provenance.** Pins are locked from the Gate-1 adjudications (`Matrix IMAGE 1/2/3`, decisions
D1–D5, corrections COR-1…8), from vLLM `requirements/{cuda,common}.txt` read at tags **v0.25.0**
and **v0.24.0**, and from PyPI resolution performed **2026-07-12**. **No GPU is available this
cycle** — all runtime behavior is code/doc-verified only; every serve profile carries a
**BLOCKED-protocol** header and refuses to launch without a visible GPU rather than fabricate a
run. No `:latest` tags anywhere; **no runtime pip** (all installs happen at build).

<img width="786" height="305" alt="Screenshot 2026-10-12 at 2 02 17 AM" src="https://github.com/user-attachments/assets/1efc7bf9-12e9-4d87-af12-5ed520c80bef" />

---

## 1. Images at a glance

| Directory | Image (suggested tag) | Base | Engine | Hosts |
|---|---|---|---|---|
| `variant-a/serve-main` | `model-opt-vllm/serve-main:vllm-0.25.0` | `nvidia/cuda:13.0.2-devel-ubuntu24.04` | **vLLM 0.25.0** | Qwen3.6-35B-A3B (gen) · Qwen3-VL-Embedding-8B (pooling) |
| `variant-a/serve-omni` | `model-opt-vllm/serve-omni:vllm-omni-0.24.1` | `vllm/vllm-openai:v0.24.0` | vLLM 0.24.0 + **vllm-omni**@`b9e9d23` | Qwen3-Omni-30B-A3B-Instruct |
| `variant-b/serve-unified` | `model-opt-vllm/serve-unified:vllm-0.24.0-omni` | `vllm/vllm-openai:v0.24.0` | vLLM 0.24.0 + **vllm-omni**@`b9e9d23` | **all three** (one container per model) |
| `quant-offline` | `model-opt-vllm/quant-offline:modelopt-0.45.0` | `nvidia/cuda:13.0.2-devel-ubuntu24.04` | **nvidia-modelopt[hf] 0.45.0** | ModelOpt PTQ job (shared by both variants) |
| `variant-c/serve-sglang` | `model-opt-vllm/serve-sglang:sglang-0.5.15.post1` | `nvidia/cuda:13.0.2-devel-ubuntu24.04` | **SGLang 0.5.15.post1** (torch 2.11.0 anchor) | **all three** (one container per model) — incl. the NVFP4 artifact (Marlin on sm90) |
| `variant-d/serve-trtllm` | `model-opt-vllm/serve-trtllm:trtllm-1.2.1` | `nvidia/cuda:13.0.2-devel-ubuntu24.04` | **TensorRT-LLM 1.2.1** (PyTorch backend; torch 2.9.1+cu130, D9) | **NONE — engine infrastructure (D8a)**: all three hosted models are explicit exclusions (manifest + `handoff/trtllm_smoke.md`) |

`quant-offline` is shared by both variants. Training (`verl`/Ray) is **out** of this deliverable:
verl v0.8.0 ships its own maintained image (`verlai/verl:vllm020.latest`) on the same
torch/CUDA anchor — use that pointer instead of duplicating it (D1).

Each image writes **`/etc/image-manifest.json`** at build (image identity, full direct-pin list,
base image, `BUILD_DATE`, as-built apt versions) and standard **OCI labels**.

---

## 2. Build commands

Every image accepts `--build-arg BUILD_DATE` (stamped into OCI labels + manifest) and
`--build-arg PIP_INDEX_URL` (a **prod-mirror hook** — point it at your internal mirror; **no
credentials are ever baked** into the image; pip uses it at build only). Always build for
`linux/amd64`.

```bash
# --- Variant A / serve-main  (vLLM 0.25.0) ---
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/serve-main:vllm-0.25.0 \
  variant-a/serve-main

# --- Variant A / serve-omni  (vLLM 0.24.0 base + vllm-omni @ commit) ---
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/serve-omni:vllm-omni-0.24.1 \
  variant-a/serve-omni

# --- Variant B / serve-unified  (single image, all three models) ---
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/serve-unified:vllm-0.24.0-omni \
  variant-b/serve-unified

# --- quant-offline  (ModelOpt PTQ job; shared) ---
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/quant-offline:modelopt-0.45.0 \
  quant-offline

# --- Variant C / serve-sglang  (SGLang 0.5.15.post1; Gate-3 D6) ---
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/serve-sglang:sglang-0.5.15.post1 \
  variant-c/serve-sglang

# --- Variant D / serve-trtllm  (TensorRT-LLM 1.2.1 infrastructure; Gate-3 D8a) ---
# Uses TWO additional build-time indexes (mirror-hookable build-args, no credentials):
#   PYTORCH_CU130_INDEX_URL (torch 2.9.1+cu130) and NVIDIA_PYPI_INDEX_URL (the trtllm wheel).
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/serve-trtllm:trtllm-1.2.1 \
  variant-d/serve-trtllm

# Prod mirror (no creds in the URL): add to any build above:
#   --build-arg PIP_INDEX_URL=https://<your-mirror>/simple
```

> BuildKit is required (the Dockerfiles use `# syntax=docker/dockerfile:1` heredocs for the
> manifest/assertion steps). It is on by default in modern Docker; otherwise `export
> DOCKER_BUILDKIT=1`.

**Build-time guards you will see in the logs:**
- `serve-main` asserts **no ray / no nvidia-modelopt / no flash-attn** (fails the build if any
  is present — Matrix IMAGE 1).
- `serve-omni` and `serve-unified` run a **before/after core-pin guard**: the build **fails if
  the vllm-omni install moved `vllm` / `torch` / `transformers` / `flashinfer`**. It prints the
  full `pip freeze` diff but decides with `comm -23` (flags only *moved/removed* pre-existing
  pins), so legitimately-added omni deps like `torchsde` do not false-fail the build.
- `quant-offline` asserts the toolchain imports and versions match the pins exactly.

---

## 3. Variant A vs Variant B — trade-off

| Dimension | **Variant A** (serve-main + serve-omni) | **Variant B** (serve-unified) |
|---|---|---|
| Images to build/operate | **2** serving images | **1** serving image |
| Generation + embedding engine | **vLLM 0.25.0** (newest gen+embedding fixes) | vLLM 0.24.0 (forfeits 0.25.0's fixes) |
| Omni engine | vLLM 0.24.0 + vllm-omni | vLLM 0.24.0 + vllm-omni (same) |
| Server-side **video-file** decode | OpenCV **+ torchcodec + PyNvVideoCodec** (0.25.0) | **OpenCV only** — no torchcodec/PyNvVideoCodec (**PARITY CAVEAT**) |
| MTP spec decode (Qwen3.6) | supported (floor v0.17.0, COR-2) | supported (floor v0.17.0, COR-2) |
| ModelOpt NVFP4 serving | supported (`--quantization modelopt`) | supported (same) |
| Ops surface | more images, freshest engine | one image, simpler rollout |
| Best when | you want the newest 0.25.0 gen/embedding path | you want a single unified image and can accept the 0.24.0 delta |

### Engine trade-off (Gate-3 additions, ruled 2026-07-14)

| Dimension | vLLM (A/B, as shipped) | **SGLang (variant-c)** | **TensorRT-LLM (variant-d)** |
|---|---|---|---|
| Hosted models | all three | **all three** (code-verified at the tag; runtime BLOCKED like everything else) | **none** — infrastructure only (D8a; re-activation per R19) |
| torch / CUDA | 2.11.0 / 13.0.2 (anchor) | 2.11.0 / 13.0.2 (**anchor held**, D9) | **2.9.1+cu130** / 13.x (D9 deviation; needs the pytorch cu130 + pypi.nvidia.com build indexes) |
| NVFP4 artifact on sm90 | `modelopt_mixed` via Marlin (D2) | `modelopt_mixed` via Marlin, floor 0.5.15 | unparseable + sm100-gated — excluded |
| Embedding runner | `--runner pooling` (D5) | `--is-embedding`; pooler LAST+normalize in-model; **chat-template caveat** (gate3 §1a) | none exists in trtllm-serve |
| Omni audio | vllm-omni `--omni` | native arch; **`audio_url` shape only** (OpenAI `input_audio` not accepted — client shim) | arch absent |
| Spec decode (MTP head) | `{"method":"mtp"}`, prefix cache force-off (D3/R3) | EAGLE alias on native head; radix stays on (**canary-gated** — vLLM bug numbers don't transfer) | MTP is DeepSeek-only |
| FP8 KV on BF16 target | `--calculate-kv-scales` | **no calibration flag — ships OFF (R21)**; NVFP4 target unaffected | pytorch backend only; moot (no models) |
| Coexistence with the others | one env per engine — **single-env coexistence proven impossible** (D7; gate3 §5: llguidance/torch/outlines_core walls, 8 executed CONFLICT probes) | same | same |

**PARITY CAVEAT (binding, Human Amendment 2026-07-12).** vLLM **0.24.0 ships no
torchcodec/PyNvVideoCodec** (OpenCV `>=4.13.0` only); **0.25.0 added `torchcodec>=0.14` +
`PyNvVideoCodec==2.0.4`** (both verified at the tags). Server-side **video-file** decode is
**off our critical path** — production frames arrive **client-extracted** (brief call type B) —
so this does not affect our workloads. But anyone sending **raw video files** to a Variant-B
endpoint decodes them through a different backend than Variant A. Treat as a feature-parity
reconcile item, not a blocker.

---

## 4. Model / profile matrix

Serve profiles live in `configs/*.sh`. **The image is selected via the `IMAGE` env var**, so the
same profile runs on either variant. Each profile is a host-side `docker run` launcher with a
BLOCKED-protocol GPU preflight and the **load-bearing parity/decision pins baked in**.

| Model | Profile | Engine CLI | Valid `IMAGE` | Key pins / ruling |
|---|---|---|---|---|
| Qwen3-VL-Embedding-8B | `embedding-pooling.sh` | `vllm serve --runner pooling` | serve-main · serve-unified | pooler `LAST`+normalize; mm-kwargs `{"fps":2.0,"max_frames":64}`; EVS/video_pruning **unset** (D5, S7, #33204) |
| Qwen3-Omni-30B-A3B-Instruct | `omni-audio.sh` | `vllm-omni serve --omni` | serve-omni · serve-unified | BF16; pinned mm-kwargs (audio uses processor default — reconcile) (D2, D5) |
| Qwen3.6-35B-A3B (BF16) | `qwen36-default.sh` | `vllm serve` | serve-main · serve-unified | BF16; **prefix caching ON**; kv `fp8_e4m3` + `--calculate-kv-scales`; **spec decode OFF** (D3, D4) |
| Qwen3.6-35B-A3B-NVFP4 | `qwen36-nvfp4.sh` | `vllm serve --quantization modelopt` | serve-main · serve-unified | NVFP4 (modelopt_mixed); kv `fp8`; `--moe-backend marlin`; **PROMOTION-GATED** on `handoff/scale_scan.py` (D2, COR-6, R2) |
| Qwen3.6-35B-A3B (MTP) | `qwen36-mtp-profile.sh` | `vllm serve --speculative-config …` | serve-main · serve-unified | MTP `num_speculative_tokens=2`; **`--no-enable-prefix-caching` MANDATORY**; **BENCHMARK-GATED, off by default** (D3, R3, R11) |

**Default generation posture (D3):** `qwen36-default.sh` — prefix caching **ON**, spec decode
**OFF**. MTP is the *tested candidate*, shipped as an explicit opt-in profile, enabled only after
the composed benchmark grid passes on sm90.

**SGLang profiles (Gate-3 D6)** live in `configs/sglang/*.sh` — engine-specific launchers (the
`sglang serve` CLI differs from `vllm serve`; the `IMAGE` env-var swap works within an engine
family only). Container port is sglang's native 30000; host `PORT` defaults are staggered
(30000–30003) so vLLM and SGLang can serve side-by-side for A/B:

| Model | Profile | Key posture / caveats |
|---|---|---|
| Qwen3.6-35B-A3B (BF16) | `sglang/qwen36-default.sh` | radix cache ON (engine default); **no fp8 KV (R21)**; spec decode OFF |
| Qwen3.6-35B-A3B-NVFP4 | `sglang/qwen36-nvfp4.sh` | `modelopt_mixed` auto-detect; Marlin on sm90; kv `fp8_e4m3` (checkpoint scales); **PROMOTION-GATED** (R2 scan on your own copy) |
| Qwen3-VL-Embedding-8B | `sglang/embedding-pooling.sh` | `--is-embedding`; pooler LAST+normalize in-model; **chat-template unset — GPU-gate item**; D5/R8 parity gate before cutover |
| Qwen3-Omni-30B-A3B-Instruct | `sglang/omni-audio.sh` | thinker-only; audio via sglang `audio_url` (**OpenAI `input_audio` not accepted** — client shim) |
| Qwen3.6-35B-A3B (MTP) | `sglang/qwen36-mtp-profile.sh` | EAGLE on native MTP head, explicit draft path; radix at engine default; **BENCHMARK+CANARY-GATED (D3)** |

There are **no `configs/trtllm/` profiles** — ruling D8(a): variant-d hosts none of the three
models; see `handoff/trtllm_smoke.md` for what it *is* for.

**Example launches** (on a Hopper host with the NVIDIA Container Toolkit):

```bash
# Embedding pooling on Variant A serve-main:
IMAGE=model-opt-vllm/serve-main:vllm-0.25.0 PORT=8001 \
  ./configs/embedding-pooling.sh

# Same profile, Variant B unified image (identical flags, different engine version):
IMAGE=model-opt-vllm/serve-unified:vllm-0.24.0-omni PORT=8001 \
  ./configs/embedding-pooling.sh

# Omni (needs an image that ships vllm-omni):
IMAGE=model-opt-vllm/serve-omni:vllm-omni-0.24.1 PORT=8002 HF_TOKEN=hf_xxx \
  ./configs/omni-audio.sh

# Qwen3.6 default generation:
IMAGE=model-opt-vllm/serve-main:vllm-0.25.0 PORT=8003 \
  ./configs/qwen36-default.sh

# MTP profile, NVFP4 target (bench-gated; promotion-gated target):
IMAGE=model-opt-vllm/serve-main:vllm-0.25.0 SPEC_TARGET=nvfp4 PORT=8004 \
  ./configs/qwen36-mtp-profile.sh
```

Common env knobs (all profiles): `IMAGE` (required), `PORT` (default 8000), `HF_HOME`
(host cache dir), `HF_TOKEN` (passed through, never baked), `SKIP_GPU_PREFLIGHT=1` (override the
BLOCKED preflight on a validated host). Extra `vllm`/`vllm-omni` flags pass straight through:
everything after the profile name is appended to the serve command.

---

## 5. HF_HOME / model-cache volume guidance

Models are pulled from the Hugging Face Hub on first launch. Persist the cache on the **host** and
mount it in — the profiles already do this (`-v "$HF_HOME:/hf-cache" -e HF_HOME=/hf-cache`):

```bash
export HF_HOME="$HOME/.cache/huggingface"   # host-side cache (survives container restarts)
export HF_TOKEN=hf_xxx                       # required for gated Qwen repos; passed as env, NEVER baked
mkdir -p "$HF_HOME"
```

- Point `HF_HOME` at fast local NVMe with room for the 30–35B checkpoints (tens of GB each).
- The images set container `HF_HOME=/hf-cache`; the host dir is mounted there read-write so a
  weight downloaded once is reused by every profile and every container.
- **Never** bake `HF_TOKEN` into an image or a compose file committed to VCS — pass it at
  `docker run` time (the profiles forward it only when set).
- For air-gapped hosts, pre-populate `HF_HOME` and add `HF_HUB_OFFLINE=1` to the profile's extra
  args.

---

## 6. Deterministic-rebuild caveats

- **apt archive drift.** System packages (`ffmpeg`, `libgl1`, `git`, …) are installed with
  `--no-install-recommends` but **not version-pinned**: Ubuntu rotates its archive and hard apt
  pins would break rebuilds once old versions are dropped. The **as-built apt versions are
  recorded in `/etc/image-manifest.json`** for audit; treat that manifest as the record of truth
  for a given build.
- **PyPI pins are time-locked.** Resolver-derived pins (`torchcodec==0.14.0`,
  `fastsafetensors==0.3.3`, `nvidia-cudnn-frontend==1.26.0`, `nvtx==0.2.15`,
  `quack-kernels==0.5.0`, `opencv-python-headless==4.13.0.92`, and the quant image's
  `transformers==5.8.1` / `datasets==5.0.0` / `accelerate==1.14.0`) were resolved on
  **2026-07-12** as the *newest satisfying* versions. They are exact `==` pins, so rebuilds are
  reproducible; re-resolve deliberately (not accidentally) if you move the lock date.
- **`quack-kernels==0.5.0` is load-bearing, not cosmetic.** vLLM floors `quack-kernels>=0.3.3`,
  but `quack 0.5.1+` hard-require `nvidia-cutlass-dsl==4.6.0`, which collides with vLLM's own
  `nvidia-cutlass-dsl[cu13]==4.5.2`. Without the `0.5.0` pin, a bare `pip install vllm==0.25.0`
  **fails to resolve**. Do not "upgrade" it without re-checking the cutlass-dsl graph.
- **`opencv-python-headless` pinned to 4.13.0.92 (newest 4.x), not the 5.0.0.93 major bump.**
  OpenCV decode is parity-relevant; a major-version jump on that path is deliberately avoided.
  Reconcile against prod's OpenCV major line at Gate-2 before adopting 5.x.
- **vllm-omni is pinned by git commit** (`b9e9d236c3f78afd405119a5b686ebebeeb53984`, the v0.24.1
  line with the RMSNorm fix), not by a moving branch or PyPI (which lags at 0.24.0).
- **Internal 4-segment wheels** (`torch 2.9.1.7+cu130`, `flash_attn 2.8.3.1+cu130`, … on the
  unreachable private index) may carry internal patches (Risk R7). Prod cutover requires a
  wheel-diff reconciliation — a migration note, not solved here.
- **serve-sglang (Gate-3):** lock tooling must allow the one beta sglang itself pins
  (`flash-attn-4==4.0.0b15` — uv: name it directly + `--prerelease=explicit`; R15). The
  `sgl-flash-attn3` HF kernel is **pre-baked at build** and seeded into the mounted HF cache by
  the entrypoint (R14) — no runtime downloads. sglang wheels are `manylinux_2_34` (glibc ≥2.34).
- **serve-trtllm (Gate-3):** lock generation is **linux-only** (the `tensorrt` wheel-stub sdists
  hard-fail on macOS); resolution/install require the two extra indexes, and the build asserts
  `torch==2.9.1+cu130` with **zero cu12-lineage packages** (a pypi-only install silently lands
  cu12.8 torch — R18-adjacent trap, rejected at build).

---

## 7. Verification status & Gate-2 lock step

**Executed so far (2026-07-12, by the orchestrator — results in-repo):**

| Check | Result | Evidence |
|---|---|---|
| NVFP4 scale-integrity scan (R2, gates `qwen36-nvfp4.sh`) | **PASS — executed.** 30,841/30,841 F8_E4M3 `weight_scale` tensors (2,052.9 MB), zero NaN (0x7F/0xFF) / negative bytes | `handoff/results/scale_scan_result.2026-07-12.json` — re-run `handoff/scale_scan.py` on **your own downloaded copy** before prod cutover |
| Pin-graph resolvability, all four images (linux/amd64 · py3.12) | **PASS — executed** via `uv pip compile --python-platform x86_64-manylinux_2_28`; every `constraints.txt` pin lands at its exact version (zero resolver drift); cu13 lineage present; the vllm-omni commit **co-resolves with the frozen vLLM 0.24.0 core without moving it** | `*/resolved-lock.txt` next to each Dockerfile |
| Docker image builds + in-image `pip check`/freeze/import matrix, negative-index test | **PENDING** — blocked in the authoring environment (Docker VM disk); run the §2 build commands on any linux/amd64 host (or after raising Docker Desktop's disk to ≥160 GB) | this section, below |
| GPU runtime validation (parity, benchmarks, MTP grid, canaries) | **BLOCKED — needs sm90** | `handoff/README.md` |

**Gate-3 cycle (executed 2026-07-14, by the orchestrator):**

| Check | Result | Evidence |
|---|---|---|
| Gate-3 feasibility research (engine×model matrix, version sets, lever map, backend inputs) | **RULED 2026-07-14** — 5-agent fleet + ~30 orchestrator tag-reads + 2-checker fact-check (no headline claim refuted) | `docs/gate3-adjudication.md` |
| Single-env coexistence (vllm+omni+sglang+tensorrt_llm, one Python env) | **IMPOSSIBLE — executed**: 8 uv CONFLICT probes incl. both best-ever pairs + exhaustive pin walks (llguidance / torch / outlines_core structural walls) → **no variant-e** (D7) | `docs/gate3-adjudication.md` §5 + Appendix B |
| Pin-graph resolvability, variant-c (linux/amd64 · py3.12 · manylinux_2_34) | **PASS — executed** via `uv pip compile --prerelease=explicit`; 194 pins; **zero drift 24/24** constraints | `variant-c/serve-sglang/resolved-lock.txt` |
| Pin-graph resolvability, variant-d (linux/amd64 · py3.12; cu130 + nvidia indexes) | **PASS — executed in a linux/amd64 container**; 177 pins; **zero drift 20/20**; single cu13 lineage (`torch==2.9.1+cu130`, zero cu12 packages) | `variant-d/serve-trtllm/resolved-lock.txt` |
| variant-c / variant-d image builds + in-image assertion matrices | **PENDING** — same environment limitation as above; §2 commands are complete | this section |
| SGLang GPU items (per-profile smoke, embedding chat-template, MTP canaries re-run, R20/R21) | **BLOCKED — needs sm90** | `configs/sglang/*.sh` headers |
| TRT-LLM smoke + R19 re-activation checklist | **BLOCKED — needs linux builder / sm90** (image hosts none of the three models by ruling) | `handoff/trtllm_smoke.md` |

### Post-build freeze (the committed lock)

Constraints files pin the **direct** dependency graph. To freeze the **full transitive** graph,
capture `pip freeze` from **inside each freshly built image** and commit it as the lock — that
frozen file, reconciled against `/etc/image-manifest.json`, is the committed Gate-2 lock.

```bash
mkdir -p locks

# serve-main / quant-offline (venv on PATH):
docker run --rm --entrypoint bash model-opt-vllm/serve-main:vllm-0.25.0 \
  -lc 'pip freeze' > locks/serve-main.lock.txt

# serve-omni / serve-unified (base-image python):
docker run --rm --entrypoint bash model-opt-vllm/serve-unified:vllm-0.24.0-omni \
  -lc 'python3 -m pip freeze' > locks/serve-unified.lock.txt

# quant-offline:
docker run --rm --entrypoint bash model-opt-vllm/quant-offline:modelopt-0.45.0 \
  -lc 'pip freeze' > locks/quant-offline.lock.txt

# inspect the build manifest (identity + direct pins + as-built apt versions):
docker run --rm --entrypoint cat model-opt-vllm/serve-main:vllm-0.25.0 \
  /etc/image-manifest.json
```

At Gate-2 also run, on a real Hopper host: the vLLM wheel-contents inspection (Risk R4:
FA3-on-cu13 backend + `fp8_e4m3` KV), a GPU smoke test per profile with the degenerate-output
detector, a `handoff/scale_scan.py` re-run **on your own downloaded checkpoint copy** (the
upstream copy already passed — see the §7 status table), and the composed MTP benchmark grid
(gates the MTP profile, D3). None of these can run in the authoring environment (no GPU).

---

## 8. Repository layout

```
model-opt-vllm/
├── README.md                         # this file
├── variant-a/
│   ├── serve-main/{Dockerfile,constraints.txt,resolved-lock.txt}   # vLLM 0.25.0 (gen + embedding)
│   └── serve-omni/{Dockerfile,constraints.txt,resolved-lock.txt}   # vLLM 0.24.0 + vllm-omni
├── variant-b/
│   └── serve-unified/{Dockerfile,constraints.txt,resolved-lock.txt}  # single 0.24.0+omni image (all 3 models)
├── quant-offline/
│   ├── Dockerfile · constraints.txt · resolved-lock.txt            # ModelOpt PTQ job (shared)
│   └── scripts/                       # volume-mounted at runtime (never COPY'd)
│       ├── quantize_omni_thinker_fp8.py    # FP8 W8A8 thinker-only (off-vendor-path, eval-gated)
│       └── quantize_qwen36_fallback.py     # fallback only — primary is the NVIDIA pre-quant
├── variant-c/                        # Gate-3 (D6): SGLang standalone
│   └── serve-sglang/{Dockerfile,constraints.txt,resolved-lock.txt}  # sglang 0.5.15.post1 (all 3 models)
├── variant-d/                        # Gate-3 (D8a): TRT-LLM engine infrastructure
│   └── serve-trtllm/{Dockerfile,constraints.txt,resolved-lock.txt}  # trtllm 1.2.1 — hosts NONE of the 3 (exclusions in manifest)
├── configs/                          # serve profiles (IMAGE selected via env)
│   ├── embedding-pooling.sh
│   ├── omni-audio.sh
│   ├── qwen36-default.sh
│   ├── qwen36-nvfp4.sh
│   ├── qwen36-mtp-profile.sh
│   └── sglang/                       # Gate-3 (D6): SGLang launchers (engine-specific CLI; port 30000)
│       ├── qwen36-default.sh · qwen36-nvfp4.sh · embedding-pooling.sh
│       └── omni-audio.sh · qwen36-mtp-profile.sh
│                                     # NO configs/trtllm/ — ruling D8(a): variant-d hosts no models
├── handoff/                          # Validation pack: scale_scan.py + degenerate_detector.py (no GPU needed),
│   │                                 #   parity_embedding.py, omni_audio_check.py, gen_bench.py, specdec_grid.py,
│   │                                 #   capacity_probe.py (per-host capacity ceiling search),
│   │                                 #   canary_repros.md, trtllm_smoke.md (Gate-3: variant-d BLOCKED runbook + R19),
│   │                                 #   README.md (GPU/endpoint-BLOCKED — see handoff/README.md)
│   └── results/                      # executed verification evidence
│       └── scale_scan_result.2026-07-12.json   # NVFP4 scan: PASS, 30,841/30,841 tensors clean
└── docs/                             # executive-summary · design-decisions · migration-notes ·
                                      #   risk-register · claim-ledger  (see §9)
```

`resolved-lock.txt` files are **metadata-level cross-platform resolutions** (uv, linux/amd64 ·
py3.12) proving each pin graph resolves with zero drift — the canonical as-built lock remains the
in-image `pip freeze` captured above.

## 9. Documentation

| Doc | Contents |
|---|---|
| `docs/executive-summary.md` | ≤1 page: what got newer, what got faster (honestly), what we deliberately did not do |
| `docs/design-decisions.md` | Locked Version Matrix, rulings D1–D5, full rejection log, corrections COR-1…8 |
| `docs/migration-notes.md` | From the legacy image: flash-attn/ray/verl dispositions, driver branches, internal-wheel reconciliation, parity procedure, behavior deltas, rollout order |
| `docs/risk-register.md` | R1–R13 with current status (R2 closed-pass) + acknowledged residuals; Gate-3 additions R14–R21 |
| `docs/claim-ledger.md` | Complete evidence trail: 37 orchestrator-executed verifications + 81 agent claims with dispositions; Gate-3 execution rows (§8) |
| `docs/gate3-adjudication.md` | **Gate-3 (RULED 2026-07-14)**: engine×model×capability matrix (SGLang / TRT-LLM), candidate version sets, lever map, TRT-LLM backend decision, coexistence impossibility evidence (11 uv probes), rulings D6–D9 + COR-9…13 + R14–R21 |
