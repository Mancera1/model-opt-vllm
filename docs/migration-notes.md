# Migration Notes — from the legacy image to this stack

Legacy baseline: CUDA 13 (cu130) · py3.12 · base torch `2.9.1.7+cu130` / flash_attn `2.8.3.1+cu130` /
flashinfer `0.6.3.1+cu130` / ray `2.52.1` / verl `0.7.0.7` / vllm `0.15+cu130`, then a **post-build
upgrade script** pulling from a private index to vllm `0.18.1+cu130` + torch `2.10.0+cu130` and
removing flash-attn. This repo replaces that pattern with build-time pins (no runtime pip, no
post-build mutation).

## What moves, what stays

| Component | Legacy (after upgrade script) | New | Notes |
|---|---|---|---|
| vLLM | 0.18.1+cu130 (internal wheel) | **0.25.0** (serve-main) / **0.24.0** (omni & variant-B) | 7 (resp. 6) upstream minors ahead; engine defaults changed — see "Behavior deltas" |
| torch | 2.10.0+cu130 (internal) | **2.11.0** (upstream cu13 default wheel) | Same CUDA 13.0.x lineage; one minor bump |
| CUDA | cu130 | cu13 (13.0.2) | unchanged lineage |
| flash-attn | removed by upgrade script | **absent as a pip package, by design** | vLLM embeds its own FA build via a CMake external project (`CMakeLists.txt:1402`); serve images build-assert no `flash_attn` module. Whether the embedded FA3 is *active* on the cu13 wheel is Gate-2/GPU-pending (R4) — the FlashInfer fallback's fp8-KV support is code-verified either way |
| flashinfer | 0.6.3.1+cu130 (internal) | **0.6.13** (serve-main) / 0.6.12 (0.24.0 images) | pinned by vLLM per release; `flashinfer-cubin` baked to avoid runtime kernel downloads |
| ray | 2.52.1 | **removed from serving images** | vLLM 0.25.0 has no ray dependency (verified: absent from all requirement files + pyproject); single-node TP uses multiprocessing. Ray returns only if you adopt multi-node serving or the verl image |
| verl | 0.7.0.7 (internal) | **out of these images** | use `verlai/verl:vllm020.latest` (verl v0.8.0's own image: vllm 0.20.2 + torch 2.11.0/cu130 + CUDA 13.0.2 + transformers 5.3.0 + flash_attn 2.8.3) — same torch/CUDA anchor as the serving images |
| transformers | (whatever the script left) | **5.13.1** (serve-main) / base-image version (0.24.0 images) / **5.8.1** (quant job) | modelopt[hf] caps transformers <5.10 → quantization isolated in its own image |
| private index | required at runtime-upgrade | **not required**; `--build-arg PIP_INDEX_URL` hook provided | point it at the prod mirror when building in prod |

## Hard prerequisites on the H100/H200 hosts

1. **Driver branch, not "≥535":** the cu13 images' `NVIDIA_REQUIRE_CUDA` enumerates driver branches
   **535 / 550 / 565 / 570 / 575** (each bounded `<branch+1`). A host on, e.g., 545.x or 560.x FAILS
   the constraint. Check `nvidia-smi --query-gpu=driver_version --format=csv` against that list.
2. NVIDIA Container Toolkit; `--gpus all --ipc=host` (profiles set these).
3. Fast NVMe `HF_HOME` volume (tens of GB per model; NVFP4 checkpoint is ~23.4 GB).

## Internal 4-segment wheels (R7 — do this before cutover)

The legacy wheels (`torch 2.9.1.7+cu130`, `flash_attn 2.8.3.1+cu130`, `verl 0.7.0.7`, …) live on a
private index unreachable from this project and **may carry internal patches**. Before replacing them
with upstream 2.11.0-era wheels:
- diff the internal wheel contents vs the upstream wheel at the matching base version (METADATA,
  RECORD, `.so` set, any `*.patch` provenance),
- identify the owning team and confirm the 4th-segment builds are rebuilds, not fixes,
- if fixes exist, re-evaluate them against torch 2.11.0 (they may be upstreamed or obsolete).
Dropping an internal fix silently is the failure mode this step exists to prevent.

## Feature-parity procedure (hard gate — production has a precomputed index)

Preprocessing is part of the model. All profiles pin `--mm-processor-kwargs` and pooler config; EVS
and `video_pruning_rate` stay unset. Before cutover:
1. On the **CURRENT production image**, run `handoff/parity_embedding.py --save-reference` over the
   fixed corpus (text + single images + frames-as-video) and `handoff/omni_audio_check.py
   --save-reference` for call type A.
2. On the NEW image(s), run both in compare mode. Gates: text cosine ≥ 0.999, multimodal ≥ 0.995,
   p99 ≥ 0.995; omni temp-0 exact/logprob tolerance + the ≤1 pt task-eval delta.
3. Also sample stored-index entries and compare against new-image outputs for the same inputs — the
   index, not the old engine, is the ultimate reference.
4. Known upstream caveat: vLLM pooling-runner video embeddings diverge from `qwen_vl_utils`
   (~0.92 cosine, issue #33204). Our gate is vs the *current image / stored index*, not the official
   implementation — do not "fix" preprocessing to match qwen_vl_utils, that would break the index.
5. Variant B only: server-side video-FILE decode uses OpenCV (no torchcodec/PyNvVideoCodec at
   0.24.0). Irrelevant for client-extracted frames; relevant if any caller sends raw video files.

## Behavior deltas to expect (0.18.1 → 0.24/0.25)

- Engine internals moved to the V2 model-runner generation; compilation + CUDA graphs are on by
  default (`enforce_eager=False`) — do not pass `--enforce-eager` in prod.
- Multimodal processor cache: ON by default (4 GiB LRU, host RAM, per-process duplication
  `gb × (api_servers + DP)`), keyed by content hash.
- Prefix caching: ON by default. **Keep MTP off wherever prefix caching is on** (open corruption
  #43559/#47194/#47087 on this exact model family).
- Serving quantized ModelOpt checkpoints needs no modelopt package at runtime (vLLM reads
  `hf_quant_config.json`; this checkpoint auto-selects `modelopt_mixed`).
- `--kv-cache-dtype fp8_e4m3` on Qwen3.6: BF16 target needs `--calculate-kv-scales`; the NVFP4
  checkpoint ships KV scales (leave the flag off).
- vLLM 0.25.0 pins its own ecosystem hard (torch==2.11.0, flashinfer==0.6.13): never "upgrade" a
  single component inside the image; move the whole vLLM version instead.
- `quack-kernels==0.5.0` is load-bearing (0.5.1+ requires cutlass-dsl 4.6.0, conflicting with vLLM's
  4.5.2) — a bare unconstrained install fails to resolve.

## Rollout order (suggested)

1. Build images on a linux/amd64 host (README §2); capture `pip freeze` locks + manifests (§7).
2. GPU smoke per profile + degenerate-output detector; wheel-contents FA3 inspection (R4).
3. Parity procedure above (gates embedding + omni).
4. `qwen36-default.sh` to production (BF16, caching ON, spec OFF).
5. `qwen36-nvfp4.sh` after re-running `scale_scan.py` on your own checkpoint copy (upstream copy
   already certified: 30,841/30,841 clean, 2026-07-12).
6. MTP profile only after `handoff/specdec_grid.py` + the 7 canary repros pass on sm90.
