# Design Decision Doc — Locked Matrix, Rulings, Rejection Log
Gate-1 adjudications (approved 2026-07-12, incl. the both-variants amendment), updated through Gate-3.
Full evidence: `docs/claim-ledger.md`. Method: 4 research agents + adversarial review; orchestrator
personally re-verified every load-bearing claim against primary sources (executed commands, tags, blobs).

## Locked Version Matrix

Common anchor, every image: **linux/amd64 · py3.12 · CUDA 13.0.2 lineage · torch==2.11.0 (default
PyPI wheel IS cu13) · sm90 first-class · NVIDIA driver in an enumerated branch 535/550/565/570/575.**
The three vLLM anchors in play (0.25.0 serving / 0.24.0 omni / 0.20.2 verl) **all pin torch==2.11.0**
— the torch/CUDA layer unifies; only the vLLM package version splits images.

| Image | Base | Key pins (full set: `constraints.txt` + `resolved-lock.txt`) |
|---|---|---|
| variant-a/serve-main | nvidia/cuda:13.0.2-devel-ubuntu24.04 | vllm==0.25.0 · torch==2.11.0 · flashinfer 0.6.13(+cubin) · transformers==5.13.1 · quack-kernels==0.5.0 (load-bearing: 0.5.1+ needs cutlass-dsl 4.6.0 vs vLLM's 4.5.2) · torchcodec==0.14.0 · opencv==4.13.0.92 (4.x, parity-conservative) — **no ray / no modelopt / no flash-attn pip pkg** |
| variant-a/serve-omni | vllm/vllm-openai:v0.24.0 (blob-verified CUDA 13.0.2) | vllm-omni @ b9e9d236c3f78afd405119a5b686ebebeeb53984 (v0.24.1: RMSNorm numerics fix absent from PyPI 0.24.0); frozen core guarded at build (vllm 0.24.0 / torch 2.11.0 / flashinfer 0.6.12) |
| variant-b/serve-unified | vllm/vllm-openai:v0.24.0 | same stack as serve-omni; hosts all three models, one container per model |
| quant-offline | nvidia/cuda:13.0.2-devel-ubuntu24.04 | nvidia-modelopt[hf]==0.45.0 · torch==2.11.0 · transformers==5.8.1 (modelopt caps <5.10; 5.9.x avoided) · datasets==5.0.0 · accelerate==1.14.0 |

## Rulings

**D1 — Topology.** One serving image for all three models at the newest pins is impossible: vllm-omni
has NO Python-level vLLM pin — it couples to `vllm/vllm-openai:v0.24.0` via Docker base only, and even
its v0.25.0rc1 still targets 0.24.0 (COR-8). Ship BOTH variants (human amendment): A = newest engine
for gen+embedding (0.25.0) + isolated omni image; B = single 0.24.0+omni image (MTP floor v0.17.0
permits it) at the cost of 0.25.0's fixes and its torchcodec/PyNvVideoCodec video stack. **verl/training
excluded**: verl v0.8.0 ships its own maintained image (`verlai/verl:vllm020.latest`) on the same
torch 2.11.0/cu130 anchor — pointer, not duplication.

**D2 — Quantization.** Qwen3.6 → **NVIDIA pre-quant `nvidia/Qwen3.6-35B-A3B-NVFP4` as-is**:
MIXED_PRECISION (161 W4A16-NVFP4 experts + 130 FP8 attention + FP8 KV, producer modelopt 0.44.0),
sm90-servable via pinned Marlin weight-only (`get_min_capability()==75`, code-verified), BF16 MTP head
intact (19 keys, zero scales). **Promotion gate EXECUTED-PASS 2026-07-12**: full 30,841-tensor
F8_E4M3 scale scan clean (`handoff/results/`). No self-quantization of Qwen3.6 (modelopt 0.45's
w4a16_nvfp4 export is "vLLM deployment support in progress" per its changelog). Omni-30B → **BF16
default**; FP8-W8A8 thinker-only recipe ships as an offline, eval-gated script (off vendor-validated
path — only NVFP4 W4A4 is vendor-documented, and that path is Blackwell-oriented). VL-Embedding-8B →
**no quantization** (dense 8B fits; tight cosine gate; #33204 already fragile).

**D3 — Speculative decoding: MTP over DFlash; OFF by default.** DFlash rejected on sm90: open H100
correctness bug #39928 (NaN/`!!!!`, "not MTP or no-spec" — quote verified), zero sm90 performance
numbers anywhere, vendor path via unmerged PR #40898, hybrid-drafter support validated only on
Qwen3.5. MTP ships as an opt-in profile with **`--no-enable-prefix-caching` mandatory** — prefix
caching + MTP is an open correctness collapse on this exact model (#43559 tool-calling 90%→50%,
#47194, #47087; 5 fix PRs unmerged at v0.25.0). NVFP4-target default adds `"moe_backend":"triton"`
(NVIDIA serve recipe; matches the benchmark grid exactly). Enablement is gated on the composed
benchmark grid + 7 canary repros (`handoff/`). Seed claim "DFlash collapses on FP8 targets" was
**REFUTED** (misattributed: prefix caching #47930 + a fixed KV-layout bug #45343→#42095).

**D4 — Levers (3 adopted).** (1) torch.compile + CUDA graphs at defaults; never `--enforce-eager` in
prod profiles. (2) Multimodal processor cache ON (default 4 GiB LRU host-RAM; content-hash keyed →
zero parity risk). (3) **FP8 KV cache (fp8_e4m3) on Qwen3.6 only** — targets the only length-scaling
KV (10/40 full-attention layers at 262K; ~2.5 GB/full-context sequence, derived not measured). BF16
profile adds `--calculate-kv-scales`; the NVFP4 checkpoint ships KV scales. Backend support verified
at the tag on BOTH paths: FLASH_ATTN (FA3 + family(90)) and FlashInfer (`fp8_e4m3` in
`supported_kv_cache_dtypes`, no capability gate) — so the lever survives even if FA3 is inactive on
the cu13 wheel (R4 mitigation).

**D5 — Parity.** Preprocessing is part of the model: profiles pin `--mm-processor-kwargs`
(embedding: `{"fps":2.0,"max_frames":64}`; pooler `LAST`+normalize) and leave EVS/`video_pruning_rate`
unset (EVS reaches the pooling runner via class inheritance and would alter tokens). Known upstream
divergence #33204 documented, not worked around. Parity vs the production precomputed index is a hard
gate executed at prod integration (`handoff/parity_embedding.py`, `--save-reference` on the CURRENT
image first).

## Rejection log (Agent D + Gate-1; every candidate not adopted)

| Candidate | Why rejected |
|---|---|
| TRTLLM-GEN attention | Code-gated `is_device_capability_family(100)` = Blackwell-only; not on sm90 (verified at tag) |
| Force FlashInfer backend on sm90 | vLLM's own non-MLA priority is FLASH_ATTN-first on Hopper; no sm90 evidence for our shapes |
| EVS / video_pruning_rate | Reaches the pooling runner; prunes tokens → violates the feature-parity gate; also NVIDIA-proprietary header (R9) |
| Prefix-caching changes as a lever | Default ON; near-zero benefit for unique-media embedding calls; MTP interaction is D3's ruling |
| Chunked-prefill / max_num_batched_tokens tuning | No GPU-verified win for our shapes; chunk-boundary FP nondeterminism vs the cosine bar unvalidated |
| `vllm[video]` extra | Empty at v0.25.0 (`'video': []`); video deps are base requirements — no-op |
| `vllm[audio]` extra on the embedding model | No audio tower; relevant only to omni serving |
| ffmpeg/codec pinning as a *perf* lever | Reproducibility/parity concern, handled in images; frames arrive client-extracted |
| fps/max_frames tuning | Parity PINS, not knobs (#33204-class drift) |
| FP8 KV on embedding/omni calls | Prefill-dominated, transient tiny KV → no benefit, adds quality risk |
| Vision-encoder FP8 (`mm_encoder_fp8`) | Direct parity risk on the embedding index; encoders stay BF16 per every documented recipe |
| `mm_ipc_gpu_memory_gb` | Default 0; niche flow control, carves KV VRAM |
| FULL cudagraphs for the pooling model | Prefill-only; decode-graph benefit ≈ 0, capture risk ≠ 0 |
| fastsafetensors / streamers | Load-time levers, not steady-state serving wins |
| async_scheduling / kv_sharing_fast_prefill / disable_hybrid_kv_cache_manager | Niche scheduler knobs, no model-relevant evidence; hybrid manager auto-tunes for Qwen3.6 |
| NGC monthly vLLM container as base | 26.06 ships vllm 0.22.1 + torch 2.13 + CUDA 13.3 — different anchor, lags upstream (COR-7) |
| DFlash + DSpark drafters | See D3; DSpark shares the sm90 kernel gap and the prefix-cache collapse (#47930) |
| verl-in-image | D1: own upstream image on the same anchor |
| Self-quantizing Qwen3.6 | D2: official pre-quant exists, scan-certified; self-quant export path marked "in progress" upstream |

## Corrections adopted (brief/seed errors found by the fleet, verified by the orchestrator)

COR-1 Qwen3.6-35B-A3B is natively **multimodal** (3 agents independently) · COR-2 MTP arch floor is
**v0.17.0** · COR-3 S9's FP8-acceptance-collapse **refuted** · COR-4 DFlash card peaks **3.61x@B200**,
no sm90 numbers · COR-5 verl `torch==2.9.1` is sglang-extra-only · COR-6 checkpoint auto-selects
**`modelopt_mixed`** · COR-7 NGC diverges from the anchor · COR-8 vllm-omni couples via **Docker base
only** (drives D1).

---

# Gate-3 additions (RULED 2026-07-14 — full evidence: `docs/gate3-adjudication.md`)

## Rulings D6–D9

**D6 — SGLang standalone (variant-c).** `sglang==0.5.15.post1` serves ALL THREE hosted models plus
the NVFP4 artifact at code level on sm90 (arch registrations, `modelopt_mixed` + W4A16_NVFP4 Marlin
path with `min_capability` 80, in-model pooler LAST+normalize, omni thinker with audio encoder,
explicit BF16-MTP-head handling for this checkpoint family — gate3 §1a, all orchestrator-read at
the tag). Floor forced by the NVFP4 artifact (W4A16_NVFP4 handling lands at 0.5.15). Anchor held
exactly (torch 2.11.0 / CUDA 13.0.2 / py3.12). Five profiles in `configs/sglang/`; MTP profile
opt-in + explicit `--speculative-draft-model-path` + canary-gated; embedding profile carries the
chat-template caveat (resolve at GPU gate).

**D7 — Coexistence.** Single-Python-environment coexistence of even TWO of {vllm, sglang,
tensorrt_llm} is **impossible at every version level** — 8 executed uv CONFLICT probes including
both best-ever pairs, plus exhaustive pin walks; structural walls: llguidance (disjoint
vllm-vs-sglang across all releases ever), torch (no trtllm admits 2.11.0 final; ceiling 2.11.0a0
through rc20), outlines→outlines_core, mistral-common, fastapi. **HUMAN RULING: no variant-e.**
The split-venv fallback was presented and declined; the two standalones are the answer.

**D8 — TensorRT-LLM standalone (variant-d): option (a).** TRT-LLM 1.2.1 (newest stable) hosts NONE
of the three models (Qwen3.6 arch absent; NVFP4 artifact `W4A16_NVFP4` not in the QuantAlgo enum +
NVFP4 GEMM sm100-gated with no Hopper fallback; no `/v1/embeddings` anywhere in trtllm-serve; omni
arch absent — gate3 §1b). Ship as **engine infrastructure**: pinned image, all three models as
LOUD exclusions (build log + manifest + README), BLOCKED runbook (`handoff/trtllm_smoke.md`),
**no model profiles**. Backend: PyTorch backend + trtllm-serve (classic `trtllm-build` engines
need the target GPU at build — impossible this cycle). Provisioning: pip into the
`nvidia/cuda:13.0.2` base with two extra build-time indexes (download.pytorch.org/whl/cu130 for
`torch==2.9.1+cu130`; pypi.nvidia.com for the 2.51 GB wheel). Re-activation trigger: R19.

**D9 — Per-engine anchor deviations.** Accepted in principle, per-image, evidence-bound:
variant-c deviates on nothing; variant-d deviates on torch (2.9.1+cu130 — TRT-LLM's own window
`>=2.9.1,<=2.10.0a0`), transformers (4.57.3), numpy (<2). linux/amd64 + sm90 stay mandatory
everywhere. Driver statements are re-derived per image (CUDA 13.x native floor ≥580 per NVIDIA's
compat table; the nvidia/cuda:13.0.x-based images keep the R10 enumerated-branch behavior).

## Corrections COR-9…COR-13 (Gate-3 fleet + orchestrator, all verified)

COR-9 tensorrt-llm's pypi.org sdist is a **functional 10,133-byte wheel-stub** (build-backend
`wheel_stub.buildapi` → pypi.nvidia.com) with complete authoritative metadata — not "~0-byte" ·
COR-10 the sglang kernel package was **renamed**: stale `sgl-kernel` (0.3.21) → real dependency
**`sglang-kernel`==0.4.4** (615 MB cu13 wheel on pypi.org) · COR-11 **trust the gates, not the
headlines**: TRT-LLM's "FP4 … on Blackwell and Hopper" doc line contradicts its own hardware
matrix (Hopper NVFP4 = ".") and `MIN_SM_VERSION=100` gates; mirror-image, sglang's docs label
NVFP4 "(Blackwell)" while its code runs our W4A16 artifact on sm90 via Marlin · COR-12 the trtllm
rc frontier is **split**: pypi.org stops at 1.3.0rc15; pypi.nvidia.com/NGC/GitHub carry rc16–rc20
(~6 weeks newer, materially different pins) · COR-13 Gate-1 D4-1's "torch.compile + CUDA graphs
at defaults" coupling is **vLLM-specific** — both new engines default to CUDA-graphs-ON /
torch.compile-OFF (experimental/prototype); lever tables track them separately.

## Gate-3 rejection log

| Candidate | Why rejected |
|---|---|
| Single-env coexistence image (the original variant-e target) | Proven unresolvable at every version level (D7); "coexistence" naming reserved and retired |
| Split-venv multi-engine image (fallback) | Presented, not promoted; human declined — demonstrator value only vs three torch stacks of weight |
| TRT-LLM 1.3.0rc lane (rc20) for Qwen3.6 text | Prerelease (against repo conventions), nvidia-index-only, text-only CausalLM, and R16 unresolved (our checkpoint's ConditionalGeneration arch name has no registered route) |
| NGC tensorrt-llm/release:1.2.1 as variant-d base | 19.5 GiB compressed; custom NGC torch (2.10.0a0+b4e4ee8) outside our provenance model; documented as the fallback base only |
| sglang official image (lmsysorg/sglang:v0.5.15.post1-cu130) as variant-c base | CUDA 13.0.1 (one patch below anchor), 13.4 GB, BUILD_TYPE=all surface; pip-into-13.0.2-base is fully pypi-resolvable and matches house audit style |
| `--reasoning-parser qwen3` in the sglang default profile | Cookbook-suggested but output-shape-affecting; omitted pending parity adjudication |
| fp8 KV on the sglang BF16 Qwen3.6 profile | No `--calculate-kv-scales` analogue exists; scales default 1.0 with documented accuracy warning (R21) |
