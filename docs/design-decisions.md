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
