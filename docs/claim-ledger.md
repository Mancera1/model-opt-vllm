# Complete Claim Ledger — Inference Container Modernization

**Standard:** every claim that influenced a pin, design decision, or go/no-go carries a row. `VERIFIED` = primary source read at the pinned tag or command executed this program (2026-07-12). Pedigree order: orchestrator-executed > orchestrator-re-verified agent claim > agent-verified > document-sourced. Agent rows below show the orchestrator disposition; **no load-bearing row ships UNVERIFIED unacknowledged** (see §6).

## 1. Orchestrator-executed verifications (my own instruments)

| id | claim | evidence (executed) |
|----|-------|---------------------|
| O-01 | PyPI latest set 2026-07-12: vllm 0.25.0 · vllm-omni 0.24.0 · verl 0.8.0 · nvidia-modelopt 0.45.0 · flashinfer-python 0.6.14 · torch 2.13.0 · transformers 5.13.1 | pypi.org/pypi/<pkg>/json (executed curl, upload dates captured) |
| O-02 | vllm v0.25.0 requirements: torch==2.11.0, torchvision==0.26.0, torchaudio==2.11.0, torchcodec>=0.14, PyNvVideoCodec==2.0.4, flashinfer 0.6.13(+cubin), transformers>=5.5.3, cutlass-dsl[cu13]==4.5.2, quack>=0.3.3, humming[cu13]==0.1.10 | raw cuda.txt+common.txt at tag (executed) |
| O-03 | vllm v0.20.2 requirements: torch==2.11.0, flashinfer 0.6.8.post1(+cubin) | raw cuda.txt at tag (executed; withheld canary matched Agent A) |
| O-04 | vllm v0.24.0 requirements: torch==2.11.0, flashinfer 0.6.12(+cubin); NO torchcodec/PyNvVideoCodec; opencv>=4.13.0 only; transformers>=5.5.3 | raw cuda.txt+common.txt at tag (executed) |
| O-05 | verl v0.8.0 Dockerfile.stable.vllm: CUDA 13.0.2/ubuntu24.04, torch 2.11.0@whl/cu130, vllm 0.20.2, transformers 5.3.0; setup.py torch==2.9.1 is SGLANG_REQUIRES-only, VLLM_REQUIRES ceiling <=0.12.0 stale | raw Dockerfile+setup.py at tag (executed) |
| O-06 | vllm-omni v0.24.1 AND v0.25.0rc1 Dockerfile.cuda: ARG BASE_IMAGE=vllm/vllm-openai:v0.24.0; requirements carry NO vllm/torch pin; transformers >= 5.5.3 (common.txt L12) | raw files at both tags (executed) |
| O-07 | GH latest releases: vllm v0.25.0 (2026-07-11) · verl v0.8.0 (2026-06-01) · vllm-omni v0.24.1 (2026-07-10) · Model-Optimizer 0.45.0 (2026-07-06) | api.github.com releases/latest (executed) |
| O-08 | All five HF repos exist: z-lab drafter, nvidia NVFP4, three Qwen serving models (HTTP 200) | huggingface.co/api/models (executed) |
| O-09 | modelopt 0.45.0 CHANGELOG (701 lines) contains NO fix for the 0.44 FP8-per-block-weight-scale NaN export bug; line 93 reduce_amax fix is a different bug; line ~11: w4a16_nvfp4 'vLLM deployment support is in progress' | raw CHANGELOG.rst at tag + grep (executed) |
| O-10 | nvidia/Qwen3.6-35B-A3B-NVFP4 hf_quant_config.json: producer modelopt 0.44.0; MIXED_PRECISION; 291 quantized_layers = 130 FP8 + 161 W4A16_NVFP4; kv FP8; exclude mtp* | raw hf_quant_config.json (executed, parsed) |
| O-11 | safetensors index: 124,468 keys; exactly 19 mtp.* keys with ZERO scale tensors (BF16 MTP head); 333 model.visual.* tensors | resolve/model.safetensors.index.json (executed, counted) |
| O-12 | z-lab draft config: DFlashDraftModel, 6 layers (5 sliding+1 full), hidden 2048, vocab 248320, block_size 16, target_layer_ids [1,6,11,16,22,27,32,37], num_target_layers 40; cardData base_model Qwen/Qwen3.6-35B-A3B | raw config.json + HF api (executed) |
| O-13 | Qwen/Qwen3.6-35B-A3B config: MULTIMODAL (vision_config, image/video token ids); text_config 40 layers = 30 linear_attention + 10 full_attention, kv_heads 2, mtp_num_hidden_layers 1, 262144 ctx, 256 experts | raw config.json (executed, counted) |
| O-14 | z-lab org: 54 repos, 25 DFlash drafters, ZERO omni matches | HF api author=z-lab (executed) |
| O-15 | NVIDIA README: Supported HW 'NVIDIA Hopper, NVIDIA Blackwell'; 'Quantized with nvidia-modelopt v0.44.0'; Test Hardware GB300; MTP serve config with moe_backend triton | raw README.md (executed) |
| O-16 | Nine spec-decode issues verified OPEN with cited titles (#39928 #43559 #47930 #40756 #47087 #47194 #35288 #47277 #39680); PR #40898 open/unmerged; PR #47914 merged 2026-07-08 'Validated on Qwen3.5-35B-A3B'; benchislett + timothysu quotes character-matched | api.github.com issues/pulls + comments (executed) |
| O-17 | Docker Hub config blobs: vllm/vllm-openai v0.25.0 AND v0.24.0 plain tags = CUDA_VERSION 13.0.2, arch list includes 9.0, NVIDIA_REQUIRE_CUDA enumerates driver branches 535/550/565/570/575 | registry-1.docker.io manifest+config blob (executed via curl) |
| O-18 | vllm v0.25.0 modelopt.py: ModelOptFp8 min_capability 89; modelopt_fp4 75; MIXED 75 with comment 'NVFP4 routed experts run via Marlin W4A16 (SM75+) … Validated on Tesla T4/A100'; W4A16 placeholder input_scale 'discarded … never used'; method name modelopt_mixed | raw modelopt.py at tag, lines read (executed) |
| O-19 | registry.py v0.25.0 registers Qwen3OmniMoe(546)/Qwen3VL(555)/Qwen3_5Moe(561)/Qwen3_5MoeMTP(634); floors: Qwen3_5MoeMTP 0@v0.16.0→1@v0.17.0; dflash 0@v0.19.1→12@v0.20.0 in speculative.py | raw registry.py+speculative.py at 4 tags (executed greps) |
| O-20 | v0.25.0 config defaults: mm_processor_cache_gb=4 (LRU); video_pruning_rate=None; enforce_eager=False; enable_prefix_caching=True; calculate_kv_scales=False; mamba_cache_dtype separate from cache_dtype | raw config/*.py at tag, lines read (executed) |
| O-21 | v0.25.0 backend facts: Hopper non-MLA priority FLASH_ATTN→FLASHINFER (major==10 flips); TRTLLM-GEN gated is_device_capability_family(100); FLASH_ATTN fp8/fp8_e4m3 KV iff FA-version==3 AND family(90) | raw cuda.py + flash_attn.py at tag (executed) |
| O-22 | evs.py exists at v0.25.0 with NVIDIA-proprietary copyright header; wired into qwen3_vl.py (L72 import; postprocess hooks) | raw evs.py + qwen3_vl.py at tag (executed) |
| O-23 | setup.py extras at v0.25.0: video=[] (empty), audio=[av,scipy,soundfile,…]; CMakeLists includes vllm_flash_attn external project (L1402) | raw setup.py + CMakeLists.txt (executed) |
| O-24 | torch 2.11.0 default PyPI wheel is cu13: requires_dist cuda-toolkit==13.0.2 + nvidia-{cudnn,nccl,nvshmem,cusparselt}-cu13; cp312 manylinux_2_28 wheel present | pypi torch/2.11.0/json (executed) |
| O-25 | vllm 0.25.0 PyPI: cp38-abi3 manylinux_2_28 wheels (x86_64+aarch64)+sdist; requires_dist contains torch==2.11.0, flashinfer-python==0.6.13, flashinfer-cubin==0.6.13, transformers>=5.5.3, opencv>=4.13.0, torchcodec>=0.14; NO ray in requirements files or pyproject | pypi vllm/0.25.0/json + 4-file grep (executed) |
| O-26 | nvidia-modelopt 0.45.0: requires-python <3.15,>=3.10; torch>=2.8; [hf] extra caps transformers<5.10,>=4.56 | pypi nvidia-modelopt/0.45.0/json (executed) |
| O-27 | FULL NaN-scale scan EXECUTED-PASS: 30,841/30,841 F8_E4M3 weight_scale tensors (2,052.9 MB) — zero 0x7F/0xFF/negative bytes; class size matched Agent B header prediction exactly | orchestrator_scale_scan.py run 2026-07-12 → handoff/results/scale_scan_result.2026-07-12.json (EXECUTED) |
| O-28 | Independent partial-scan replication on tensors DISJOINT from Agent B sample (layer-20 experts, F8_E4M3): clean; also exposed the F32-vs-F8_E4M3 tensor-class instrument trap now baked into scan tooling | ranged safetensors reads (executed) |
| O-29 | uv 0.11.28 cross-platform resolution (x86_64-manylinux_2_28, py3.12): all four image graphs RESOLVE; every constraints pin exact in lock (zero drift); cu13 lineage present; vllm-omni@commit co-resolves holding vllm 0.24.0/torch 2.11.0/flashinfer 0.6.12 | uv pip compile ×3 + pin-diff script (EXECUTED) → */resolved-lock.txt |
| O-30 | quack-kernels cutlass boundary: 0.5.0 needs cutlass-dsl>=4.5.2; 0.5.1 ==4.6.0.dev0; 0.6.1 ==4.6.0 → 0.5.0 is the only newest-satisfying pin compatible with vllm's ==4.5.2 | pypi quack-kernels/{0.5.0,0.5.1,0.6.1}/json requires_dist (executed; triangulates author+reviewer) |
| O-31 | opencv-python-headless 4.13.0.92 exists on PyPI (newest 4.x; 5.0.0.93 is the unconstrained-resolver pick) | pypi (executed) |
| O-32 | --runner and --pooler-config are real vLLM v0.25.0 flags; model.py:597 recommends '--runner pooling' for embedding models | raw arg_utils.py + model.py at tag (executed) |
| O-33 | GATE-3: vllm-omni@commit main() requires --omni in argv; OmniServeCommand name='serve'; console script vllm-omni registered; official docs serve OUR EXACT model: 'vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni'; default deploy yaml qwen3_omni_moe.yaml auto-loads | raw cli/main.py, cli/serve.py, docs qwen3_omni.md, pyproject at commit (executed) |
| O-34 | GATE-3: vLLM v0.24.0 has ALL profile flags: --runner(796) --pooler-config(852) --kv-cache-dtype(1153) --calculate-kv-scales(1168) --mm-processor-kwargs(1245) --moe-backend(1482); SpeculativeConfig.moe_backend key exists at 0.24.0(L107) and 0.25.0(L112) → variant-B profile compatibility REAL | raw arg_utils.py + speculative.py at v0.24.0 (executed) |
| O-35 | GATE-3: FlashInferBackend.supported_kv_cache_dtypes includes fp8 + fp8_e4m3 (+e5m2) with NO capability gate (only nvfp4-KV is family(100)-gated) → FP8-KV survives even if FA3 inactive on cu13 (R4 mitigation verified) | raw flashinfer.py L332-338 + backend.py L168-173 at tag (executed) |
| O-36 | GATE-3: NVIDIA_REQUIRE_CUDA in cu13 images enumerates driver BRANCHES 535/550/565/570/575 (each <next+1): drivers like 545.x/560.x FAIL — 'driver >=535' is too generous; documented in migration notes | Docker config blob REQUIRE_CUDA string (captured, executed) |
| O-37 | vllm-omni v0.24.1 release delta over 0.24.0: Qwen-Image perf-regression fix + RMSNorm default-op priority restore (numerics-relevant) → pin the GH commit, PyPI 0.24.0 lacks it | api.github.com releases/tags/v0.24.1 (executed) |

## 2. Agent A — Compatibility matrix

| id | claim | conf | status | orchestrator disposition |
|----|-------|------|--------|--------------------------|
| A-01 | verl v0.8.0 ACTUALLY runs vllm==0.20.2 + torch==2.11.0(cu130) + CUDA 13.0.2 + py3.12 + transformers==5.3.0 in its own stable image and CI; the release-notes '0.20.2' is correct, setup.py metadata is stale. | high | VERIFIED | RE-VERIFIED (O-05) |
| A-02 | vllm/vllm-openai:v0.25.0 PLAIN Docker tag is CUDA 13.0.2 (sm90 in arch list); '-cu129' is the CUDA 12.9 alternate. PyPI `pip install vllm==0.25.0` likewise yields a cu13 stack. | high | VERIFIED | RE-VERIFIED (O-17) |
| A-03 | torch==2.11.0's default PyPI wheel is CUDA 13; therefore vllm 0.25.0 (hard dep torch==2.11.0) installs a cu13 torch with no special index. cp312 wheel exists. | high | VERIFIED | RE-VERIFIED (O-24) |
| A-04 | vllm-omni v0.24.1 AND v0.25.0rc1 both target vllm 0.24.0 via docker base; NO vllm-omni release targets vllm 0.25.0. Co-install of omni with vllm 0.25.0 is untested upstream. | high | VERIFIED | RE-VERIFIED (O-06) |
| A-05 | vllm-omni v0.24.1 declares NO vllm/torch version constraint in pyproject/requirements; only version-relevant Python pin is transformers>=5.5.3; vllm coupling is docker-base-only. | high | VERIFIED | RE-VERIFIED (O-06) |
| A-06 | torch==2.11.0 is the SINGLE unifying anchor: vllm 0.25.0, vllm 0.24.0 (omni base), and vllm 0.20.2 (verl) all pin torch 2.11.0/cu13 on py3.12. | high | VERIFIED | RE-VERIFIED (O-02/03/04) |
| A-07 | flashinfer pins track vllm version: 0.6.13 (vllm 0.25.0), 0.6.12 (vllm 0.24.0), 0.6.8.post1 (vllm 0.20.2). All are pure-python py3-none-any wheels (cp312-trivial). | high | VERIFIED | RE-VERIFIED (O-02/03/04) |
| A-08 | nvidia-modelopt 0.45.0 CORE requires only torch>=2.8 and python <3.15,>=3.10 (compatible with torch 2.11.0/py3.12); its [hf] extra caps transformers<5.10,>=4.56. | high | VERIFIED | RE-VERIFIED (O-26) |
| A-09 | vllm 0.25.0 does NOT hard-depend on ray (absent from common/cuda/build/test requirements). ray is a free axis: verl needs >=2.41.0, vllm imposes no ceiling; ray 2.52.1 has cp312 wheels, latest 2.56.0. | high | VERIFIED | RE-VERIFIED (O-25) |
| A-10 | transformers diverges by role: vllm 0.25.0 & vllm-omni require >=5.5.3; verl v0.8.0 image pins ==5.3.0; vllm 0.20.2 itself excludes !=5.3.* yet verl force-installs 5.3.0 after vllm. | high | VERIFIED | RE-VERIFIED (O-02/04/05) |
| A-11 | verl v0.8.0 setup.py torch==2.9.1 is in SGLANG_REQUIRES only; VLLM_REQUIRES contains NO torch pin (only tensordict + 'vllm>=0.8.5,<=0.12.0'). The brief conflated the two. | high | VERIFIED | RE-VERIFIED (O-05) |
| A-12 | NGC nvcr.io/nvidia/vllm is anonymously listable; latest tag 26.06-py3. 26.06 bundles vllm 0.22.1 + torch 2.13 + CUDA 13.3.0 + py3.12 + flashinfer 0.6.12 + transformers 5.6.0 (lags upstream vllm; torch/CUDA newer than the 2.11.0/13.0.2 anchor). | medium | VERIFIED | ACCEPTED doc-sourced; moot (NGC rejected) |
| A-13 | vllm 0.25.0 PyPI ships cp38-abi3 manylinux_2_28 wheels (x86_64 + aarch64) which are cp312-compatible via the CPython stable ABI, plus an sdist. | high | VERIFIED | RE-VERIFIED (O-25) |
| A-14 | vllm/vllm-openai:v0.24.0 (omni base, plain tag) is also CUDA 13.0.2 with sm90 in the arch list. | high | VERIFIED | RE-VERIFIED (O-17) |
| A-15 | verl v0.8.0 builds its stable vllm image FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 and installs torch from download.pytorch.org/whl/cu130 — the raw-CUDA-base + pip pattern is verl's own proven route to a cu130 stack. | high | VERIFIED | RE-VERIFIED (O-05) |
| A-16 | sm90/Hopper (arch 9.0) is a first-class build target for vllm 0.25.0 in both cu129 and cu130 variants. | high | VERIFIED | ACCEPTED (arch list corroborated by O-17) |
| A-17 | All non-torch/vllm proposed pins publish cp312-installable wheels: transformers 5.13.1, nvidia-modelopt 0.45.0, vllm-omni 0.24.0, flashinfer-python/cubin are py3-none-any (pure python). | high | VERIFIED | RE-VERIFIED (O-25/O-31 class) |
| A-18 | ray 2.52.1 (baseline) has cp312 manylinux wheels; ray latest is 2.56.0. ray is co-installable across serving (unconstrained) and verl (>=2.41.0). | high | VERIFIED | ACCEPTED (low stakes) |
| A-19 | vllm 0.25.0 adds video-decode deps absent in 0.20.2 (torchcodec>=0.14, PyNvVideoCodec==2.0.4) — relevant to the VL-Embedding video-mode preprocessing/parity gate. | high | VERIFIED | RE-VERIFIED (O-02 vs O-04) |
| A-20 | verl docker/README.md prose + Release-History are STALE vs the v0.8.0 Dockerfile: README says base nvidia/cuda:12.9.1-devel-ubuntu22.04 and last release-history entry is vllm 0.17.0, but the tagged Dockerfile uses nvidia/cuda:13.0.2-devel-ubuntu24.04 + vllm 0.20.2. | high | VERIFIED | ACCEPTED (stale-doc finding) |

## 3. Agent B — Quantization / ModelOpt

| id | claim | conf | status | orchestrator disposition |
|----|-------|------|--------|--------------------------|
| B-01 | nvidia/Qwen3.6-35B-A3B-NVFP4 (MIXED_PRECISION) is servable on sm90/Hopper in vLLM v0.25.0: NVFP4 W4A16 experts run via pinned Marlin (weight-only, BF16 activations), FP8 attention/KV native. | high | VERIFIED | RE-VERIFIED (O-18) |
| B-02 | Native NVFP4 W4A4 (cutlass FP4 GEMM with input quantization) requires Blackwell FP4 tensor cores (sm100+); on sm90 a W4A4-labeled checkpoint falls back to Marlin weight-only, so raw W4A4 is servable-but-degraded, not natively. | high | VERIFIED | RE-VERIFIED (O-18; sm100 constant acknowledged as inferred) |
| B-03 | This checkpoint is W4A4-shaped (input_scale tensors on disk, config_groups input_activations num_bits=4) but its per-layer quant_algo label 'W4A16_NVFP4' makes vLLM serve it weight-only and DISCARD the activation scales. | high | VERIFIED | RE-VERIFIED (O-18/O-10) |
| B-04 | Producer = modelopt 0.44.0; format MIXED_PRECISION = 161 W4A16_NVFP4 (experts/shared/lm_head) + 130 FP8 W8A8 (linear_attn + self_attn) + FP8 KV; zero W4A4 layers. | high | VERIFIED | RE-VERIFIED (O-10) |
| B-05 | The NVFP4 checkpoint ships a BF16 MTP head (19 mtp.* keys, unquantized, in exclude_modules); vLLM 0.25.0 registers Qwen3_5MoeMTP to consume it -> native MTP spec-decode is compatible with this exact checkpoint. | high | VERIFIED | RE-VERIFIED (O-11) |
| B-06 | The 0.44 NaN-per-block-weight-scale export bug is NOT listed as fixed in the 0.45.0 changelog; the 0.45 FP8 fix (reduce_amax) is the unrelated natively-FP8-weights fix. | high | VERIFIED | RE-VERIFIED (O-09) |
| B-07 | The 0.44 NaN bug is real and precisely characterized: float32->FP8 E4M3 cast of per-block weight scales occasionally emits NaN bytes 0x7F/0xFF; a single NaN in any weight_scale collapses output to `!!!!`. It affects the Hopper Marlin FP4 path too (Marlin consumes per-block E4M3 weight_scale). | high | VERIFIED | ACCEPTED (primary docs quote) |
| B-08 | Sampled scale-integrity scan of the NVFP4 checkpoint is CLEAN: 17 F8_E4M3 weight_scale tensors (32.8MB incl. full 31.78MB lm_head.weight_scale) contain zero 0x7F/0xFF (NaN) bytes and zero negatives (max byte 0x7E=448). | high | VERIFIED | RE-EXECUTED + SUPERSEDED by full scan (O-27/O-28) |
| B-09 | FP8 W8A8 (modelopt) requires sm89+ (native FP8 tensor cores); Hopper sm90 satisfies it. FP8 KV cache is native on Hopper. | high | VERIFIED | RE-VERIFIED (O-18) |
| B-10 | vLLM 0.25.0 MIXED_PRECISION dispatch is inclusion-based (_resolve_quant_algo looks up quantized_layers, returns None->BF16 for unlisted layers), so the MoE router mlp.gate / vision tower / mtp being absent from exclude_modules does NOT break loading of this 0.44 checkpoint (NVBug 5718750 does not bite the mixed path). | high | VERIFIED | ACCEPTED (code-read) |
| B-11 | vLLM 0.25.0 registers all three target architectures: Qwen3_5MoeForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration (->thinker), Qwen3VLForConditionalGeneration, plus Qwen3_5MoeMTP. | high | VERIFIED | RE-VERIFIED (O-19) |
| B-12 | Qwen3.6-35B-A3B is natively a MULTIMODAL VL model (vision_config + text_config, image/video tokens), not a text-only generation model; 40 layers (30 linear-attn + 10 full-attn, full_attention_interval=4), 256 experts/8 active, 262K ctx; the NVFP4 checkpoint keeps the 333-tensor vision tower BF16. | high | VERIFIED | RE-VERIFIED (O-13) |
| B-13 | The vLLM-Omni thinker-only recipe exclusions map exactly onto the Instruct checkpoint's real module prefixes. | high | VERIFIED | ACCEPTED (prefix audit) |
| B-14 | vLLM-Omni documents only NVFP4 W4A4 thinker quant as validated; an FP8-only thinker recipe/exclusion list is not separately documented (FP8 checkpoint path merely 'exists'). | medium | VERIFIED | ACCEPTED medium (doc extraction) |
| B-15 | vLLM-Omni warns --enforce-eager degrades W4A4 TPOT ~10x (FP4 needs CUDA graphs); irrelevant to our Omni Call-type-A which is prefill-only with ~no decode. | high | VERIFIED | ACCEPTED (doc quote) |
| B-16 | Qwen3-VL-Embedding-8B is a DENSE model (36-layer text, hidden 4096, no experts), architecture Qwen3VLForConditionalGeneration, with a 351-tensor vision tower and 262K ctx. | high | VERIFIED | ACCEPTED (config read) |
| B-17 | nvidia-modelopt[hf]==0.45.0 pins transformers<5.10,>=4.56 — a co-install conflict with a modern serving transformers (§6: 5.13.1), reinforcing offline-only placement. | high | VERIFIED | RE-VERIFIED (O-26) |
| B-18 | nvidia-modelopt 0.45.0 ships a pure wheel (py3-none-any), requires-python <3.15,>=3.10 (py3.12 OK), torch>=2.8. | high | VERIFIED | RE-VERIFIED (O-26 class) |
| B-19 | NVFP4 accuracy is within ~1pt of BF16 (max gap τ²-Telecom 94.7 vs 95.5), but the numbers are Blackwell-pedigree and do not transfer to the Hopper W4A16 path. | high | VERIFIED | ACKNOWLEDGED UNVERIFIED-transfer (GPU handoff) |
| B-20 | NVFP4 weight VRAM reduction ~3.06x vs BF16 (16->4 bit); this TRANSFERS to Hopper because weights are FP4 on both W4A16 and W4A4 paths (~23.4GB checkpoint). | high | VERIFIED | ACCEPTED (arithmetic on shard sizes) |
| B-21 | vLLM-Omni's Hopper servability of a raw-NVFP4 (W4A4) omni-thinker checkpoint is UNVERIFIED (docs recommend --linear-backend cutlass/--force-cutlass-fp8, which are Blackwell-oriented); a Marlin weight-only fallback is assumed from main-vLLM but vllm-omni source was not read. | medium | UNVERIFIED | ACKNOWLEDGED UNVERIFIED; moot under D2 (omni default BF16) |
| B-22 | modelopt 0.45 changelog is dated 0.45 (2026-07-02); adds first-class W4A16_NVFP4 weight-only (mtq.W4A16_NVFP4_CFG / --qformat w4a16_nvfp4, no calibration forward) but notes 'vLLM deployment support is in progress' for it. | medium | VERIFIED | RE-VERIFIED (O-09) |

## 4. Agent C — Speculative decoding

| id | claim | conf | status | orchestrator disposition |
|----|-------|------|--------|--------------------------|
| C-03 | DFlash (and EAGLE3) has an OPEN sm90/H100 correctness bug: Qwen3.5-9B DFlash gives run-to-run-inconsistent, sometimes NaN->'!!!!' output on short/sub-1-block prompts; MTP and no-spec are NOT affected; enforce-eager mitigates. | high | VERIFIED | RE-VERIFIED (O-16, quote matched) |
| C-11 | All DFlash speedup evidence is Blackwell B200; Hopper-optimized non-causal decode kernels (XQA) are not integrated into vLLM; no sm90 DFlash speedup number exists. | high | VERIFIED | ACCEPTED + corroborated (O-16 tracker/PR states) |
| C-01 | vLLM method '"dflash"' first ships as an accepted SpeculativeConfig method in v0.20.0 (2026-04-27); absent through v0.19.1. | high | VERIFIED | RE-VERIFIED (O-19) |
| C-04 | Native MTP for the Qwen3.6/qwen3_5_moe arch first ships in vLLM v0.17.0 (2026-03-07): Qwen3_5MoeForConditionalGeneration + Qwen3_5MoeMTP registered; absent v0.16.0. | high | VERIFIED | RE-VERIFIED (O-19) |
| C-02 | DFlash for OUR exact Qwen3.6 hybrid-GDN target is NOT release-validated: z-lab card routes vLLM DFlash via UNMERGED PR #40898; merged hybrid-SWA drafter support (#47914 -> v0.25.0) was validated on Qwen3.5-35B-A3B, not 3.6. | high | VERIFIED | RE-VERIFIED (O-16) |
| C-07 | prefix-caching + MTP is an OPEN correctness collapse on Qwen3.6-35B-A3B (~20% acc / tool-calling 90->50% / tool-call leakage / needle-recall failure / garbage loops), rooted in mamba/GDN state-cache reuse, not fixed at v0.25.0. | high | VERIFIED | RE-VERIFIED (O-16, quote matched) |
| C-12 | nvidia/Qwen3.6-35B-A3B-NVFP4 ships the native MTP head (19 mtp.* tensors) -> MTP composable with the NVFP4 target with no separate head; producer is modelopt 0.44.0 (NaN-bug version). | high | VERIFIED | RE-VERIFIED (O-10/O-11) |
| C-08 | z-lab DFlash draft config: DFlashDraftModel, model_type qwen3, 6 layers (5 sliding + 1 full attention), hidden 2048, block_size 16, num_target_layers 40, target_layer_ids [1,6,11,16,22,27,32,37]. | high | VERIFIED | RE-VERIFIED (O-12) |
| C-09 | DFlash draft = 385,906,176 params (0.386B); no embed_tokens/lm_head tensors (shared with target); fc (33.6M) fuses 8 target-layer hidden states. | high | VERIFIED | ACCEPTED (header-computed params; fc cross-check) |
| C-10 | No DFlash draft exists for the omni thinker (Qwen3-Omni-30B-A3B) anywhere on HF. | high | VERIFIED | RE-VERIFIED (O-14) |
| C-14 | Seed S9's 'DFlash acceptance collapses on FP8 targets' is UNCORROBORATED in primary vLLM sources; documented DFlash acceptance collapses are prefix-caching (#47930) and a KV-LAYOUT bug on an FP4 target (#45343), not FP8-intrinsic; z-lab ships FP8-target drafts (GLM-5.1-FP8-DFlash). | medium | REFUTED | REFUTATION ADOPTED (COR-3); sources verified (O-16) |
| C-16 | MTP is not a guaranteed win even ignoring caching: it can be net-slower than the no-spec CUDA-graph baseline and can corrupt at high concurrency. | high | VERIFIED | RE-VERIFIED (O-16: #35288/#47277/#39680 open, titles matched) |
| C-13 | MTP + FP8 target is documented-unstable: MTP with Qwen3.6-27B-FP8 crashes on long sequences (accept->100%, illegal memory access) at v0.19.1. | medium | VERIFIED | RE-VERIFIED open (O-16); version-currency at 0.25.0 acknowledged OPEN |
| C-05 | qwen3_next-family MTP method '"mtp"' consolidated in v0.11.0 (2025-10-02); qwen3-next model support landed v0.10.2 (2025-09-13, PR #24526). | medium | VERIFIED | ACCEPTED medium (v0.11.0 floor; not load-bearing) |
| C-18 | The z-lab draft requires hybrid-SWA drafter support (config layer_types = 5 sliding + 1 full); that support merged only via #47914 (2026-07-08 -> v0.25.0); tracker's 'Hybrid DFlash' box was unchecked at 2026-06-18. | medium | VERIFIED | ACCEPTED medium (PR bodies read) |
| C-06 | Target Qwen3.6-35B-A3B has a native MTP head and is a hybrid-GDN MoE: mtp_num_hidden_layers=1, 40 layers (30 linear_attention + 10 full_attention), 256 experts/8 active, 262K ctx, vocab 248320. | high | VERIFIED | RE-VERIFIED (O-13) |
| C-15 | DFlash carries the same prefix-caching hazard as MTP: draft acceptance collapses with automatic prefix caching enabled. | medium | VERIFIED | RE-VERIFIED (O-16) |
| C-19 | DFlash config.json declares the target only structurally (num_target_layers 40 + target_layer_ids); the explicit 'Qwen/Qwen3.6-35B-A3B' target is in HF card metadata/README, not config.json. | high | VERIFIED | RE-VERIFIED (O-12) |
| C-20 | Qwen3.6-35B-A3B is MULTIMODAL (Qwen3_5MoeForConditionalGeneration with vision_config); spec-decode runs on the language_model path, and #47914 specifically fixes lm_head sharing for *ForConditionalGeneration targets. | high | VERIFIED | RE-VERIFIED (O-13/O-16) |
| C-21 | quant x cache interaction is non-monotonic: fp8 KV cache was reported to RECOVER accuracy vs bf16 KV under the prefix-cache+MTP bug. | low | UNVERIFIED | ACKNOWLEDGED UNVERIFIED low (single reporter; in benchmark grid) |
| C-17 | DFlash landed incrementally: core #36847 (2026-03-30), Qwen3Next target #45319 (2026-06-12 -> v0.23.0), FlashInfer support #43081 (2026-06-22 -> v0.24.0), hybrid-SWA #47914 + layer-mismatch #48113 (2026-07-08/10 -> v0.25.0). | medium | VERIFIED | ACCEPTED medium (date→tag mapping) |

## 5. Agent D — Performance levers

| id | claim | conf | status | orchestrator disposition |
|----|-------|------|--------|--------------------------|
| D-01 | vLLM v0.25.0 pins flashinfer-python==0.6.13 (and flashinfer-cubin==0.6.13); this governs the image over PyPI-latest 0.6.14 because setup.py bakes cuda.txt into the CUDA wheel's Requires-Dist and the official Dockerfile installs -r requirements/cuda.txt. | high | VERIFIED | RE-VERIFIED (O-25/O-02) |
| D-02 | EVS (vllm/multimodal/evs.py) exists at v0.25.0, reduces video tokens to int(total_tokens*(1-q)), is wired into Qwen3VLForConditionalGeneration, reaches the --runner pooling path (which subclasses that class), and is DEFAULT OFF (video_pruning_rate=None). Enabling it alters token count -> violates the feature-parity gate. | high | VERIFIED | RE-VERIFIED (O-22/O-20) |
| D-03 | Multimodal processor cache default state at v0.25.0 is ON, 4 GiB, LRU, host-RAM, duplicated per process (total = gb*(api_server_count+data_parallel_size)); flag --mm-processor-cache-gb, 0 disables. | high | VERIFIED | RE-VERIFIED (O-20) |
| D-04 | enforce_eager defaults False at v0.25.0 so torch.compile + CUDA graphs are ON by default on CUDA/Hopper; --enforce-eager is opt-in and disables them. | high | VERIFIED | RE-VERIFIED (O-20) |
| D-05 | TRTLLM-GEN attention is Blackwell/sm100-only and unavailable on sm90/Hopper at v0.25.0. | high | VERIFIED | RE-VERIFIED (O-21) |
| D-06 | On sm90/Hopper the default attention backend for our non-MLA models is FLASH_ATTN (first priority), with FlashInfer second; on sm100 the order flips (FlashInfer first). | high | VERIFIED | RE-VERIFIED (O-21) |
| D-07 | FLASH_ATTN backend supports fp8_e4m3 KV cache only when FA version==3 AND device is capability-family 90 (Hopper); fp8_e5m2 is NOT supported on this backend. | high | VERIFIED | RE-VERIFIED (O-21) |
| D-08 | Qwen3.6-35B-A3B is a hybrid model: 40 layers, full_attention_interval=4, layer_types = 30 linear_attention + 10 full_attention; 262144 max positions; num_key_value_heads=2; native MTP (mtp_num_hidden_layers). | high | VERIFIED | RE-VERIFIED (O-13) |
| D-09 | --kv-cache-dtype fp8 compresses ONLY Qwen3.6's 10 full-attention layers; the 30 linear/gated-delta layers use a mamba-style state governed by mamba_ssm_dtype/mamba_cache_dtype, not kv_cache_dtype. Those 10 layers are the only length-scaling KV at 262K. | medium | VERIFIED | RE-VERIFIED mechanism (O-20); VRAM fraction stays derived |
| D-10 | Prefix caching is ON by default at v0.25.0 (enable_prefix_caching=True). | high | VERIFIED | RE-VERIFIED (O-20) |
| D-11 | Chunked prefill is ON by default; the SchedulerConfig field default max_num_batched_tokens=2048 (DEFAULT_MAX_NUM_BATCHED_TOKENS) is labelled 'mainly for convenience when testing', not the production-resolved value. | high | VERIFIED | ACCEPTED (field default read) |
| D-12 | vllm[video] extra is empty at v0.25.0; video-IO deps are base requirements. | high | VERIFIED | RE-VERIFIED (O-23) |
| D-13 | vllm[audio] extra = [av, scipy, soundfile, soxr, mistral_common[audio]] at v0.25.0; irrelevant to VL-Embedding (no audio tower), relevant to omni only. | high | VERIFIED | RE-VERIFIED (O-23) |
| D-14 | Qwen3_5MoeForConditionalGeneration (Qwen3.6) is registered in vLLM v0.25.0 with native MTP support (Qwen3_5MoeMTP), so it is loadable and the FP8-KV recommendation applies. | high | VERIFIED | RE-VERIFIED (O-19) |
| D-15 | Qwen/Qwen3.6-35B-A3B config declares MULTIMODAL capability: arch Qwen3_5MoeForConditionalGeneration with top-level vision_config, image_token_id, video_token_id - contradicting the brief's 'MoE generation' text-only framing. | high | VERIFIED | RE-VERIFIED (O-13) |
| D-16 | vllm/multimodal/evs.py carries an NVIDIA-proprietary copyright header (not plain Apache-2.0), a licensing note for anyone repackaging. | high | VERIFIED | RE-VERIFIED (O-22) |
| D-17 | FlashAttention (FA3) is compiled INTO the vllm wheel via CMake external project (cmake/external_projects/vllm_flash_attn.cmake), not via the pip vllm-flash-attn package that setup.py skips on non-cu12; so FA can be present on cu13 despite the skipped pip dep. | medium | UNVERIFIED | ACKNOWLEDGED UNVERIFIED; mitigated by O-35 (FlashInfer fp8-KV fallback) + Gate-2 wheel inspection pending |
| D-18 | modelopt 0.45 kv/fp8_cast preset produces calibrated KV scales that vLLM loads when calculate_kv_scales=False (its default = 'scales loaded from the model'). | medium | UNVERIFIED | ACKNOWLEDGED UNVERIFIED; moot for the pre-quant path (checkpoint ships scales) |

## 6. UNVERIFIED / residual acknowledgment (Gate-3 audit)

Every non-VERIFIED or transfer-limited row and its disposition:

- **B-19** [VERIFIED] — ACKNOWLEDGED UNVERIFIED-transfer (GPU handoff)
- **B-21** [UNVERIFIED] — ACKNOWLEDGED UNVERIFIED; moot under D2 (omni default BF16)
- **C-14** [REFUTED] — REFUTATION ADOPTED (COR-3); sources verified (O-16)
- **C-21** [UNVERIFIED] — ACKNOWLEDGED UNVERIFIED low (single reporter; in benchmark grid)
- **D-17** [UNVERIFIED] — ACKNOWLEDGED UNVERIFIED; mitigated by O-35 (FlashInfer fp8-KV fallback) + Gate-2 wheel inspection pending
- **D-18** [UNVERIFIED] — ACKNOWLEDGED UNVERIFIED; moot for the pre-quant path (checkpoint ships scales)

Handoff-author documented open items (in-script, not fabricated): embeddings request schema for frames-as-video; `vllm:spec_decode_*` metric names at 0.25.0; Qwen3-Omni AutoProcessor audio-calibration API; NVFP4-MTP moe_backend=triton on-hardware confirmation; vLLM-Omni `input_audio` content-type acceptance. All BLOCKED on a live sm90 endpoint; none load-bearing for the build.

## 7. Refutations & corrections adopted (highlights)

- **C-14 REFUTED seed S9**: "DFlash acceptance collapses on FP8 targets" — real causes are prefix caching (#47930) and a fixed KV-layout bug (#45343→#42095); z-lab ships FP8-target drafters.
- **COR-2**: MTP floor for this arch is v0.17.0 (not "~0.19.x").
- **COR-1 (×3 agents)**: Qwen3.6-35B-A3B is natively multimodal (vision tower ships BF16).
- **COR-4**: DFlash card peaks at 3.61x@conc1 on B200 (not "5–6x"); zero sm90 numbers exist.
- **COR-5**: verl setup.py `torch==2.9.1` is sglang-extra-only; the vllm path runs torch 2.11.0/cu130.
- Full correction set (COR-1…COR-8) in docs/design-decisions.md.

## 8. Gate-3 cycle — orchestrator-executed verifications (2026-07-14)

Method + full citation set (C1–C86) live in `docs/gate3-adjudication.md`; the rows below are the
ledger entries for what was EXECUTED this cycle (same pedigree standard as §1). Fleet: 5
schema-forced research agents (all canaries matched; 0 errors) + 2 adversarial fact-checkers on
the artifact (54 citation re-checks; no headline claim refuted; 9 minor findings fixed and
re-inspected).

| id | claim | evidence (executed) |
|----|-------|---------------------|
| O-38 | pypi latest 2026-07-14: sglang 0.5.15.post1 (torch==2.11.0, flashinfer[cu13]==0.6.12, transformers==5.12.1, sglang-kernel==0.4.4, flash-attn-4==4.0.0b15); tensorrt-llm 1.2.1 (torch>=2.9.1,<=2.10.0a0, transformers==4.57.3, flashinfer==0.6.4) | pypi JSON walks (executed curl; gate3 C31/C59/C84/C85) |
| O-39 | sglang v0.5.15.post1 registers Qwen3_5MoeForConditionalGeneration (+MTP arch), Qwen3OmniMoeForConditionalGeneration (audio encoder, talker off), Qwen3VLForConditionalGeneration with Pooler(LAST, normalize=True) | raw model files read at the tag (executed; gate3 C1–C10) |
| O-40 | sglang consumes the ModelOpt MIXED_PRECISION export on sm90: `modelopt_mixed` auto-detect; W4A16_NVFP4→Marlin A16 methods; FP4 min_capability 80; MoE Marlin fallback for cap∈[8.0,10.0); FP8-KV scales load path | modelopt_quant.py + model_config.py read at tag (executed; gate3 C14–C19) |
| O-41 | NVFP4-artifact floor is sglang 0.5.15: grep W4A16_NVFP4 = 0 @v0.5.14, 4 @v0.5.15 | dual-tag grep (executed; gate3 C21) |
| O-42 | sglang flag surface at tag: kv dtypes {auto,fp8_e5m2,fp8_e4m3,bf16,fp4_e2m1}; NO --calculate-kv-scales (0 hits); spec algos EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM/DFLASH; --is-embedding; radix default ON; mamba radix archs include our arch | server_args.py + overrides.py reads (executed; gate3 C22–C27) |
| O-43 | TRT-LLM 1.2.1 registers NO qwen3_5/omni arch; W4A16_NVFP4 absent from QuantAlgo enum; NVFP4 GEMM MIN_SM_VERSION=100 (CuteDSL [100,103]); Hopper matrix row NVFP4="."; trtllm-serve route set has NO /v1/embeddings; "MTP is currently only supported by Deepseek." | registry listings + mode.py + torch_custom_ops.py + openai_server.py + docs read at tag (executed; gate3 C41–C51) |
| O-44 | TRT-LLM 1.3.0rc adds Qwen3_5 as *ForCausalLM-only thin wrappers (rc15 file = 76 lines; exclude-normalizer drops visual*/mtp.*); routing is strict architectures[0] → ConditionalGeneration unrouted (R16) | modeling_qwen3_5.py + modeling_auto.py read at rc tags (executed; gate3 C66/C67) |
| O-45 | Single-env coexistence IMPOSSIBLE: 8 uv CONFLICT probes (P1/P2/P4–P9 incl. best-ever pairs vllm0.20.2×sglang0.5.11 and vllm0.19.1×trtllm-rc13 — both die on llguidance) + independent marker-evaluated pin-intersection sweep; two instruments, zero disagreements | uv 0.11.28 runs + compare script (EXECUTED; gate3 §5a + Appendix B; probe stderr preserved) |
| O-46 | Structural coexistence walls: llguidance disjoint vllm(≥1.3)-vs-sglang(<0.8) at every release; no trtllm ever admits torch 2.11.0 (ceiling 2.11.0a0 at rc20); outlines==0.1.11→outlines_core==0.1.26 vs vllm's ==0.2.14; mistral-common; fastapi windows disjoint | pin walks + outlines metadata read (executed; gate3 C81–C86) |
| O-47 | variant-c lock RESOLVES: 194 pins, zero drift 24/24 constraints (py3.12, manylinux_2_34, --prerelease=explicit with flash-attn-4 named directly — pydantic/einops stay stable) | uv pip compile (EXECUTED) → variant-c/serve-sglang/resolved-lock.txt |
| O-48 | variant-d lock RESOLVES with correct flavor: 177 pins, zero drift 20/20; torch==2.9.1+cu130 with SINGLE cu13 lineage (nccl-cu13==2.27.7 = torch's documented coupling; zero -cu12 packages); executed in linux/amd64 container (tensorrt wheel-stub sdists hard-fail on macOS) | containerized uv pip compile ×2 (EXECUTED) → variant-d/serve-trtllm/resolved-lock.txt |
| O-49 | pypi-only trtllm resolution lands cu12.8-lineage torch (wrong flavor — rejected by the variant-d build assert); trtllm 1.3.0rc15 is unsatisfiable even alone from pypi.org+pypi.nvidia.com (cuda-python>=13 vs torch 2.10's cuda-bindings==12.9.4) | uv runs (EXECUTED; gate3 C68/C69) |
| O-50 | torch 2.9.1+cu130 wheel EXISTS on download.pytorch.org/whl/cu130 (index carries 2.9.0→2.13.0 +cu130, PEP 658 metadata present); first negative grep was a shell-quoting instrument error caught by a second instrument | index fetch ×2 (EXECUTED; gate3 C71) |
| O-51 | sglang standalone needs prerelease allowance ONLY for flash-attn-4==4.0.0b15 (no-flag run fails on exactly that pin; failure artifact preserved: probes/s_alone_noprerelease.err.txt) | uv runs (EXECUTED; gate3 Appendix B S-alone) |
| O-52 | Artifact fact-check: 54 citation re-checks by 2 adversarial checkers — every headline quote character-consistent at the tags; findings limited to 4 locator/coverage minors + 5 consistency minors/nits, all fixed and fixer-lines re-inspected | fact-check workflow wf_1244affb-c39 (EXECUTED) |
