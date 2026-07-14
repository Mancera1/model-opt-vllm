# Gate-3 Adjudication — SGLang / TensorRT-LLM / Coexistence Feasibility (**RULED 2026-07-14**)

> **HUMAN RULING (2026-07-14, interactive session):** **D6 APPROVED** as drafted (sglang
> 0.5.15.post1 standalone, variant-c, five profiles). **D7 → no variant-e** (single-env
> coexistence impossible; split-venv fallback NOT built; the two standalones are the answer).
> **D8 → option (a)** (variant-d ships as engine infrastructure at stable 1.2.1 with all three
> hosted models as explicit exclusions + BLOCKED runbook; **no model profiles**; pip path with
> the two extra indexes per §4; R19 governs re-activation). **D9 APPROVED** as drafted.
> Open questions: (1) TensorRT-LLM confirmed; (2) one standalone image per engine; (3) moot —
> variant-e dropped; (4) per-engine anchor deviations accepted. Body below preserved as the
> decision record; option text not rewritten.

**Method & provenance.** Research date **2026-07-14**, repo at `a137586` (main, clean). Five
schema-forced research agents (SGLang platform, SGLang×models, TRT-LLM platform, TRT-LLM×models,
dependency-pin cartography; workflow `wf_5da823b4-503`, 0 errors, all embedded canary checks
matched orchestrator-known answers). Orchestrator's own instruments: **11 executed `uv` 0.11.28
resolution probes** (8 conflict probes, 3 standalone resolution attempts of which 2 produced
locks — 2 of the 3 run inside a linux/amd64 container because the `tensorrt` shim sdists refuse
to build on macOS), one marker-evaluated pairwise
pin-intersection sweep (linux/x86_64/cp312), and ~30 direct code/doc reads at git tags. Every
headline claim below carries ≥2 independent confirmations, one of which is an orchestrator read;
pedigree tags: **[exec]** orchestrator-executed command, **[read]** orchestrator read the primary
source at the tag, **[agent]** agent-verified with quote (spot-checked), **[doc]** vendor-doc-only
(weakest — flagged wherever load-bearing). Citations `C1…C86` in Appendix A; probe transcripts in
Appendix B. **No GPU exists this cycle**: every "SUPPORTED" in this document means
*code/metadata-verified at the tag*; all runtime behavior is **BLOCKED** pending an sm90 session.

Settled Gate-1/2 rulings (D1–D5, COR-1…8, R1–R13) are quoted, never re-opened.

---

## 1. Engine × model × capability matrix

Rows are the three hosted workloads (+ the quantized artifact). "Version floor" = first release
where the *capability for that exact model/artifact* exists. A cell without a citation would be
UNSUPPORTED by rule; every cell here is cited.

### 1a. SGLang (evaluated at `sglang==0.5.15.post1`, tag `v0.5.15.post1`, pypi 2026-07-14)

| Capability | Qwen3.6-35B-A3B (BF16 gen) | nvidia/Qwen3.6-35B-A3B-NVFP4 | Qwen3-VL-Embedding-8B (pooling) | Qwen3-Omni-30B-A3B-Instruct (audio in) |
|---|---|---|---|---|
| Supported (code level) | **SUPPORTED** — `Qwen3_5MoeForConditionalGeneration` registered, hybrid GDN implemented; on sm90 GDN decode runs triton with FP32 SSM state (FlashInfer GDN path is SM100+) [C1,C2,C30] | **SUPPORTED** — `hf_quant_config.json` auto-parses to `modelopt_mixed` (MIXED_PRECISION), same method name vLLM selects (Gate-1 O-18) [C14,C20] | **SUPPORTED** — arch registered **with built-in `Pooler(pooling_type=LAST, normalize=True)`** — byte-identical semantics to our vLLM D5 pins; `forward(get_embedding=True)` returns pooler output [C9,C10] | **SUPPORTED** — `Qwen3OmniMoeForConditionalGeneration` registered with a real audio encoder; `enable_talker=False` (thinker-only = our call type A) [C6] |
| Version floor | file at 0.5.9; **official floor 0.5.10** [doc] [C28,C29] | **0.5.15** — `W4A16_NVFP4` handling absent ≤0.5.14 (grep: 0 hits at v0.5.14, 4 at v0.5.15 [exec]) [C21] | 0.5.3 (pooler present at first tag) [C29] | 0.5.4 [C29] |
| OpenAI-compatible serving | **YES** — `/v1/chat/completions`; shared Qwen-VL/Omni processor handles text+image/video [C37,C38,C28] | YES (same path; quantization-agnostic) [C14,C38] | **YES with caveats** — `/v1/embeddings` exists; multimodal payload via `MultimodalEmbeddingInput`; may require `--chat-template` (docs' multimodal-embedding example uses one; no Qwen3-VL-Embedding-specific template found; model not doc-listed) [C11,C12,C13,C38] | **YES via sglang's `audio_url` content part** — OpenAI-official `input_audio` is **absent** from the protocol (grep 0 hits) → clients using the OpenAI SDK audio shape need a shim [C7,C8] |
| Consumes our ModelOpt NVFP4 artifact on sm90 | n/a (BF16 row) | **SUPPORTED (code)** — W4A16_NVFP4 linears route to a Marlin-based A16 method **unconditionally**; NVFP4 FusedMoE auto-falls back to Marlin for `(8,0) <= capability < (10,0)`; `get_min_capability()==80`; the SM100+ hard gate applies only to native *dense* NVFP4 GEMM, which this artifact does not use. Docs label NVFP4 "(Blackwell)" — code contradicts for this artifact (COR-11-style docs-vs-code, see §6) [C15,C16,C17,C18] | n/a (never quantized, D2) | n/a (BF16 default, D2) |
| FP8 KV cache (sm90) | **PARTIAL** — `--kv-cache-dtype fp8_e4m3` exists, but **no `--calculate-kv-scales` analogue** (grep 0 hits): scales come from a `--quantization-param-path` JSON or default to 1.0 with a documented accuracy warning [C22,C23] | **SUPPORTED** — checkpoint's FP8 KV scales load via `ModelOptFp8KVCacheMethod` on RadixAttention (no 1.0-scale caveat) [C19,C20] | NA — prefill-only workload; sglang even ships `--prefill-only-disable-kv-cache` for embedding mode [C23a] | PARTIAL — flag applies; same no-calibration caveat; D4 scoped FP8-KV to Qwen3.6 only anyway [C22] |
| Prefix-cache analogue (RadixAttention) | **SUPPORTED incl. hybrid-GDN** — arch is in `_MAMBA_RADIX_CACHE_ARCHS` + `_MAMBA_EXTRA_BUFFER_ARCHS`; `--mamba-radix-cache-strategy {auto,no_buffer,extra_buffer,…}`; radix ON by default [C26,C27] | SUPPORTED (arch-keyed, same) [C26] | PARTIAL — radix runs in embedding mode (value limited to shared instruction prefixes) [C23a,C27] | SUPPORTED — plain full-attention radix (not mamba-listed) [C26,C27] |
| Spec decode (MTP analogue) | **SUPPORTED (code)** — `--speculative-algorithm EAGLE` (NEXTN is an alias→EAGLE); draft ModelConfig **rewrites this exact arch → `Qwen3_5ForCausalLMMTP`** to consume the checkpoint's native 19-tensor MTP head. Caveat: arch is not in the auto-draft-path list → profile must pass explicit `--speculative-draft-model-path <same repo>` [C3,C4,C24,C25] | **SUPPORTED (code)** — `qwen3_5_mtp.py` has explicit handling for **this checkpoint family**: "The MTP model is unquantized in the nvfp4 checkpoint" → nulls draft quant_config for `modelopt_fp4`/`modelopt_mixed` (BF16 MTP head loads intact, matches D2/O-11) [C5] | NA — no decode phase [C24] | **UNSUPPORTED** — no omni MTP/draft mapping (absence greps recorded), no mtp weights in checkpoint, no HF draft found [C29a] |

**sglang spec-decode × prefix-cache note (differs from vLLM D3):** vLLM's Gate-1 ruling force-disables
prefix caching with MTP because of open vLLM bugs (#43559-class, R3). sglang's hybrid-mamba radix
cache is *designed to coexist* with spec decode (`extra_buffer` strategy) [C26,C28]. The vLLM bug
numbers do **not** transfer; whether sglang has an equivalent failure mode is **unknown → the
sglang MTP profile stays opt-in + benchmark-gated exactly like D3, and the `handoff/` canary
repros must be re-run against the sglang endpoint before enabling.** No re-adjudication of D3.

### 1b. TensorRT-LLM (evaluated at `tensorrt-llm==1.2.1`, tag `v1.2.1` = newest stable; rc-lane deltas flagged)

| Capability | Qwen3.6-35B-A3B (BF16 gen) | nvidia/Qwen3.6-35B-A3B-NVFP4 | Qwen3-VL-Embedding-8B (pooling) | Qwen3-Omni-30B-A3B-Instruct (audio in) |
|---|---|---|---|---|
| Supported (code level) | **UNSUPPORTED at 1.2.1** — arch appears nowhere (registry read [read] + full-checkout greps `qwen3_5\|qwen3.6\|gated delta` [agent]); nearest relative is text-only `Qwen3NextForCausalLM`. **rc-lane**: `modeling_qwen3_5.py` first appears in 1.3.0rc — registers **`Qwen3_5MoeForCausalLM` (text-only)**; our checkpoint's `architectures[0]` is the **ConditionalGeneration** name and routing is strict `architectures[0]` lookup → `Unknown architecture` unless a shim exists (**unresolved, R16**) [C41,C42,C43,C66,C67] | **UNSUPPORTED** — triple-kill: (1) base arch absent; (2) the export's per-layer algo string `W4A16_NVFP4` **does not exist in TRT-LLM's `QuantAlgo` enum** (repo-wide grep: 0 hits) — the checkpoint cannot even parse; (3) NVFP4 GEMM is hardware-gated `MIN_SM_VERSION=100` / CuteDSL `[100,103]`, Hopper row in the hardware matrix = "." , and **no Marlin-like Hopper fallback exists** [C44,C45,C46,C42] | **PARTIAL/UNSUPPORTED** — `Qwen3VLForConditionalGeneration` is registered for *generation* (beta, not in the support tables), but **no embedding semantics exist anywhere**: no `/v1/embeddings` route (full route list read [read]), no pooling runner; `trtllm-serve mm_embedding_serve` is an EPD-disaggregation *encoder* server, not an embedder [C72,C48,C49] | **UNSUPPORTED** — `Qwen3OmniMoe` appears nowhere in code/docs/examples (grep 0 hits); the serving layer's generic `audio_url` part has no model to feed (only listed audio arch is Phi4MM; `input_audio` is a TODO) [C41,C42,C50] |
| Version floor | none (absent from all stables; 1.2 release notes list Qwen3-Next/Qwen3-VL only). rc-lane: 1.3.0rc, text-only, with R16 caveat [C43,C66] | none [C44] | 1.2 for *generation* only (beta) [C72,C43] | none [C41] |
| OpenAI-compatible serving | trtllm-serve is OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`, `/v1/responses`) — **NA for this model** (won't load) [C48] | NA [C48] | **UNSUPPORTED for embeddings** — no `/v1/embeddings` route exists at all [C48] | NA (arch absent) [C48,C50] |
| Consumes our ModelOpt NVFP4 artifact on sm90 | n/a | **UNSUPPORTED** (see triple-kill above). Note the mechanism *exists* (pytorch backend parses `hf_quant_config.json`, handles MIXED_PRECISION per-layer + kv algo) — it is the specific `W4A16_NVFP4` algo + sm90 that are dead [C47,C44,C45,C46] | n/a | n/a |
| FP8 KV cache (sm90) | Engine-level **SUPPORTED**: Hopper row FP8-KV = Y; `KvCacheConfig(dtype='fp8')` — **pytorch backend only** (TRT backend asserts `dtype=='auto'`). NA for this model (won't load) [C45,C54] | NA [C54] | NA [C54] | NA [C54] |
| Prefix-cache analogue | Engine default: KV block reuse **ON** (`enable_block_reuse`). Warning sign for any future hybrid-GDN port: `Qwen3NextForCausalLM` row says **KV Cache Reuse = No** (linear-attention state defeats block reuse) [C53,C52] | NA (arch won't load) [C41] | NA (embedding mode absent) [C48] | NA (arch won't load) [C41] |
| Spec decode (MTP analogue) | **UNSUPPORTED for our models** — "MTP is currently only supported by Deepseek." [read]; `MTPDecodingConfig` thinking-phase token defaults are DeepSeek vocab ids; Qwen3Next row: MTP=No, EAGLE-3=No. EAGLE/DraftTarget/NGram configs exist for other archs [C51,C52,C55] | NA (arch won't load) [C41] | NA (no decode phase) [C48] | NA (arch won't load) [C41] |

**Bottom line 1b:** at the newest stable, the TensorRT-LLM image can serve **zero of the three
hosted workloads**. This is the matrix's honest answer, not a build defect. Options in §6/D8.

### 1c. vLLM baseline (already ruled at Gate-1/2 — reference only, no re-adjudication)

| Capability | Qwen3.6 BF16 | NVFP4 artifact | VL-Embedding | Omni audio |
|---|---|---|---|---|
| Existing ruling | SUPPORTED, floor v0.17.0 for MTP arch (COR-2, O-19) | SUPPORTED on sm90 via Marlin, `modelopt_mixed`, min_cap 75 (O-18, D2) | SUPPORTED `--runner pooling`, pooler LAST+normalize (O-32, D5) | SUPPORTED via vllm-omni@b9e9d23 on the 0.24.0 base (O-33, D1/COR-8) |

---

## 2. Candidate version sets (newest stable at research date, with coupling)

### SGLang — candidate `sglang==0.5.15.post1` (uploaded 2026-07-14) — **RECOMMENDED PIN**

| Axis | Value | Provenance |
|---|---|---|
| Python | ≥3.10; wheels cp310–cp313 **manylinux_2_34** x86_64 (+aarch64) → py3.12 OK; base glibc must be ≥2.34 (Ubuntu 24.04 = 2.39 ✓) | [C31] |
| torch | **`torch==2.11.0` exact + `torchaudio==2.11.0`; torchvision unpinned (resolves 0.26.0)** — *identical to the repo anchor*; torch 2.11.0's default PyPI wheel is cu13 (O-24) | [C31,C32] |
| CUDA lineage | cu13 via extras: `flashinfer_python[cu13]==0.6.12` + `flashinfer_cubin==0.6.12`, `nvidia-cutlass-dsl[cu13]==4.5.2`, `cuda-python>=13.0`; official image base is `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` (one patch below our 13.0.2; python-level pins are patch-agnostic) | [C31,C33,C36] |
| Key exact pins | `sglang-kernel==0.4.4` (renamed from `sgl-kernel`; 615 MB dep-free cu13 wheel on pypi.org), `transformers==5.12.1`, `torchao==0.17.0`, `xgrammar==0.2.1`, `torchcodec==0.11.1`, `apache-tvm-ffi==0.1.11`, `tilelang==0.1.11`, `tokenspeed_mla==0.1.7`, `openai==2.6.1`, `outlines==0.1.11`, **`flash-attn-4==4.0.0b15` (a beta pin inside a stable release — lock tooling needs `--prerelease=allow`, R15)** | [C31,C34,C32] |
| Resolution | **PASS [exec]**: `uv pip compile` (py3.12, x86_64-manylinux_2_34, `--prerelease=allow`) → **194 pins, zero drift** | [C32] |
| Serving | console script `sglang serve --model-path …` (new canonical entrypoint; `python -m sglang.launch_server` legacy-works); default port 30000; full OpenAI surface incl. `/v1/embeddings` | [C37,C38] |
| Landmine | `[tool.kernels.dependencies] "kernels-community/sgl-flash-attn3"` — a **runtime HF kernel download**; offline/airgapped images must pre-bake it (R14) | [C35] |

Why not 0.5.14/0.5.13 (older, transformers 5.8.1): they cannot load the NVFP4 artifact
(`W4A16_NVFP4` handling lands at 0.5.15) [C21]. Why not wait for vLLM-0.25-alignment: **no sglang
release pins flashinfer 0.6.13** (max 0.6.12 across the entire walk) [C84] — moot anyway per §5.

### TensorRT-LLM — candidate `tensorrt-llm==1.2.1` (stable, uploaded 2026-04-20); rc-lane alternative flagged

| Axis | 1.2.1 (stable) | 1.3.0rc20 (prerelease frontier — **pypi.nvidia.com/NGC only**; pypi.org rc-line stops at rc15) |
|---|---|---|
| Python | ≥3.10,<4 — but **real wheels are cp310/cp312 only** (no cp311/cp313, entire history) → py3.12 OK | same |
| torch | **`torch>=2.9.1,<=2.10.0a0`** (= 2.9.x only; excludes 2.10.0 final). Intended flavor: **`torch==2.9.1+cu130` from `download.pytorch.org/whl/cu130`** — the extra index is embedded at `requirements.txt` L1; the wheel exists there [exec]. Plain-PyPI torch 2.9.1 is **cu12.8-lineage** → a pypi-only install yields a **mixed cu12-torch + cu13-TRT env** (resolves [exec], wrong flavor — see D8) | `torch>=2.10.0,<=2.11.0a0` — **still excludes torch 2.11.0 final** (PEP 440: 2.11.0a0 < 2.11.0). **Unsatisfiable from pypi.org + pypi.nvidia.com** (rc15 probed [exec]; neither index hosts a cu13 torch 2.10): needs `cuda-python>=13` while PyPI torch 2.10.0 pins `cuda-bindings==12.9.4` → rc lane requires the pytorch cu130 index or the NGC container |
| CUDA | cu13: `tensorrt~=10.14.1` (→ `tensorrt-cu13` chain), `nvidia-nccl-cu13`, `cuda-python>=13` | cu13: `tensorrt~=10.15.1`; NGC rc20 container = CUDA 13.1.1 on NGC PyTorch 26.02 |
| Key exact pins | `transformers==4.57.3`, `flashinfer-python==0.6.4`, `triton==3.5.1`, `xgrammar==0.1.32`, `nvidia-cutlass-dsl==4.3.4`, `nvidia-modelopt[torch]~=0.37.0` (below the checkpoint's 0.44.0 producer and our quant image's 0.45.0), `fastapi<=0.121.3`, `llguidance==0.7.29`, `mistral-common==1.8.6`, `numpy<2` | `transformers==5.5.4`, `flashinfer-python==0.6.12`, `triton==3.6.0`, `nvidia-cutlass-dsl[cu13]==4.5.0` |
| Wheels/index | pypi.org = **10,133-byte wheel-stub sdist** (build-backend `wheel_stub.buildapi` → fetches the real wheel from pypi.nvidia.com at install; PKG-INFO carries the full authoritative requires_dist, byte-matching requirements.txt at the tag). Real wheels: pypi.nvidia.com, **2.51 GB cp312**, bare `linux_x86_64` tags (≤rc19); **no PEP 658 metadata** on that index | rc16+ switch to `manylinux_2_28` tags; rc16–rc20 absent from pypi.org entirely |
| NGC container | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` [agent, config-blob]: CUDA 13.1.0, **NGC torch 2.10.0a0+b4e4ee8 (custom build)**, py3.12, ~19.5 GiB compressed, `NVIDIA_REQUIRE_CUDA=cuda>=9.0` (loose), bundles compat driver libs 590.44.01. Official warning [read]: the PyPI wheel (torch 2.9.1) "may be incompatible" with the NGC container's torch — **never mix the two paths** | `release:1.3.0rc20` exists |
| Resolution | **PASS [exec]** (linux container): 180 pins. **Lock generation is linux-only** — the `tensorrt`→`tensorrt-cu13` shim sdists execute an OS check that hard-fails on macOS | rc15 FAIL from pypi.org + pypi.nvidia.com [exec]; rc20 not probed (wheels absent from both probe indexes' torch lineage) |
| Provenance sources | [C58,C59,C60,C61,C62,C63,C64,C68,C69,C70,C71,C65] | |

Driver note (R10 extension, do **not** copy the old analysis): CUDA 13.x native minimum driver is
**≥580** per NVIDIA's compatibility table [read, C70]; the existing repo images pass on enumerated
enterprise branches 535/550/565/570/575 via their blob-verified `NVIDIA_REQUIRE_CUDA` (O-17/O-36);
the TRT-LLM NGC container gates only `cuda>=9.0` but bundles 590.44.01 forward-compat libs. Any
new image built on `nvidia/cuda:13.0.x` inherits the same enumerated-branch behavior as the
existing images; an NGC-based image has the loose gate + compat-lib mechanism. Per-image driver
statements must be written per base at build time (Phase 1/2 acceptance item).

---

## 3. Lever mapping (Gate-1 D3/D4 → engine equivalents; settled rulings quoted, not re-opened)

| Gate-1 lever (ruling) | vLLM (as shipped) | SGLang 0.5.15.post1 | TensorRT-LLM 1.2.1 (pytorch backend) |
|---|---|---|---|
| Prefix caching ON (D3) | `--enable-prefix-caching` (default ON) | **RadixAttention, default ON**; hybrid-GDN handled via mamba radix cache, `--mamba-radix-cache-strategy` [C26,C27] | **KV block reuse, default ON** (`kv_cache_config.enable_block_reuse`) [C53] — but N/A-in-practice: no hosted model loads |
| Spec decode OFF by default; MTP opt-in `{"method":"mtp","num_speculative_tokens":2}` (D3) | as ruled | default OFF; opt-in `--speculative-algorithm EAGLE` (NEXTN alias) + `--speculative-num-steps/-eagle-topk/-num-draft-tokens`; native MTP head consumed via arch rewrite; **explicit `--speculative-draft-model-path` required** [C24,C25,C4] | `speculative_config: {decoding_type: MTP…}` exists but **MTP = DeepSeek-only** at 1.2.1 → **N/A for our archs** [C51] |
| MTP × prefix-cache interlock (D3: `--no-enable-prefix-caching` MANDATORY with MTP on vLLM) | as ruled (open bugs R3) | **does not map 1:1** — sglang's `extra_buffer` mamba strategy coordinates radix cache *with* spec decode; vLLM bug numbers don't transfer; keep MTP benchmark-gated + re-run canaries on sglang (no re-adjudication) [C26,C28] | N/A |
| torch.compile + CUDA graphs at defaults; never `--enforce-eager` (D4-1) | compile+graphs ON | **PARTIAL**: CUDA graphs ON by default (decode=full, prefill=breakable); `torch.compile` **OFF** (`--enable-torch-compile`, "Experimental"); no `--enforce-eager` flag exists at all (grep 0) [C39] | **PARTIAL**: `cuda_graph_config` ON by default (beta); `torch_compile_config` **OFF** (prototype) [C55] → on both new engines "defaults" = graphs-on/compile-off (COR-13 proposal) |
| Multimodal processor cache ON (D4-2) | `mm_processor_cache_gb=4` default ON | **PARTIAL/absent**: no processor-output cache; nearest are `--enable-mm-global-cache` (post-encoder embedding cache) and `--enable-prefix-mm-cache`, both **OFF** by default and semantically different [C40] | **NA**: no equivalent key (grep of llm_args yields only encoder/disagg knobs) [C55] |
| FP8 KV cache fp8_e4m3 on Qwen3.6 only; BF16 target adds `--calculate-kv-scales` (D4-3) | as ruled (O-34/O-35) | `--kv-cache-dtype fp8_e4m3` ✓; **no calculate-kv-scales analogue** — scales via `--quantization-param-path` JSON or default 1.0 (accuracy warning); NVFP4 artifact's shipped KV scales have a code-verified load path (`ModelOptFp8KVCacheMethod`) [C22,C23,C19] | `KvCacheConfig(dtype='fp8')`, pytorch backend only; no runtime-calibration flag found; Hopper FP8-KV = Y [C54,C45] |
| Chunked prefill (vLLM default ON) | ON | ON with GPU-tiered auto sizing (`--chunked-prefill-size`) [C39] | **OFF by default** (`enable_chunked_prefill=False`) — differs from vLLM [C56] |
| Overlap scheduler (no Gate-1 ruling — new surface) | n/a | on (default scheduling) | ON (`disable_overlap_scheduler=False`); two-model spec-decode flows require disabling [C57] |

---

## 4. TensorRT-LLM backend decision inputs (for D8)

| Option | What it is | Trade-offs | GPU-less cycle implication |
|---|---|---|---|
| **PyTorch backend (default)** | `LLM` class *is* `_TorchLLM`; `trtllm-serve` default `--backend pytorch`; loads HF checkpoints directly, no compile step [C55, C48] | Full lever set lives here (block reuse, fp8 KV, CUDA graphs, spec configs). Costs: torch 2.9.x wall; cp310/cp312 only; 2.5 GB wheel | **Fully shippable**: image + import assertions + profile YAML schema-checks; serving smoke stays BLOCKED |
| **Classic `trtllm-build` engine flow** | Ahead-of-time TensorRT engine compilation ("legacy" in the tag's docs) | Engine build **selects/times kernels on the target GPU and allocates device memory** → requires the GPU class at build; engines are device-type-bound by default; `KvCacheConfig.dtype` unsupported on TRT backend [C68, C54] | **Prebuilt engines are IMPOSSIBLE this cycle** (no GPU; an engine built elsewhere wouldn't target sm90). Ship at most a BLOCKED runbook of exact build commands |
| **AutoDeploy backend (beta)** | `--backend _autodeploy`: graph-transforms HF models into the pytorch runtime | Beta, "subject to change"; narrower model coverage | Nothing extra to ship; do not base defaults on it |
| **Serving front-end** | `trtllm-serve` (OpenAI-compatible: completions/chat/responses; **no /v1/embeddings**) + `--config` YAML mapped onto `TorchLlmArgs` keys | Profile YAMLs are schema-checkable GPU-less [C48] | Author + lint profiles now; first request is BLOCKED |
| **Base image A: pip-into-`nvidia/cuda:13.0.x-devel`** | House-style constraints/lock/manifest, fully auditable | Requires **two extra indexes**: `download.pytorch.org/whl/cu130` (torch 2.9.1+cu130 — wheel verified present [exec, C71]) and `pypi.nvidia.com` (the 2.5 GB trtllm wheel; wheel-stub sdist can also pull it) — both are build-time-only, no credentials; bare `linux_x86_64` wheel tag is pip-installable in-container by platform-tag semantics (no install executed this cycle; uv under a strict manylinux target rejects it, R17) | Buildable now; lock generation must run on linux (shim sdists) — same class of environment note as R12 |
| **Base image B: NGC `release:1.2.1` (digest-pinned)** | Everything pre-integrated by NVIDIA | 19.5 GiB compressed; **custom NGC torch** (2.10.0a0+b4e4ee8) — outside our provenance model; NVIDIA AI product EULA label; loose driver gate | Pullable/pinnable now |

**Recommendation (D8):** PyTorch backend + trtllm-serve; base image A (pip path) for house
consistency, with the two extra indexes recorded as a D-ruling deviation; NGC digest-pin documented
as the fallback base. **But see D8-scope first: at 1.2.1 the image hosts none of our models.**

---

## 5. Coexistence pre-probe — executed results

Instruments: `uv pip compile` 0.11.28 (the O-29 method; py3.12; `x86_64-manylinux_2_34` where
sglang participates — its wheels demand it — else `_2_28`) **plus** an independent
marker-evaluated pin-intersection sweep over pypi metadata. Both instruments agree on every case.
Verbatim resolver verdicts: Appendix B.

### 5a. Probe outcomes [all exec — citations C73–C80]

| Probe | Input set | Outcome | First blocker (uv verbatim) | Full contested surface (metadata sweep) |
|---|---|---|---|---|
| P1 | vllm 0.25.0 + sglang 0.5.15.post1 | **CONFLICT** | `apache-tvm-ffi` ==0.1.9 vs ==0.1.11 | + flashinfer 0.6.13/0.6.12, llguidance, torchcodec ≥0.14/==0.11.1, tilelang, tokenspeed-mla, outlines_core (7) |
| P2 | vllm 0.24.0 + sglang 0.5.15.post1 | **CONFLICT** | `apache-tvm-ffi` (same) | 4+ (llguidance, tilelang, tokenspeed-mla, outlines_core) |
| P7 | vllm 0.24.0 + sglang 0.5.14 (flashinfer/tvm-ffi/torch all aligned) | **CONFLICT** | `llguidance` >=1.7,<1.8 vs >=0.7.11,<0.8 | + tilelang, tokenspeed-mla, outlines_core |
| P8 | vllm 0.20.2 + sglang 0.5.11 — **best-aligned (vllm,sglang) pair in existence** (torch 2.11.0, flashinfer 0.6.8.post1, cutlass 4.4.2, xgrammar 0.1.32 all agree) | **CONFLICT** | `llguidance` >=1.3,<1.4 vs <0.8 | + outlines_core 0.2.14 vs 0.1.26 (transitive via sglang's `outlines==0.1.11`) |
| P4 | vllm 0.25.0 + tensorrt-llm 1.2.1 | **CONFLICT** | `torch` ==2.11.0 vs >=2.9.1,<=2.10.0a0 | 9 packages incl. transformers, flashinfer, xgrammar, cutlass-dsl, llguidance, mistral-common, **fastapi**, tvm-ffi |
| P5 | sglang 0.5.15.post1 + tensorrt-llm 1.2.1 | **CONFLICT** | `torch` (same window) | 8 packages (transformers 5.12.1/4.57.3, torchao, xgrammar, …) |
| P9 | vllm 0.19.1 + tensorrt-llm 1.3.0rc13 — **closest (vllm,trtllm) pair ever** (torch 2.10.0 ∈ window, flashinfer both ==0.6.6, transformers 4.57.3 admitted, triton both 3.6.0) | **CONFLICT** | `llguidance` >=1.3,<1.4 vs ==0.7.29 | + cutlass-dsl, mistral-common |
| P6 | **all four**: vllm 0.24.0 + vllm-omni@b9e9d23 + sglang 0.5.15.post1 + tensorrt-llm 1.2.1 | **CONFLICT** | `apache-tvm-ffi` | superset of the above |
| S-alone | sglang 0.5.15.post1 | **RESOLVES** — 194 pins (needs `--prerelease=allow` for `flash-attn-4==4.0.0b15`) | — | — |
| T-alone | tensorrt-llm 1.2.1 | **RESOLVES** — 180 pins (linux container; pypi-only → cu12.8-flavor torch, see §2) | — | — |
| TRC-alone | tensorrt-llm 1.3.0rc15 | **FAIL even alone from pypi.org + pypi.nvidia.com** (neither hosts a cu13 torch 2.10) — `cuda-python>=13` vs torch 2.10.0's `cuda-bindings==12.9.4` | — | — |
| (control) | vllm 0.24.0 + vllm-omni@commit | **RESOLVES** — pre-existing O-29 evidence (zero drift) | — | — |

### 5b. Structural analysis — why no version archaeology rescues this

1. **torch wall.** No tensorrt-llm release *ever* admits torch 2.11.0 final (1.1.0 ≤2.9.0; 1.2.x
   ≤2.10.0a0; every 1.3.0rc through rc20 ≤2.11.0a0, and 2.11.0a0 < 2.11.0 per PEP 440) [C85,C65].
   Every vllm ≥0.20.0 and every sglang ≥0.5.11 pins `torch==2.11.0` exactly; sglang has **no
   torch-2.10 release at all** (jumps 2.9.1 → 2.11.0) [C84,C86].
2. **llguidance fork.** sglang requires `<0.8.0` at every version walked; vllm requires `>=1.3.0`
   at every version since ≥0.17.1; trtllm pins `==0.7.29`. vllm-vs-sglang and vllm-vs-trtllm are
   **disjoint at every version pair in existence** — proven twice by uv on the best-ever pairs
   (P8, P9) [C81].
3. **Transitive poison.** sglang's `outlines==0.1.11` hard-pins `outlines_core==0.1.26` [read];
   vllm pins `outlines_core==0.2.14` directly [C82]. Plus `mistral-common` (trtllm ==1.8.6 vs
   vllm floors ≥1.9.1+) [C83] and even **fastapi** (trtllm ≤0.121.3 [C58] vs vllm 0.24.0's ≥0.133,<0.137 — §5a P4 contested-surface sweep, independently re-confirmed from vllm 0.24.0 pypi metadata by the fact-check pass).
4. **transformers epochs.** trtllm stable = 4.57.3 (4.x) vs sglang exact-5.x pins (5.6.0→5.12.1)
   and vllm-omni's `>=5.5.3`; the rc lane crosses to ==5.5.3/5.5.4 but keeps the torch wall [C85].
5. **flashinfer exact-pin lottery.** All three engines exact-pin it at almost every release;
   coincidental matches exist pairwise (vllm 0.24≡sglang 0.5.13+≡trtllm-rc20 at 0.6.12!) but each
   such pair dies on torch or llguidance anyway [C84,C85,C65].
6. The only torch-consistent trio ({vllm 0.16.0, sglang 0.5.10.x, trtllm 1.2.x} at torch 2.9.1)
   is **below the vllm v0.17.0 Qwen3.6 floor (COR-2)** — a resolving-but-cannot-serve museum
   piece — and it does not resolve anyway (flashinfer 0.6.3 vs 0.6.7.postX; transformers epochs)
   [C86,C84,C85].

### 5c. Verdict + fallback (presented, not promoted)

**Single-environment coexistence of even TWO of {vllm, sglang, tensorrt_llm} is impossible at
every version level; the four-package target is therefore impossible.** Evidence: 8 uv-executed
CONFLICTs including both best-ever pairs, + the exhaustive pin walk (two independent instruments,
zero disagreements).

**Nearest split alternative (fallback — requires human promotion at Gate-3):** one image, one
CUDA-13 userland, **per-engine Python virtualenvs**: `venv-vllm-omni` (vllm 0.24.0 +
vllm-omni@b9e9d23 — the O-29-proven pair), `venv-sglang` (0.5.15.post1), `venv-trtllm` (1.2.1;
its venv carries its own torch 2.9.1+cu130). Each venv is individually lock-proven above
(§5a S-alone/T-alone/control). Cross-venv import assertion = run each venv's python. The word
"coexistence" is **reserved** per the mission brief; proposed naming: `variant-e/serve-split-venv`
with README language "split-venv multi-engine image (single-env coexistence proven impossible —
see §5)". Alternative human choice: drop variant-e and let the standalone images be the answer.
Non-viable dodges, for completeness: `pip install --no-deps` / sequential-clobber installs would
"succeed" only by moving frozen core pins — exactly what the existing serve-omni `comm -23` guard
exists to fail — and are rejected as anti-pattern (runtime behavior of a clobbered env is
undefined).

---

## 6. Proposals (each requires human ratification)

### New rulings

- **D6 — SGLang standalone (variant-c).** Pin `sglang==0.5.15.post1` (floor forced by the NVFP4
  artifact: `W4A16_NVFP4` handling lands at 0.5.15 [C21]). Anchor preserved: py3.12 · CUDA 13.0.2
  base · torch==2.11.0 (sglang's own pin equals the repo anchor). All three models + the NVFP4
  artifact are code-supported (§1a); serve profiles: `configs/sglang/{qwen36-default,qwen36-nvfp4,
  embedding-pooling,omni-audio,qwen36-mtp-profile}.sh` mirroring existing conventions (BLOCKED
  preflight, IMAGE env, passthrough args; MTP profile opt-in + explicit
  `--speculative-draft-model-path` + canary-gated per §1a note; embedding-pooling carries the §1a
  chat-template caveat — no Qwen3-VL-Embedding-specific template found, resolve at profile
  authoring, verify at first GPU smoke). Build needs `--prerelease=allow`
  in lock tooling (R15) and must pre-bake the `sgl-flash-attn3` HF kernel (R14).
- **D7 — Coexistence verdict.** Single-env coexistence **impossible at any version level**
  (§5, two instruments). The deliverable becomes: (a) the split-venv fallback image
  `variant-e/serve-split-venv` as specified in §5c, **or** (b) no variant-e. **Human picks.**
  Naming must not say "coexistence" for (a).
- **D8 — TensorRT-LLM standalone (variant-d) scope + provisioning.** Backend: **PyTorch backend +
  trtllm-serve** (§4). Provisioning: **pip path into `nvidia/cuda:13.0.2-devel-ubuntu24.04`** with
  two additional build-time indexes (download.pytorch.org/whl/cu130 for `torch==2.9.1+cu130`;
  pypi.nvidia.com for the trtllm wheel), no credentials, mirror-hookable; NGC digest-pin
  documented as alternative. **Scope decision (human):**
  - **(a) RECOMMENDED:** ship the image as engine infrastructure with **all three models as
    explicit exclusions** (header + README), a generic BLOCKED import/serve-smoke runbook, and
    **no model profiles** (an honest zero — silent scope-narrowing would be a defect; loud
    exclusion is the design);
  - **(b)** additionally ship an **rc-lane experiment** (1.3.0rc20, nvidia-index-only,
    prerelease) for Qwen3.6 **text-only** — blocked on R16 (ConditionalGeneration routing
    unverified) and contrary to the no-prerelease convention;
  - **(c)** defer variant-d entirely until TRT-LLM ships Qwen3.5/3.6 multimodal + an embedding
    path in a stable.
- **D9 — Per-engine anchor deviations.** The py3.12/CUDA-13.0.2/torch-2.11.0 anchor holds for
  variant-c unchanged. Variant-d deviates by necessity: **torch 2.9.1+cu130** (TRT-LLM's own
  window), transformers 4.57.3, numpy<2 — recorded as deviations with provenance [C58,C59,C71];
  linux/amd64 + sm90 remain mandatory. Per-image driver statements re-derived per §2 driver note
  (never copied from R10).

### Corrections (COR-9…COR-13)

- **COR-9** — tensorrt-llm's pypi.org sdist is a **functional 10,133-byte wheel-stub** (backend
  `wheel_stub.buildapi` → pypi.nvidia.com) with complete authoritative metadata — not "~0-byte";
  resolution probes need no NVIDIA index, installs do [C60].
- **COR-10** — the sglang kernel package was **renamed**: `sgl-kernel` (pypi) is stale at 0.3.21;
  current releases hard-pin **`sglang-kernel`** (==0.4.4) as an ordinary pypi dependency [C34].
- **COR-11** — TRT-LLM's quantization doc headline "supports FP4 … on the latest Blackwell and
  Hopper GPUs" is contradicted by its own hardware matrix (Hopper NVFP4 = ".") and every NVFP4
  GEMM gate (`MIN_SM_VERSION=100`, CuteDSL `[100,103]`); mirror-image: sglang's docs label NVFP4
  "(Blackwell)" while its code runs our W4A16 artifact on sm90 via Marlin. **Trust the gates, not
  the headlines** [C45,C46,C17,C18,C28].
- **COR-12** — the tensorrt-llm rc frontier is **split**: pypi.org stops at 1.3.0rc15 while
  pypi.nvidia.com/NGC/GitHub carry rc16–rc20 (≈6 weeks newer, materially different pins) [C65].
- **COR-13** — Gate-1 D4-1's "torch.compile + CUDA graphs at defaults" coupling is
  **vLLM-specific**: on both new engines the default is CUDA-graphs-ON / torch.compile-OFF
  (experimental/prototype). Lever tables must track them separately [C39,C55].

### New risks (R14+)

| # | Risk | Notes |
|---|---|---|
| R14 | sglang declares an HF-hub kernel (`kernels-community/sgl-flash-attn3`) fetched at **runtime** — violates no-runtime-download posture unless pre-baked at build | bake + assert in Dockerfile; airgap blocker otherwise [C35] |
| R15 | sglang stable hard-pins a **beta** (`flash-attn-4==4.0.0b15`) — lock tooling must allow prerelease for exactly this pin; upstream may churn | [C31,C32] |
| R16 | TRT-LLM (rc lane only): checkpoint's `architectures[0]=Qwen3_5MoeForConditionalGeneration` has **no registered route** even at rc20 (`*ForCausalLM` only; strict architectures[0] lookup raises "Unknown architecture") — a config-compat shim may exist but was not found | blocks D8(b) until resolved [C66,C67] |
| R17 | trtllm wheels ≤rc19 use bare `linux_x86_64` tags — pip-in-container installs them, but strict-manylinux resolvers (uv with `--python-platform`) reject; rc20+ fixed | affects lock/build tooling only [C61] |
| R18 | NGC-container torch (custom `2.10.0a0+b4e4ee8`) vs PyPI wheel torch (2.9.1) are **incompatible provisioning paths** — mixing them is an explicit upstream warning | one path per image, asserted in manifest [C63,C64] |
| R19 | (if D8a) variant-d ships an engine that **serves none of the three hosted models** — the image is infrastructure + runbook only; must be loudly labeled in image header, README row, and manifest | exit criteria: TRT-LLM stable gains Qwen3.5/3.6-multimodal or embedding serving |
| R20 | sglang sm90 GDN decode runs triton with **FP32 SSM state** (SM100 uses flashinfer/bf16) — numerics/perf differ by arch generation; parity + capacity implications unmeasured | GPU-gated bench item [C30] |
| R21 | sglang FP8-KV on **BF16** targets has no runtime scale calibration (`--calculate-kv-scales` equivalent absent); scales default 1.0 with documented accuracy warning | sglang `qwen36-default` profile ships **without** fp8 KV until the GPU session validates a `quantization_param_path` route; NVFP4 profile unaffected (checkpoint ships KV scales; load path code-verified) [C22,C23,C19] |

### Directory layout & image tags (Phase 1–3, upon ratification)

```
variant-c/serve-sglang/{Dockerfile,constraints.txt,resolved-lock.txt}
    tag: model-opt-vllm/serve-sglang:sglang-0.5.15.post1
variant-d/serve-trtllm/{Dockerfile,constraints.txt,resolved-lock.txt}
    tag: model-opt-vllm/serve-trtllm:trtllm-1.2.1        (scope per D8; entire directory absent under D8(c))
variant-e/serve-split-venv/{Dockerfile,constraints.txt per venv,resolved-lock.txt per venv}
    tag: model-opt-vllm/serve-split-venv:multi-2026-07   (only if D7(a); name avoids "coexist")
configs/sglang/*.sh   (5 profiles per D6)
configs/trtllm/*.sh   (none under D8(a); rc-experiment profile only under D8(b))
docs/: this file; risk-register gains R14–R21; design-decisions gains D6–D9 + COR-9…13;
claim-ledger gains the Gate-3 execution rows (fleet + probes + orchestrator reads).
```

Inherited invariants apply to every new image: linux/amd64, sm90 first-class, no `:latest`,
no runtime pip, exact `==` pins with provenance comments, `resolved-lock.txt` via uv (with the
per-engine flags recorded: `--prerelease=allow` for sglang; linux-host execution + extra indexes
for trtllm), `/etc/image-manifest.json`, `com.model-opt-vllm.*` OCI labels, before/after core-pin
guard wherever more than one engine/package-family installs sequentially (all venvs of variant-e),
`PIP_INDEX_URL` mirror hook, `BUILD_DATE`, no baked credentials, HF cache mounted at runtime,
BLOCKED-protocol GPU preflight in every profile.

### Open questions → recommendations

1. **"TensorRT" means TensorRT-LLM?** Assumed **yes** throughout (bare TensorRT has no LLM serving
   front-end; trtllm-serve is the product analogue). **Confirm.** If bare TensorRT was meant, this
   document's variant-d section is void and Phase 2 needs a new research pass.
2. **One standalone image per engine?** **Yes.** The A/B split existed because vllm-omni couples
   to the vLLM 0.24 Docker base (COR-8) — a forcing function with no analogue here: one sglang env
   serves all three models (§1a); TRT-LLM has nothing to split (§1b). A literal A/B-style pair per
   engine would duplicate images with zero delta.
3. **Must the coexistence image include vllm-omni's audio path?** Moot for single-env (impossible
   regardless, §5). For the split-venv fallback: **include** the omni venv (vllm 0.24.0 +
   omni@b9e9d23 is the already-proven pair, O-29) — recommended venv set: {vllm-0.24.0+omni,
   sglang-0.5.15.post1, trtllm-1.2.1}. Adding a fourth venv (vllm 0.25.0, serve-main parity) is
   possible but doubles the vLLM disk share for little demo value — default: omit, note in README.
4. **Anchor deviations acceptable?** **Yes, in principle — recommended** (per-engine D9 rulings
   with provenance). H100/H200-only fleet (no local GPU) does not change this: all three engine
   lineages are CUDA-13-family; per-image driver statements per §2's R10-extension note. The
   binding alternative (force torch 2.11.0 everywhere) would simply exclude TensorRT-LLM from the
   program entirely (§5b torch wall).

---

## Appendix A — citations

All read 2026-07-14. `[read]` = orchestrator read; `[exec]` = orchestrator-executed command;
`[agent]` = fleet-verified with quote (orchestrator spot-checked the load-bearing ones);
`[doc]` = vendor documentation (weakest tier).

**SGLang (tag v0.5.15.post1 unless noted; base URL `https://raw.githubusercontent.com/sgl-project/sglang/v0.5.15.post1/`)**
- **C1** [read] `python/sglang/srt/models/qwen3_5.py:2179` — "EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]"
- **C2** [read] `qwen3_5.py:1789` — "class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):" (+ GDN classes at :157, :577)
- **C3** [read] `qwen3_5_mtp.py:413` — "EntryClass = [Qwen3_5ForCausalLMMTP]"
- **C4** [read] `configs/model_config.py:600-606` — draft-model arch rewrite: `"Qwen3_5MoeForConditionalGeneration"` → `self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"`
- **C5** [read] `qwen3_5_mtp.py:61-65` — "# The MTP model is unquantized in the nvfp4 checkpoint." + `quant_config.get_name() in ("modelopt_fp4","modelopt_mixed")` → `quant_config = None`
- **C6** [read] `qwen3_omni_moe.py:733` — "EntryClass = Qwen3OmniMoeForConditionalGeneration"; `:555` "self.enable_talker = False"; audio encoder classes :73/:200
- **C7** [read] `entrypoints/openai/protocol.py:522-523` — `type: Literal["audio_url"]`; grep `input_audio` in file = 0 hits
- **C8** [agent] `entrypoints/openai/serving_chat.py:606` — "audio_data=processed_messages.audio_data," ; `multimodal/processors/qwen_vl.py:685` — "audio_data=request_obj.audio_data,"
- **C9** [read] `qwen3_vl.py:1293` — "self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)"
- **C10** [agent] `qwen3_vl.py:1436` — `return self.pooler(hidden_states, forward_batch)` (get_embedding branch)
- **C11** [read] `server_args.py:478` — `is_embedding: A[bool, "Whether to use a CausalLM as an embedding model."] = False`
- **C12** [agent] `entrypoints/openai/protocol.py:1055-1058` — "class MultimodalEmbeddingInput(BaseModel):"
- **C13** [doc] docs.sglang.io/docs/supported-models/embedding_models — multimodal embedding example uses `--is-embedding --chat-template gme-qwen2-vl`; page does not list Qwen3-VL-Embedding
- **C14** [read+agent] `layers/quantization/modelopt_quant.py:638-644` — `quant_method == "modelopt_mixed"`; `configs/model_config.py:1153-1163` — MIXED_PRECISION → `return {"quant_method": "modelopt_mixed", …}`
- **C15** [read] `modelopt_quant.py:815-816, 829-830` — `W4A16_NVFP4` → `ModelOptNvFp4A16LinearMethod` / `ModelOptNvFp4FusedMoEMethod(nvfp4a16_config)`
- **C16** [read] `modelopt_quant.py` `ModelOptFp4Config.get_min_capability` → "return 80" (mixed inherits at :651-653); FP8 min_cap 89 at :430-431
- **C17** [read] `modelopt_quant.py:1899-1900` — "use_marlin_fallback = (8, 0) <= capability < (10, 0)"
- **C18** [agent] `modelopt_quant.py:1523-1527` — "ModelOpt NVFP4 native dense GEMM backends require SM100+. Use --fp4-gemm-backend marlin on SM80-SM90."
- **C19** [read] `modelopt_quant.py:820-821` — mixed `get_quant_method` → `ModelOptFp8KVCacheMethod(self.fp8_config)` for RadixAttention
- **C20** [read, prior cycle O-10 + agent canary] `huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/raw/main/hf_quant_config.json` — producer modelopt 0.44.0; MIXED_PRECISION; `"kv_cache_quant_algo": "FP8"`; experts `"quant_algo": "W4A16_NVFP4"`
- **C21** [exec] grep `W4A16_NVFP4` in `modelopt_quant.py`: **0 at v0.5.14, 4 at v0.5.15**
- **C22** [read] `server_args.py:567` — kv choices `["auto","fp8_e5m2","fp8_e4m3","bf16","bfloat16","fp4_e2m1"]`
- **C23** [read] `server_args.py:546-557` — `quantization_param_path`: "Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy issues."
- **C23a** [read] `server_args.py:753` — `--prefill-only-disable-kv-cache` "…only valid with --is-embedding, --chunked-prefill-size=-1, --disable-radix-cache…"
- **C24** [read] `server_args.py:1583-1586` — "Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH."
- **C25** [agent] `arg_groups/speculative_hook.py:42-49` — NEXTN/EAGLE alias resolution; auto-draft-path list lacks Qwen3_5 archs (S2 trace)
- **C26** [read] `arg_groups/overrides.py:960-990` — `_MAMBA_RADIX_CACHE_ARCHS` and `_MAMBA_EXTRA_BUFFER_ARCHS` both contain "Qwen3_5MoeForConditionalGeneration"
- **C27** [read] `server_args.py:755` — `disable_radix_cache … = False`
- **C28** [doc] docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6 — "SGLang >=0.5.10 is required for Qwen3.6."; full `sglang serve` example incl. `--speculative-algorithm EAGLE`; NVFP4 table row labeled "(Blackwell)"; prose names a nonexistent `--mamba-scheduler-strategy` flag (code: `--mamba-radix-cache-strategy`, `server_args.py:1919-1923`)
- **C29** [agent, tag-404 probes] `qwen3_5.py` 404@v0.5.8 / 200@v0.5.9; `qwen3_vl.py` 404@v0.5.2 / 200@v0.5.3 (pooler already present); `qwen3_omni_moe.py` 404@v0.5.3 / 200@v0.5.4
- **C29a** [agent] omni spec-decode absence: models dir has no omni-MTP file; `model_config.py` draft rewrites lack Qwen3OmniMoe; no mtp.* weights in checkpoint; HF search empty
- **C30** [agent] `server_args.py:4676` — "SM100+ FlashInfer GDN decode requires bf16 state; SM90 uses float32."
- **C31** [read/exec] `pypi.org/pypi/sglang/0.5.15.post1/json` requires_dist — torch==2.11.0, torchaudio==2.11.0, flashinfer_python[cu13]==0.6.12, flashinfer_cubin==0.6.12, nvidia-cutlass-dsl[cu13]==4.5.2, transformers==5.12.1, torchao==0.17.0, xgrammar==0.2.1, torchcodec==0.11.1 (linux-x86_64-applicable marker), sglang-kernel==0.4.4, flash-attn-4==4.0.0b15, apache-tvm-ffi==0.1.11, tilelang==0.1.11, tokenspeed_mla==0.1.7, llguidance>=0.7.11,<0.8.0, outlines==0.1.11, openai==2.6.1; wheels cp310–cp313 manylinux_2_34
- **C32** [exec] `uv pip compile` sglang==0.5.15.post1 (py3.12, x86_64-manylinux_2_34, --prerelease=allow) → RESOLVES, 194 pins; resolved: torch==2.11.0, torchvision==0.26.0, sglang-kernel==0.4.4, llguidance==0.7.30, flash-attn-4==4.0.0b15
- **C33** [read] `docker/Dockerfile` L1-2 — `ARG CUDA_VERSION=13.0.1` / `FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04`; L195-201 CUINDEX case hard-errors on other values
- **C34** [agent+exec] `pypi.org/pypi/sglang-kernel/0.4.4/json` — cp310-abi3 manylinux2014 x86_64 wheel 615,071,908 B, requires_dist None; legacy `sgl-kernel` stale at 0.3.21
- **C35** [agent] `python/pyproject.toml` `[tool.kernels.dependencies]` — `"kernels-community/sgl-flash-attn3" = 1`
- **C36** [agent] Docker Hub lmsysorg/sglang — `v0.5.15.post1`(=cu130, digest sha256:289cf51d…), `-cu130`, `-cu129`, `-runtime` variants (2026-07-14)
- **C37** [agent] `python/pyproject.toml` L180 `sglang = "sglang.cli.main:main"`; `launch_server.py` legacy warning; default port 30000 (`server_args.py:978`)
- **C38** [agent] `entrypoints/http_server.py` — `/v1/chat/completions` L1614, `/v1/completions` L1606, `/v1/embeddings` L1625, `/v1/rerank` L1836 (+transcriptions/realtime/score/messages)
- **C39** [agent] CUDA-graph defaults `model_executor/cuda_graph_config.py:101,109` (decode FULL, prefill BREAKABLE on CUDA); `--enable-torch-compile` "Experimental feature." (`server_args.py:1564-1567`); grep `enforce_eager` = 0 hits; chunked-prefill auto-tiered (`server_args.py:655-658, 3499-3537`)
- **C40** [agent] `server_args.py:2075-2078, 2097-2100` — `--enable-prefix-mm-cache` ("Currently only supports mm-only"), `--enable-mm-global-cache` (ViT-skip embedding cache); both default OFF

**TensorRT-LLM (tag v1.2.1 unless noted; base URL `https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v1.2.1/`)**
- **C41** [read] GitHub contents API `tensorrt_llm/_torch/models?ref=v1.2.1` (47 files) — qwen set = qwen/qwen2vl/qwen3/qwen3_moe/qwen3_next/qwen3vl/qwen3vl_moe/qwen_moe; **no qwen3_5, no omni**
- **C42** [agent] full-checkout greps at the tag: `qwen3_5|qwen3\.5|qwen3\.6|qwen3_6|gated[_ ]?delta` → only qwen3_next/fla internals; `qwen.?omni|Qwen3OmniMoe` → 0; `W4A16_NVFP4` → 0; `/embeddings` in `tensorrt_llm/serve` → 0 routes
- **C43** [agent] `docs/source/models/supported-models.md:30` Qwen3Next row; `docs/source/release-notes.md:22` — "Added beta support for K-EXAONE, Nemotron Nano V3, Qwen3-Next and Qwen3-VL."
- **C44** [read] `tensorrt_llm/quantization/mode.py:23-48` — QuantAlgo members include W4A16, W4A16_AWQ, W4A16_GPTQ, W4A16_MXFP4, NVFP4, W4A8_NVFP4_FP8, NVFP4_AWQ — **no W4A16_NVFP4**
- **C45** [read] `docs/source/features/quantization.md` — L23 headline "supports FP4 and FP8 quantization on the latest Blackwell and Hopper GPUs." vs Hardware Support Matrix L121-127: Hopper row NVFP4 "." / FP8 "Y" / FP8-KV "Y"
- **C46** [read] `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:556-557` — "# Minimum supported architecture: SM100 (Blackwell)" / `MIN_SM_VERSION = 100`; :813-818 CuteDSL "requires SM 100 (B200) or SM 103 (B300)"
- **C47** [agent] `tensorrt_llm/_torch/model_config.py:477-481` — "# quantized ckpt in modelopt format" `cached_file(checkpoint_dir,'hf_quant_config.json')`; :253-289 MIXED_PRECISION per-layer + `kv_cache_quant_algo` handling
- **C48** [read] `tensorrt_llm/serve/openai_server.py:255-291` (LLM role) + :321-336 (MM_ENCODER role) — route-registration blocks read in full (list abridged here): health/version/models/metrics/kv_cache_events/completions/chat/responses/memory-ops; **no /v1/embeddings**
- **C49** [agent] `tensorrt_llm/commands/serve.py:573` — `@click.command("mm_embedding_serve")`; `openai_server.py:206` asserts `MultimodalEncoder` (EPD encoder role)
- **C50** [agent] `tensorrt_llm/serve/chat_utils.py:63` — `"audio_url",` in VALID_MESSAGE_CONTENT_MM_PART_TYPES; :58 "# TODO: Add \"input_audio\"…"; only audio-modality arch row = Phi4MM (`supported-models.md:63`)
- **C51** [read] `docs/source/features/speculative-decoding.md:101` — "MTP is currently only supported by Deepseek."; :16 table "| MTP | Deepseek V3/R1 |"; DeepSeek token ids in `MTPDecodingConfig` (`llm_args.py:1125-1127`)
- **C52** [agent] `supported-models.md:42` — Qwen3Next row: KV Cache Reuse **No**, MTP **No**
- **C53** [agent] `docs/source/features/kvcache.md:59` — "Block reuse across requests is enabled by default…"; `llm_args.py:1632-1635` `enable_block_reuse` default True
- **C54** [agent] `llm_args.py:1698-1701` KvCacheConfig `dtype: default "auto"`; :3188 `elif …dtype == 'fp8'`; :2695 TRT-backend assert "KvCacheConfig.dtype is not supported by the TensorRT backend."
- **C55** [agent] `llm_args.py:2777-2778` `cuda_graph_config … default_factory=CudaGraphConfig` (ON, beta); :2864-2865 `torch_compile_config … default=None … status="prototype"`; mm-cache grep → only `mm_encoder_only`
- **C56** [agent] `llm_args.py:2006` — `enable_chunked_prefill: bool = Field(default=False,`
- **C57** [agent] `llm_args.py:2792-2794` — `disable_overlap_scheduler` default False; two-model spec flows need it True (speculative-decoding.md)
- **C58** [read+agent] `requirements.txt` at v1.2.1 — L1 `--extra-index-url https://download.pytorch.org/whl/cu130`; L24 `torch>=2.9.1,<=2.10.0a0`; L28-29 "# torch 2.9.1+cu130 depends on nvidia-nccl-cu13==2.27.7"; transformers==4.57.3, flashinfer-python==0.6.4, fastapi<=0.121.3, llguidance==0.7.29, mistral-common==1.8.6, nvidia-modelopt[torch]~=0.37.0 — byte-consistent with pypi metadata [read of pypi JSON]
- **C59** [exec] `pypi.org/pypi/tensorrt-llm/{1.2.1,1.2.0,1.1.0,1.0.0,1.3.0rc15}/json` requires_dist walk (see §2/§5b)
- **C60** [agent] extracted `tensorrt_llm-1.2.1.tar.gz` (10,133 B) — pyproject: `requires = ["wheel-stub"]`, `build-backend = "wheel_stub.buildapi"`, `index_url = "https://pypi.nvidia.com/"`; PKG-INFO carries full requires_dist
- **C61** [agent] pypi.nvidia.com/tensorrt-llm/ — 333 files, cp310+cp312 only; 1.2.1 cp312 x86_64 = 2,514,282,461 B (HTTP HEAD); bare `linux_x86_64` tags ≤rc19, `manylinux_2_28` from rc16+ (rc20 confirmed); rc15.post1+rc16–rc20 absent from pypi.org
- **C62** [agent] pypi.nvidia.com simple-index HTML: zero `data-dist-info-metadata`/`data-core-metadata` attributes → resolvers must download full wheels for metadata
- **C63** [agent] `nvcr.io/v2/nvidia/tensorrt-llm/release/tags/list` (anonymous token) — tags 0.20.0…1.3.0rc20; config blob of `release:1.2.1` (amd64): `CUDA_VERSION=13.1.0.036`, `com.nvidia.pytorch.version=2.10.0a0+b4e4ee8`, `NVIDIA_PYTORCH_VERSION=25.12`, `NVIDIA_REQUIRE_CUDA=cuda>=9.0`, compat driver 590.44.01, 20,940,306,865 B ≈ 19.5 GiB compressed; rc20 blob: `NVIDIA_PYTORCH_VERSION=26.02`, CUDA 13.1.1
- **C64** [read] `docs/source/installation/linux.md:43` — "The TensorRT LLM wheel on PyPI is built with PyTorch 2.9.1. This version may be incompatible with the NVIDIA NGC PyTorch 25.12 container… install the pre-built wheel located at `/app/tensorrt_llm` inside the TensorRT LLM NGC Release container instead."
- **C65** [agent] `requirements.txt` at v1.3.0rc20 — `torch>=2.10.0,<=2.11.0a0`, `transformers==5.5.4`, `flashinfer-python==0.6.12`, `nvidia-cutlass-dsl[cu13]==4.5.0`, `tensorrt~=10.15.1`
- **C66** [read] `tensorrt_llm/_torch/models/modeling_qwen3_5.py` at v1.3.0rc15 (76 lines) — registers only `Qwen3_5MoeForCausalLM` + `Qwen3_5ForCausalLM` (thin Qwen3Next wrappers); `_normalize_qwen35_exclude_modules` strips `model.language_model.` prefixes and **drops** `model.visual*`/`mtp.*` from NVFP4/FP8 exclude lists
- **C67** [read] `tensorrt_llm/_torch/models/modeling_auto.py:19,62` at v1.3.0rc15 (same routing at v1.2.1: `:16, :41-43`) — routing takes `pretrained_config.architectures[0]`; unmatched → `Unknown architecture for AutoModelForCausalLM: …`
- **C68** [exec] `uv pip compile` tensorrt-llm==1.2.1 (linux/amd64 container, py3.12, +pypi.nvidia.com) → RESOLVES, 180 pins: torch==2.9.1 (+`nvidia-*-cu12==12.8.x` from PyPI torch) alongside `tensorrt-cu13` 10.14.1.48.post1 / `nvidia-nccl-cu13==2.28.9` / `cuda-python 13.3.1`; transformers==4.57.3, fastapi==0.121.3, flashinfer==0.6.4, nvidia-modelopt==0.37.0, numpy==1.26.4. macOS lock attempt fails: "RuntimeError: TensorRT currently only builds wheels for Linux and Windows" (shim sdist)
- **C69** [exec] `uv pip compile` tensorrt-llm==1.3.0rc15 (linux container, prerelease, +nvidia index) → **NO SOLUTION**: "torch==2.10.0 depends on cuda-bindings==12.9.4 … tensorrt-llm==1.3.0rc15 depends on cuda-python>=13"
- **C70** [read] docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html — table row: "CUDA 13.x >= 580 N/A (backward compatibility applies for newer drivers)"; "CUDA 12.x >= 525 < 580 …"
- **C71** [exec] download.pytorch.org/whl/cu130/torch/ index — versions present: 2.9.0, **2.9.1**, 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0 (+cu130 local tags; PEP 658 metadata attributes present)
- **C72** [agent] `tensorrt_llm/_torch/models/modeling_qwen3vl.py:1134` — `@register_auto_model("Qwen3VLForConditionalGeneration")` (beta generation support; not in support tables)

**Coexistence / cross-engine [all exec unless noted]**
- **C73–C80** — probes P1,P2,P4,P5,P6,P7,P8,P9 as tabulated in §5a; verbatim cores in Appendix B
- **C81** llguidance disjointness: sglang `<0.8.0,>=0.7.11` at every walked version; vllm `>=1.3.0,<1.4.0` (0.19.1/0.20.2) → `>=1.7.0,<1.8.0` (0.24/0.25); trtllm `==0.7.29` — uv verdicts P7/P8/P9 all terminate on llguidance [exec + agent]
- **C82** [read] `pypi.org/pypi/outlines/0.1.11/json` → `outlines_core==0.1.26`; vllm 0.24.0 requires_dist → `outlines_core==0.2.14` [agent quote of pypi JSON]
- **C83** mistral-common: trtllm ==1.8.6 vs vllm 0.24.0 >=1.11.3 (my sweep) / >=1.9.1+ at older vllm [agent]
- **C84** [exec] sglang pin walk 0.5.9→0.5.15.post1 (pypi JSON): torch 2.9.1 (≤0.5.10.post1) → 2.11.0 (≥0.5.11); flashinfer 0.6.3/0.6.7.postX/0.6.8.post1/0.6.11.post1/0.6.12 (max 0.6.12; **no 0.6.13 anywhere**); transformers 4.57.1→5.3.0→5.6.0→5.8.1→5.12.1; tvm-ffi range→==0.1.9(0.5.12-14)→==0.1.11(0.5.15+)
- **C85** [exec] trtllm pin walk 1.0.0→1.3.0rc20: torch windows ≤2.8.0a0 / ==2.9.0 / [2.9.1,2.10.0a0] / [2.10.0,2.11.0a0]; transformers 4.53.1/4.56.0/4.57.3/5.5.3/5.5.4; cuda lineage cu12→cu13 at 1.1.0
- **C86** [agent] vllm torch boundary walk (pypi JSON): 0.20.0+ ==2.11.0 (0.20.2 independently repo-verified, O-03); 0.17.0–0.19.1 ==2.10.0; 0.16.0 ==2.9.1; transformers floors admit 4.57.3 only ≤0.23.0-era (`!=5.x-early,>=4.56` forms)

## Appendix B — probe inputs & verbatim resolver verdicts

Probe artifacts (inputs, full stderr, locks) are preserved in the session scratchpad
(`probes/*.in|.err.txt|.lock.txt`) and can be committed under `handoff/results/` on request.
Command shape: `uv pip compile <in> --python-version 3.12 --python-platform <plat> [--prerelease=allow] [--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match]`.

- **P1** `vllm==0.25.0 + sglang==0.5.15.post1` → "Because vllm==0.25.0 depends on apache-tvm-ffi==0.1.9 and sglang==0.5.15.post1 depends on apache-tvm-ffi==0.1.11 … unsatisfiable."
- **P2** `vllm==0.24.0 + sglang==0.5.15.post1` → same apache-tvm-ffi verdict.
- **P7** `vllm==0.24.0 + sglang==0.5.14` → "vllm==0.24.0 depends on llguidance…>=1.7.0,<1.8.0 and sglang==0.5.14 depends on llguidance>=0.7.11,<0.8.0 … unsatisfiable."
- **P8** `vllm==0.20.2 + sglang==0.5.11` → "vllm==0.20.2 depends on llguidance…>=1.3.0,<1.4.0 and sglang==0.5.11 depends on llguidance>=0.7.11,<0.8.0 … unsatisfiable."
- **P4** `vllm==0.25.0 + tensorrt-llm==1.2.1` → "vllm==0.25.0 depends on torch==2.11.0 and tensorrt-llm==1.2.1 depends on torch>=2.9.1,<=2.10.0a0 … incompatible."
- **P5** `sglang==0.5.15.post1 + tensorrt-llm==1.2.1` → "sglang==0.5.15.post1 depends on torch==2.11.0 and tensorrt-llm==1.2.1 depends on torch>=2.9.1,<=2.10.0a0 … incompatible."
- **P9** `vllm==0.19.1 + tensorrt-llm==1.3.0rc13` (prerelease) → "vllm==0.19.1 depends on llguidance…>=1.3.0,<1.4.0 and tensorrt-llm==1.3.0rc13 depends on llguidance==0.7.29 … incompatible."
- **P6** all four → apache-tvm-ffi verdict (first blocker).
- **S-alone** `sglang==0.5.15.post1` → RESOLVES (194 pins) with `--prerelease=allow`; without the flag: "there is no version of flash-attn-4==4.0.0b15 … pre-releases weren't enabled (try: `--prerelease=allow`)" (failure re-executed and preserved as `probes/s_alone_noprerelease.err.txt`).
- **T-alone** `tensorrt-llm==1.2.1` → RESOLVES (180 pins) in linux/amd64 container; on macOS: "RuntimeError: TensorRT currently only builds wheels for Linux and Windows" (tensorrt/tensorrt-cu13 shim sdists).
- **TRC-alone** `tensorrt-llm==1.3.0rc15` → NO SOLUTION: "torch==2.10.0 depends on cuda-bindings…==12.9.4 … tensorrt-llm==1.3.0rc15 depends on cuda-python>=13 … unsatisfiable."

---

*Draft prepared 2026-07-14 by the Gate-3 orchestrator. Nothing in Phases 1–3 begins until the
human rules on D6–D9, the COR-9…13 corrections, R14–R21, and the four open questions.*
