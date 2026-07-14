# serve-trtllm (Variant D) — BLOCKED smoke runbook + R19 re-activation checklist

**Scope (Gate-3 ruling D8(a), 2026-07-14).** The `serve-trtllm` image is **engine
infrastructure**: TensorRT-LLM 1.2.1 hosts **none** of the three hosted models
(evidence: `docs/gate3-adjudication.md` §1b; exclusions are printed at build and recorded in
`/etc/image-manifest.json`). This runbook is everything that CAN be exercised, and the
checklist that re-activates the image when upstream support lands.

Everything below is **BLOCKED on environment** as marked — nothing here has been executed in
the authoring environment (no linux/amd64 build host with disk, no GPU). Complete and runnable
as written.

## 1. Build + in-image assertions (BLOCKED: linux/amd64 build host — R12-class)

```bash
docker build --platform linux/amd64 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t model-opt-vllm/serve-trtllm:trtllm-1.2.1 \
  variant-d/serve-trtllm
```

The build itself runs the acceptance gates: `pip check`, the 13-pin exact-version matrix,
the **torch==2.9.1+cu130 flavor assert** (fails on any cu12-lineage contamination — R18),
`import tensorrt` + `import tensorrt_llm` under CUDA driver stubs, and the manifest write
with the exclusions block. A green build = the infrastructure claim is verified.

Post-build, capture the canonical lock (Gate-2 pattern):

```bash
docker run --rm --entrypoint bash model-opt-vllm/serve-trtllm:trtllm-1.2.1 \
  -lc 'pip freeze' > locks/serve-trtllm.lock.txt
docker run --rm --entrypoint cat model-opt-vllm/serve-trtllm:trtllm-1.2.1 \
  /etc/image-manifest.json
```

## 2. CLI surface smoke (BLOCKED: built image; no GPU needed)

```bash
docker run --rm --entrypoint trtllm-serve model-opt-vllm/serve-trtllm:trtllm-1.2.1 --help
```

Expect the OpenAI-compatible server help: `--backend {pytorch,tensorrt,_autodeploy}` with
pytorch as default, `--config` YAML hook (maps onto `TorchLlmArgs`: `kv_cache_config`,
`cuda_graph_config`, `speculative_config`, `enable_chunked_prefill`,
`disable_overlap_scheduler`). Note there is **no `/v1/embeddings`** in the route set.

## 3. GPU serve smoke — NON-hosted reference model (BLOCKED: sm90 host)

Serves a model TensorRT-LLM 1.2.1 *does* support, purely to prove the runtime path and for
engine-comparison baselines. The documented Qwen3Next example from the tag's own support
matrix (`docs/source/models/supported-models.md:30` at v1.2.1):

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v "$HF_HOME:/hf-cache" -e HF_HOME=/hf-cache ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  model-opt-vllm/serve-trtllm:trtllm-1.2.1 \
  Qwen/Qwen3-Next-80B-A3B-Thinking --host 0.0.0.0 --port 8000
# then:
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Next-80B-A3B-Thinking","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

Caveats from the verified support matrix: this arch has KV-reuse=No and MTP=No in TRT-LLM —
do not read its numbers as predictive of a future Qwen3.6 port beyond order-of-magnitude.
(80B needs multi-GPU or quantized weights; size the host accordingly — this is a comparison
exercise, not a deployment.)

## 4. What must NOT be attempted on this image

- Serving any of the three hosted models (loud-fails or silently wrong; see exclusions).
- `trtllm-build` engine compilation anywhere but the target sm90 GPU class (device-timing
  build; engines are device-type-bound — gate3 §4).
- Installing the NGC container's torch or wheel into this env, or vice versa (R18: the PyPI
  wheel is built for torch 2.9.1; NGC uses a custom 2.10.0a0 build — never mix).

## 5. R19 re-activation checklist (run when a TRT-LLM STABLE release notes any of our archs)

1. Confirm in the new tag's code (not docs headline): `Qwen3_5MoeForConditionalGeneration`
   routes in `_torch/models` (the rc-lane registered only `*ForCausalLM` — R16), and/or a
   `/v1/embeddings`-capable path exists, and/or `Qwen3OmniMoe*` appears.
2. For the NVFP4 artifact: `W4A16_NVFP4` present in `tensorrt_llm/quantization/mode.py`'s
   QuantAlgo enum AND a Hopper-capable path for it (the 1.2.1 gates were `MIN_SM_VERSION=100`
   with no Marlin fallback).
3. Re-run the Gate-3 §5-style pin walk for the new version (torch window, transformers epoch,
   the two extra indexes still required?).
4. Bump `constraints.txt` + regenerate `resolved-lock.txt` (linux container; both extra
   indexes; zero-drift check), rebuild, then author `configs/trtllm/*.sh` profiles mirroring
   the sglang set — **only at that point** does variant-d gain model profiles.
5. New model profiles inherit the full gate chain: scale-scan promotion gate (NVFP4), parity
   procedure (embedding), canary suite (spec decode).
