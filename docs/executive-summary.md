# Executive Summary — Inference Container Modernization (2026-07-13)

**What this is.** A reproducible replacement for the legacy "base image + post-hoc upgrade script"
pattern: everything pinned at build time, no runtime pip, public sources with a prod-mirror build-arg
hook. Two serving topologies shipped side-by-side for evaluation, plus one offline quantization image
and a complete GPU validation pack. Demonstration-grade by agreement: no GPU and no prod access this
cycle, so **every runtime number is deferred to runnable, pass/fail-criteria'd handoff scripts — none
are estimated.**

**What got newer.** vLLM 0.18.1 → **0.25.0** (gen + embedding) and **0.24.0 + vllm-omni 0.24.1**
(omni); torch 2.10.0 → **2.11.0** on the unchanged CUDA 13.0.x lineage; flashinfer 0.6.3.1 → 0.6.13;
transformers → 5.13.1 (serving). The load-bearing discovery: vLLM 0.25.0, 0.24.0, and even verl's
0.20.2 **all pin torch==2.11.0**, so one torch/CUDA anchor unifies serving, omni, and training — only
the vLLM package version forces the image split (vllm-omni couples to vLLM via Docker base only).

**What got faster — honestly.** Nothing is *measured* faster this cycle (no GPU). What ships is the
evidence-backed configuration to make it so: NVIDIA's pre-quantized NVFP4 Qwen3.6 (~3× weight-VRAM,
Hopper-servable via Marlin, integrity-certified below), FP8 KV cache on the one model where KV
scales with 262K context, compile/CUDA-graph defaults kept on, the multimodal processor cache kept
on, and an opt-in MTP speculative-decoding profile — plus the benchmark grid that decides it.

**What was verified by execution (not by reading):**
- **NVFP4 checkpoint integrity scan: PASS** — all 30,841 F8_E4M3 scale tensors (2.05 GB) clean of
  the modelopt-0.44 NaN-scale defect. The quantized profile's promotion gate is closed for the
  upstream copy.
- **All four image pin-graphs resolve** for linux/amd64 · py3.12 (cross-platform resolution; zero
  drift vs constraints; the vllm-omni commit co-resolves without moving the frozen vLLM core).
- ~40 orchestrator-executed source verifications (tags, wheels metadata, registry blobs, issue
  states, config defaults) — full trail in `docs/claim-ledger.md` (118 rows).

**What we deliberately did NOT do.** Force omni onto vLLM 0.25.0 (untested upstream); self-quantize
Qwen3.6 (official pre-quant exists; upstream export path marked "in progress"); quantize the
embedding model (tight cosine gate, no VRAM pressure); enable MTP by default (open prefix-caching
corruption on this exact model); adopt DFlash (open sm90 correctness bug, zero sm90 numbers,
unmerged vendor path); TRTLLM-GEN/EVS/forced-FlashInfer and 12 other levers (rejection log in
`docs/design-decisions.md`); bundle verl (its own image shares our torch anchor).

**What still needs a GPU / prod access.** Image builds + in-image checks (locally blocked only by
Docker VM disk — README §2 runs anywhere linux/amd64); per-profile smoke + degenerate-output scans;
the embedding/omni **feature-parity procedure against the production index** (hard gate, two-phase
scripts shipped); the composed MTP benchmark grid + 7 canary repros that decide spec-decode; the
FA3-on-cu13 inspection (its failure mode is already mitigated: FlashInfer's fp8-KV support is
code-verified). Risk register: `docs/risk-register.md`.

**Bottom line.** The modernization is designed, pinned, adversarially reviewed, and integrity-checked
end-to-end; the two things that could not be honest here — performance numbers and index parity — are
packaged as executable gates rather than claims.
