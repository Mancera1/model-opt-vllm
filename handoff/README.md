# GPU Handoff Pack — Qwen3.6 / Omni / VL-Embedding validation & spec-decode gate

This directory is the **GPU handoff** for the inference-container modernization.
It is the runnable form of every runtime check the no-GPU Gate-1 research could
not execute. It pairs with the offline quantization jobs in
`../quant-offline/scripts/`.

Authoritative decisions live in `gate1_adjudications.md` (binding rulings).
Model/version facts trace to `agentB.json` (quantization) and `agentC.json`
(spec-decode). This README is operational glue, not a source of truth.

---

## 0. BLOCKED protocol (read first — non-negotiable)

**There is no GPU in this cycle.** Every script that talks to a served model or
runs vLLM is marked **BLOCKED** in its own docstring: it is complete and runnable,
but it must be executed on real NVIDIA **Hopper (H100/H200, sm90)** hardware.

- **No simulated numbers, ever.** Do not hand-write, estimate, or "fill in" any
  latency, throughput, acceptance, cosine, or eval number. Run the script against
  real hardware or leave that cell blank. Fabricated numbers are this project's
  worst failure mode (brief §4).
- **Blackwell numbers do not transfer to sm90.** Any B200/GB200/GB300 figure in
  the research (DFlash speedups, NVFP4 accuracy table) is Blackwell-pedigree and
  must be **re-measured** on Hopper — do not port it.
- **Public sources / no secrets.** These scripts contain no credentials. Only
  `HF_TOKEN` (optional, for gated HF repos) and TLS bundle env vars are read.

Two scripts need **neither GPU nor a served endpoint** and were validated locally
this session: `scale_scan.py` and `degenerate_detector.py`.

---

## 1. What's in the pack

| # | file | needs | one-liner |
|---|------|-------|-----------|
| 1 | `scale_scan.py` | network only | Full FP8/NVFP4 scale-integrity scan (NaN weight-scale gate). |
| 2 | `degenerate_detector.py` | nothing | `!!!!`-collapse / repeated-token / n-gram-loop detector (module + CLI). |
| 3 | `parity_embedding.py` | endpoint | Embedding cosine parity for Qwen3-VL-Embedding-8B (pooling runner). |
| 4 | `omni_audio_check.py` | endpoint | Omni Call-type-A audio parity + degeneracy diff. |
| 5 | `gen_bench.py` | endpoint | Qwen3.6 TTFT/TPOT/throughput at concurrency {1,4,8,16,32}. |
| 6 | `specdec_grid.py` | GPU + vllm | Composed MTP grid: target×spec×prefix×kv×concurrency (DFlash excluded, D3). |
| 7 | `capacity_probe.py` | GPU + docker | Bisects a host's ceiling for `--max-num-batched-tokens` or `--gpu-memory-utilization` per model (boot + load-burst per probe). Informs per-host overrides; never edits the profiles. |
| 8 | `canary_repros.md` | endpoint/GPU | The 7 open upstream issues that gate MTP, each with a repro. |
| 9 | `README.md` | — | this file. |
| — | `../quant-offline/scripts/quantize_omni_thinker_fp8.py` | GPU | FP8 W8A8 thinker-only PTQ (off-vendor path, eval-gated). |
| — | `../quant-offline/scripts/quantize_qwen36_fallback.py` | GPU | FALLBACK Qwen3.6 self-quant (primary = NVIDIA pre-quant). |

`scale_scan.py` and `degenerate_detector.py` are pure standard library. The
endpoint drivers need `numpy` (and optionally `PIL`; a stdlib PNG encoder is used
if `PIL` is absent). The quant jobs need the IMAGE-3 offline env
(`torch==2.11.0`, `nvidia-modelopt[hf]==0.45.0`, `transformers` in [4.56, 5.10)).

---

## 2. Environment variables

| var | used by | purpose |
|-----|---------|---------|
| `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) | `scale_scan.py` | optional; only for gated HF repos. Sent to `huggingface.co` only; stripped on the CDN redirect (correct). |
| `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` | all networked scripts | optional CA bundle for TLS behind a proxy. `--insecure` disables verification (loudly). |
| `SSL_CERT_FILE` unset + `certifi` installed | all networked scripts | `certifi` bundle is used automatically. |

No other environment configuration is required. Endpoints, models, and file paths
are passed as explicit `--flags`.

---

## 3. Run order

Everything except steps 1 is BLOCKED (needs GPU/endpoint).

1. **Scale gate (no GPU, do this first).** The NVFP4 target was produced by
   modelopt 0.44.0 (the NaN-scale-bug version); it is trusted only after a clean
   FULL scan (ruling R2). Two partial scans were already clean; this is the
   30,841-tensor gate.
   ```
   python3 scale_scan.py --repo nvidia/Qwen3.6-35B-A3B-NVFP4 --out scan_report.json
   # must print: RESULT: PASS   (bad tensors: 0, count 30841)
   ```
2. **(Optional) offline quant jobs (GPU).** Only if you are NOT using the NVIDIA
   pre-quant (Qwen3.6) or you are quantizing the omni thinker. Each job
   auto-reruns `scale_scan.py` on its export:
   ```
   python3 ../quant-offline/scripts/quantize_omni_thinker_fp8.py --output ./omni-fp8 ...
   python3 ../quant-offline/scripts/quantize_qwen36_fallback.py --format fp8 --output ./q36-fp8 ...
   ```
3. **Parity gates (GPU endpoints).** Create the reference on the CURRENT/prod
   image FIRST, then gate the candidate image:
   ```
   # embeddings (Qwen3-VL-Embedding-8B, pooling runner)
   python3 parity_embedding.py --base-url $EMB --model Qwen/Qwen3-VL-Embedding-8B --save-reference emb_ref.jsonl   # on prod
   python3 parity_embedding.py --base-url $EMB --model Qwen/Qwen3-VL-Embedding-8B --reference emb_ref.jsonl         # on candidate
   # omni audio (Qwen3-Omni-30B-A3B-Instruct, vLLM-Omni)
   python3 omni_audio_check.py --base-url $OMNI --model Qwen/Qwen3-Omni-30B-A3B-Instruct --save-reference omni_ref.json   # on BF16
   python3 omni_audio_check.py --base-url $OMNI --model <candidate> --reference omni_ref.json --logprobs                 # on FP8 candidate
   ```
4. **Baseline generation perf (GPU endpoint).**
   ```
   python3 gen_bench.py --base-url $GEN --model nvidia/Qwen3.6-35B-A3B-NVFP4 --out gen_bench.json
   ```
5. **Composed spec-decode grid + canaries (GPU + vllm).** This is the spec-decode
   go/no-go input:
   ```
   python3 specdec_grid.py --launch --base-url http://localhost:8000 --port 8000 \
       --max-model-len 32768 --log-dir ./vllm_logs --out grid_results.json
   # then work through canary_repros.md on the intended profile.
   ```
6. **Enable MTP profile — only if step 5 is clean (see §5).**

---

## 4. Pass-criteria table (constants embedded in each script)

| script | PASS criteria (constants) |
|--------|---------------------------|
| `scale_scan.py` | 0 bad bytes (any byte ≥ `0x7F`) across all `F8_E4M3 *.weight_scale` tensors **and** `count == 30841` for the default NVFP4 repo (`--mode nvfp4`). `--mode fp8`: E4M3 weight NaN (`0x7F/0xFF`) + F32 scale NaN/Inf = 0. |
| `degenerate_detector.py` | 0 degenerate generations. Thresholds: `MAX_CHAR_RUN=25`, `MAX_TOKEN_RUN=10`, `NGRAM_N=3` consecutive `>5`, ngram fraction `≥0.50`, dominant non-alnum char `≥0.50`. |
| `parity_embedding.py` | text min cosine `≥ 0.999`; multimodal (image+video) min cosine `≥ 0.995`; overall p99 coverage (1st-percentile) cosine `≥ 0.995`. |
| `omni_audio_check.py` | every item exact greedy 0-diff vs reference; 0 degenerate; `--logprobs` top-logprob max abs diff `≤ 1e-3`. **Separate gate:** audio task-eval delta `≤ 1 pt`. |
| `gen_bench.py` | 0 request errors **and** 0 degenerate generations. Optional latency SLOs (`--ttft-slo-ms`/`--tpot-slo-ms`) OFF by default — the scope contract sets no latency number, so none is invented. |
| `specdec_grid.py` | per cell: 0 degenerate **and** (MTP ⇒ 0-diff vs same-config no-spec baseline, `LOSSLESS_DIFF_MAX=0`) **and** no acceptance anomaly (`rate < 0.999`, `accepted < drafted`) **and** 0 errors. Whole grid must be clean to enable MTP. |
| `capacity_probe.py` | a probe PASSes iff the server reaches `/health` **and** survives the concurrent load burst (all 200s, container alive, still healthy). Reports the highest passing value; exit 1 only if the `--lo` floor fails. **A per-host tuner, not a promotion gate** — record the ceiling, set deployment overrides deliberately, leave `configs/*` at vLLM defaults. |
| `canary_repros.md` | none of #43559/#47194/#47087/#35288/#40756/#47930/#39928 reproduces on the intended profile. |
| `quantize_omni_thinker_fp8.py` | export produces safetensors **and** auto `scale_scan.py --mode fp8` == PASS. Then REQUIRED: omni `≤ 1 pt` task-eval (separate, serving). |
| `quantize_qwen36_fallback.py` | export produces safetensors **and** auto `scale_scan.py` (mode per format) == PASS. `w4a16_nvfp4`: also confirm it loads in vLLM (deploy support "in progress", B-22). |

All quality tolerances trace to the scope contract (brief §2): embedding cosine
`≥ 0.999` text / `≥ 0.995` multimodal, `p99 ≥ 0.995`; omni fixed-prompt diff +
task-eval `≤ 1 pt`; degenerate-output detector on every quantized/spec-decode run.

---

## 5. The rule that gates `qwen36-mtp-profile.sh`

`qwen36-mtp-profile.sh` is the **opt-in MTP serve profile** (ruling D3:
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` **with
`--no-enable-prefix-caching` mandatory**), produced with the serving configs — it
is NOT part of this handoff pack. The **default** profile is the opposite:
**prefix caching ON, MTP OFF**.

**Enable `qwen36-mtp-profile.sh` ONLY IF, on real sm90 hardware:**
1. `specdec_grid.py` is **fully clean** at the profile's intended prefix-caching
   setting (every executed cell `PASS=YES`: 0 degenerate, 0-diff lossless vs
   baseline, no acceptance anomaly), **AND**
2. **none** of the 7 canaries in `canary_repros.md` reproduces on that profile
   (especially #43559/#47087/#35288/#40756), **AND**
3. the net TPOT gain over the no-spec CUDA-graph baseline exceeds your threshold
   at the production concurrency (MTP can be net-slower — #47277/#39680).

If prefix caching is **mandatory** for your traffic and cannot be made clean with
MTP on, the honest outcome is **spec-decode NEITHER** for now (ruling D3): keep
the default profile (prefix caching ON, MTP OFF). DFlash is not an escape hatch —
it is excluded for sm90 (open H100 bug #39928, no sm90 perf numbers, unmerged PR
#40898); see `canary_repros.md`.

---

## 6. Provenance

Gate-1 research: 4 Opus agents (max effort), 2026-07-12, re-verified against
primary sources (raw.githubusercontent at tags, PyPI JSON, HF raw/API, GitHub API,
ranged safetensors reads). vLLM HEAD at lock: **v0.25.0 (2026-07-11)**; torch
anchor **2.11.0/cu13**; nvidia-modelopt **0.45.0**; NVFP4 producer **0.44.0**.
This pack was authored under those rulings; it invents no new facts and no numbers.
