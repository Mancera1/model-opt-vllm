# Canary Repros — 7 upstream vLLM issues that gate spec-decode on Qwen3.6-35B-A3B

**Status of all 7: OPEN as of 2026-07-12** (issue states verified in Gate-1
research against `api.github.com`; see `gate1_adjudications.md` §4 R3/R11 and the
Agent-C ledger). None is confirmed fixed at vLLM **v0.25.0**.

**No GPU this cycle → every repro is a BLOCKED handoff.** Never record a
simulated outcome. Each canary is run on a real Hopper (H100/H200, sm90) node and
its result is one input to the go/no-go on enabling MTP (see `README.md`).

Common setup used below:
- Targets: `BF16 = Qwen/Qwen3.6-35B-A3B`, `NVFP4 = nvidia/Qwen3.6-35B-A3B-NVFP4`
  (run `scale_scan.py` first — the NVFP4 target must PASS the scale gate).
- Detector: `python3 handoff/degenerate_detector.py` (repeated-token / n-gram
  loop / `!!!!` collapse).
- Grid driver: `handoff/specdec_grid.py` already encodes cells that reproduce
  #43559/#47194/#47087/#35288/#40756 as part of its mandatory per-cell scans;
  the invocations below isolate one issue each.
- `MTP2 = --speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
  (add `,"moe_backend":"triton"` for the NVFP4 target).

DFlash (#47930, #39928) is documented here as the **reason DFlash is excluded**
(ruling D3); DFlash cells are NOT in `specdec_grid.py`.

---

## #43559 — prefix-caching + MTP accuracy collapse (mamba/GDN state-cache reuse)
- **One line:** With `--enable-prefix-caching` AND MTP, Qwen3.6-35B-A3B (and 27B)
  drops ~20% accuracy; tool-calling falls from ~90% → ~50% (reporter timothysu,
  tool-eval-bench). Root cause (maintainer zack041): mamba/GDN state-cache
  boundary reuse under speculative decoding. 5 fix PRs still OPEN, Blackwell-only.
- **Gates:** any profile that runs **MTP with prefix caching ON** (i.e. the
  default profile with MTP added). This is the single biggest reason the default
  profile keeps **MTP OFF while prefix caching is ON**.
- **Repro:**
  ```
  # A) prefix-cache ON + MTP  (suspected-bad)
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt --moe-backend marlin \
    --enable-prefix-caching --max-model-len 32768 --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' --port 8000
  # run your tool-calling eval (tool-eval-bench or equivalent), record accuracy.
  # B) control: same server WITHOUT --speculative-config (MTP off) OR with
  #    --no-enable-prefix-caching (prefix off). Record accuracy.
  ```
  Or one command that captures the greedy divergence directly:
  ```
  python3 handoff/specdec_grid.py --assume-served --base-url http://localhost:8000 \
    --only-config target=NVFP4,spec=mtp2,prefix=on,kv=fp8_e4m3
  ```
- **Expected if PRESENT (current):** config-B accuracy ~90%, config-A ~50%
  (~20-40 pt drop); `specdec_grid` reports `0diff=DIFF` (greedy output diverges
  from the no-spec baseline) for the `prefix=on` MTP cell.
- **Expected if FIXED:** A and B accuracy within noise; `0diff=OK` (byte-identical
  greedy output vs the no-spec baseline).

## #47194 — prefix-caching + MTP(k=3): tool-call leakage + needle-recall failure
- **One line:** Qwen3.6 hybrid + prefix-cache + `num_speculative_tokens=3` leaks
  tool-call tokens and fails long-context needle retrieval, while the no-MTP path
  is correct. OPEN.
- **Gates:** MTP profiles with **k≥3 AND prefix caching ON**; long-context /
  agentic tool-use traffic specifically.
- **Repro:**
  ```
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt --moe-backend marlin \
    --enable-prefix-caching --max-model-len 262144 --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' --port 8000
  # 1) send an agentic multi-turn tool-calling conversation; inspect for tool-call
  #    token leakage into user-visible content.
  # 2) place a "needle" fact early in a >100k-token context and ask for it.
  python3 handoff/specdec_grid.py --assume-served --base-url http://localhost:8000 \
    --only-config target=NVFP4,spec=mtp3,prefix=on,kv=fp8_e4m3
  ```
- **Expected if PRESENT:** tool-call markup appears in assistant content; needle
  fact is missed; `specdec_grid` flags `0diff=DIFF` and/or the degenerate detector
  trips on leaked control tokens.
- **Expected if FIXED:** clean tool-call parsing, needle recalled, `0diff=OK`.

## #47087 — native MTP garbage token loops on deep agentic conversations
- **One line:** On deep agentic conversations, native MTP degenerates into garbage
  token loops on Qwen3.6-35B-A3B — reported on **both BF16 and FP8**. OPEN.
- **Gates:** MTP on long agentic/multi-turn traffic (the `specdec_grid` long
  (`max_tokens>=8192`) agentic prompt set exists to catch this).
- **Repro:**
  ```
  # BF16 target, MTP, deep multi-turn agentic prompts, long generations:
  vllm serve Qwen/Qwen3.6-35B-A3B --enable-prefix-caching --max-model-len 32768 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --port 8000
  python3 handoff/specdec_grid.py --assume-served --base-url http://localhost:8000 \
    --only-config target=BF16,spec=mtp2,prefix=on,kv=auto
  # (specdec_grid runs the agentic prompt set at >=8k tokens and pipes each
  #  generation through degenerate_detector.py automatically.)
  ```
- **Expected if PRESENT:** the degenerate detector reports `LOOP`
  (`token-run`/`ngram-loop`) on one or more long generations → cell `PASS=NO`.
- **Expected if FIXED:** coherent long-form output; degenerate detector reports 0.

## #35288 — MTP corrupted output at concurrency ≥ 4 (V1 engine)
- **One line:** MTP produces corrupted output once concurrency reaches ≥ 4 on the
  V1 engine (reported cu129/py3.12). OPEN.
- **Gates:** any MTP profile at production concurrency ≥ 4 (i.e. essentially all
  real traffic).
- **Repro:**
  ```
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt --moe-backend marlin \
    --no-enable-prefix-caching --max-model-len 32768 --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' --port 8000
  # compare greedy output at concurrency 1 vs 4/8/16/32 (prefix caching OFF to
  # isolate from #43559):
  python3 handoff/specdec_grid.py --assume-served --base-url http://localhost:8000 \
    --only-config target=NVFP4,spec=mtp2,prefix=off,kv=fp8_e4m3
  ```
- **Expected if PRESENT:** cells at concurrency ≥ 4 show `0diff=DIFF` vs the
  concurrency-matched no-spec baseline and/or `degen=LOOP`, while concurrency 1 is
  clean.
- **Expected if FIXED:** `0diff=OK` at every concurrency.

## #40756 — MTP + FP8 target: illegal memory access on long sequences (accept→100%)
- **One line:** MTP with an FP8 Qwen3.6 target crashes with illegal memory access
  on long sequences; the pre-crash signature is **acceptance rate → 100% /
  accepted == drafted**, then a crash after ~26k tokens (reported at v0.19.1,
  TP=4). OPEN; re-verify at v0.25.0.
- **Gates:** MTP on the FP8-attention NVFP4 target (or any FP8 target) with long
  generations / large `--max-model-len`.
- **Repro:**
  ```
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt --moe-backend marlin \
    --kv-cache-dtype fp8 --no-enable-prefix-caching --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' --port 8000
  # drive >30k-token generations and WATCH the acceptance metric:
  python3 handoff/specdec_grid.py --assume-served --base-url http://localhost:8000 \
    --only-config target=NVFP4,spec=mtp2,prefix=off,kv=fp8_e4m3
  # (specdec_grid scrapes vllm:spec_decode_* and flags acceptance_anomaly when
  #  acceptance rate >= 0.999 / accepted==drafted.)
  ```
- **Expected if PRESENT:** `anom=ANOM` (acceptance → ~100%) shortly before the
  server crashes / OOMs; server log shows an illegal memory access.
- **Expected if FIXED:** acceptance stays in a normal band (well below 1.0); no
  crash across long generations.

## #47930 — DFlash/DSpark draft acceptance collapses with automatic prefix caching
- **One line:** DFlash (and DSpark) draft acceptance collapses when automatic
  prefix caching is enabled (reported cu130/torch2.11). OPEN.
- **Gates:** any DFlash profile with automatic prefix caching. **DFlash is
  EXCLUDED from our plan (ruling D3)** — this canary documents that DFlash does
  not escape the prefix-cache hazard either.
- **Repro (only if you evaluate DFlash despite D3):**
  ```
  vllm serve Qwen/Qwen3.6-35B-A3B --enable-prefix-caching --max-model-len 32768 \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":2}' \
    --port 8000
  # measure draft acceptance from vllm:spec_decode_* with prefix caching ON vs OFF.
  ```
- **Expected if PRESENT:** acceptance rate near 0 with prefix caching ON, normal
  with it OFF.
- **Expected if FIXED:** acceptance unaffected by prefix caching.

## #39928 — DFlash/EAGLE3 sub-1-block NaN → "!!!!" on sm90 (H100)
- **One line:** On H100 (sm90), DFlash and EAGLE3 give run-to-run-inconsistent,
  sometimes NaN → `"!!!!"` output on short/sub-1-block prompts; **MTP and no-spec
  are NOT affected** (reporter benchislett: "reproduce when using EAGLE3 as well
  as DFlash. But not MTP or no-spec"); `--enforce-eager` mitigates. OPEN. This is
  **the primary sm90 correctness reason DFlash is rejected (ruling D3).**
- **Gates:** DFlash/EAGLE3 on sm90 (the reason they are not shipped). MTP is the
  clean alternative here.
- **Repro (only if you evaluate DFlash despite D3):**
  ```
  vllm serve Qwen/Qwen3.6-35B-A3B \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":2}' \
    --port 8000
  # send a SHORT prompt (~16 tokens) at temperature 0, 100 times; check run-to-run
  # consistency and collapse:
  #   for i in $(seq 100); do curl -s .../v1/completions -d '{"model":"...","prompt":"Hello, list three colors:","max_tokens":16,"temperature":0}' ; done
  #   ... | python3 handoff/degenerate_detector.py --field text
  # Cross-check that MTP is clean on the SAME prompt (swap method to "mtp").
  ```
- **Expected if PRESENT:** run-to-run-inconsistent outputs at temp 0; some runs
  produce NaN/`"!!!!"` (degenerate detector: `DEGENERATE`); MTP on the same prompt
  stays consistent and clean.
- **Expected if FIXED:** DFlash deterministic at temp 0 and no collapse — but note
  even a fixed #39928 does not clear the other DFlash blockers (no sm90 perf
  numbers; unmerged vendor PR #40898; #47930).

---

### How these feed the go/no-go
A canary that reproduces (PRESENT) on a given config **blocks** enabling MTP for
that config. The honest posture (ruling D3): the default profile is **prefix
caching ON, MTP OFF**; MTP is an opt-in profile enabled **only** if
`specdec_grid.py` is clean at the intended prefix-caching setting AND #43559,
#47087, #35288, #40756 do not reproduce AND the temp-0 greedy diff is 0. If prefix
caching is mandatory and cannot be made clean, spec-decode is **NEITHER** for now.
