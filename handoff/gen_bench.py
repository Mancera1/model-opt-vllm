#!/usr/bin/env python3
"""Qwen3.6-35B-A3B generation benchmark: TTFT / TPOT / throughput.

============================ BLOCKED PROTOCOL ============================
REQUIRES A RUNNING GPU-SERVED ENDPOINT. No GPU this cycle -> runnable handoff.
NEVER fabricate latency/throughput numbers -- run against a real endpoint or not
at all (the project's worst failure mode is simulated numbers).

HARDWARE / ENDPOINT NEEDED
    1x NVIDIA Hopper (H100/H200, sm90). Qwen/Qwen3.6-35B-A3B (BF16) OR
    nvidia/Qwen3.6-35B-A3B-NVFP4 served on vLLM v0.25.0 with an OpenAI-compatible
    API. Example (NVFP4 default profile, prefix caching ON, MTP OFF -- the
    default per ruling D3):

        vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt --trust-remote-code \
          --moe-backend marlin --max-model-len 32768 --reasoning-parser qwen3 \
          --enable-prefix-caching --port 8000

    (For MTP / prefix-caching / kv / spec-decode SWEEPS use specdec_grid.py.)

EXACT INVOCATION
    python3 gen_bench.py --base-url http://localhost:8000 \
        --model nvidia/Qwen3.6-35B-A3B-NVFP4 --out gen_bench.json
    # optional latency SLO gates (no scope-contract latency number exists, so
    # these are OFF by default -- set them yourself, never invented here):
    #   --ttft-slo-ms 400 --tpot-slo-ms 25

EXPECTED-OUTPUT TEMPLATE (healthy)
    conc  reqs  ttft_ms(p50/p99)   tpot_ms(p50/p99)   tok/s(agg)  degen  err
       1     8   120.4 / 145.2       18.1 / 21.0          55.1       0     0
       4     8   180.7 / 260.9       20.3 / 27.4         190.2       0     0
      ...
    RESULT: PASS   (criteria: 0 request errors AND 0 degenerate generations
                    AND, if SLOs set, p99 within SLO)
=========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degenerate_detector as dd  # noqa: E402

# ---------------------------------------------------------------------------
# Config / PASS-FAIL criteria
# ---------------------------------------------------------------------------
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
PROMPT_COUNT = 16
PROMPT_SEED = 20260712
DEFAULT_MAX_TOKENS = 256
# Correctness gate (the scope contract has no latency SLO; latency SLOs are
# user-supplied via flags and default OFF -- we never invent them):
REQUIRE_ZERO_DEGENERATE = True
REQUIRE_ZERO_ERRORS = True

HTTP_TIMEOUT_S = 600
_SSL_CTX: Optional[ssl.SSLContext] = None

_WORDS = ("system latency throughput cache tensor kernel gradient cluster "
          "schedule pipeline memory bandwidth quantize attention router expert "
          "context window decode prefill benchmark parity checkpoint scale").split()


def _make_ssl_context(insecure: bool) -> Optional[ssl.SSLContext]:
    if insecure:
        print("[warn] TLS verification DISABLED (--insecure)", file=sys.stderr)
        return ssl._create_unverified_context()
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and os.path.exists(cafile):
        return ssl.create_default_context(cafile=cafile)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def build_prompts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        body = " ".join(rng.choice(_WORDS) for _ in range(40))
        prompts.append(
            f"[prompt {i}] Explain in detail, step by step, the following "
            f"engineering topic and its trade-offs: {body}. Provide a thorough "
            f"multi-paragraph answer.")
    return prompts


def _pct(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


@dataclass
class ReqResult:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    out_tokens: int = 0
    wall_ms: float = 0.0
    text: str = ""
    error: Optional[str] = None


def stream_one(base_url: str, model: str, api: str, prompt: str,
               max_tokens: int) -> ReqResult:
    """One streaming request; measures TTFT and TPOT. Uses stream usage for an
    exact completion-token count when the server provides it."""
    url = base_url.rstrip("/") + ("/v1/chat/completions" if api == "chat"
                                  else "/v1/completions")
    if api == "chat":
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}]}
    else:
        payload = {"model": model, "prompt": prompt}
    payload.update({"max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
                    "stream": True, "stream_options": {"include_usage": True}})
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")

    r = ReqResult()
    t0 = time.perf_counter()
    first_t: Optional[float] = None
    text_parts: List[str] = []
    chunk_tokens = 0
    usage_tokens: Optional[int] = None
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S,
                                    context=_SSL_CTX) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = usage["completion_tokens"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                piece = (ch.get("delta", {}).get("content") if api == "chat"
                         else ch.get("text")) or ""
                if piece:
                    if first_t is None:
                        first_t = time.perf_counter()
                    text_parts.append(piece)
                    chunk_tokens += 1
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"
        return r

    end_t = time.perf_counter()
    r.text = "".join(text_parts)
    r.out_tokens = usage_tokens if usage_tokens is not None else chunk_tokens
    r.wall_ms = (end_t - t0) * 1000.0
    if first_t is not None:
        r.ttft_ms = (first_t - t0) * 1000.0
        if r.out_tokens > 1:
            r.tpot_ms = (end_t - first_t) * 1000.0 / (r.out_tokens - 1)
    return r


@dataclass
class LevelStats:
    concurrency: int
    n: int
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p99: float = 0.0
    agg_tok_s: float = 0.0
    total_out_tokens: int = 0
    wall_s: float = 0.0
    degenerate: int = 0
    errors: int = 0
    reasons: List[str] = field(default_factory=list)


def run_level(base_url: str, model: str, api: str, concurrency: int,
              prompts: List[str], max_tokens: int, reqs: int) -> LevelStats:
    results: List[ReqResult] = [None] * reqs  # type: ignore
    lock = threading.Lock()
    counter = {"i": 0}

    def worker(slot: int):
        prompt = prompts[slot % len(prompts)]
        res = stream_one(base_url, model, api, prompt, max_tokens)
        with lock:
            results[slot] = res

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(worker, range(reqs)))
    wall_s = time.perf_counter() - t0

    ok = [r for r in results if r is not None and r.error is None and r.out_tokens > 0]
    errs = [r for r in results if r is None or r.error is not None]
    ttfts = sorted(r.ttft_ms for r in ok if r.ttft_ms > 0)
    tpots = sorted(r.tpot_ms for r in ok if r.tpot_ms > 0)
    total_tokens = sum(r.out_tokens for r in ok)

    st = LevelStats(concurrency=concurrency, n=reqs)
    st.ttft_p50, st.ttft_p99 = _pct(ttfts, 50), _pct(ttfts, 99)
    st.tpot_p50, st.tpot_p99 = _pct(tpots, 50), _pct(tpots, 99)
    st.total_out_tokens = total_tokens
    st.wall_s = wall_s
    st.agg_tok_s = total_tokens / wall_s if wall_s > 0 else 0.0
    st.errors = len(errs)
    for r in ok:
        res = dd.analyze(r.text)
        if res.degenerate:
            st.degenerate += 1
            st.reasons.extend(res.reasons[:2])
    return st


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api", choices=("completions", "chat"), default="completions")
    p.add_argument("--concurrency", default=",".join(map(str, CONCURRENCY_LEVELS)),
                   help="comma list (default 1,4,8,16,32)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--requests-per-level", type=int, default=0,
                   help="0 => auto = max(2*concurrency, 8)")
    p.add_argument("--ttft-slo-ms", type=float, default=None,
                   help="if set, gate p99 TTFT (else not gated)")
    p.add_argument("--tpot-slo-ms", type=float, default=None,
                   help="if set, gate p99 TPOT (else not gated)")
    p.add_argument("--out", help="write full JSON results here")
    p.add_argument("--insecure", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _SSL_CTX
    args = build_arg_parser().parse_args(argv)
    _SSL_CTX = _make_ssl_context(args.insecure)
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    prompts = build_prompts(PROMPT_COUNT, PROMPT_SEED)

    print(f"model={args.model} api={args.api} max_tokens={args.max_tokens} "
          f"prompts={len(prompts)} (seed={PROMPT_SEED})")
    print(f"{'conc':>4s} {'reqs':>4s} {'ttft_ms p50/p99':>18s} "
          f"{'tpot_ms p50/p99':>18s} {'tok/s':>8s} {'degen':>5s} {'err':>4s}")

    all_stats: List[LevelStats] = []
    for c in levels:
        reqs = args.requests_per_level or max(2 * c, 8)
        st = run_level(args.base_url, args.model, args.api, c, prompts,
                       args.max_tokens, reqs)
        all_stats.append(st)
        print(f"{st.concurrency:4d} {st.n:4d} "
              f"{st.ttft_p50:8.1f} /{st.ttft_p99:8.1f} "
              f"{st.tpot_p50:8.1f} /{st.tpot_p99:8.1f} "
              f"{st.agg_tok_s:8.1f} {st.degenerate:5d} {st.errors:4d}")

    total_degen = sum(s.degenerate for s in all_stats)
    total_err = sum(s.errors for s in all_stats)
    slo_ok = True
    slo_notes: List[str] = []
    for s in all_stats:
        if args.ttft_slo_ms is not None and s.ttft_p99 > args.ttft_slo_ms:
            slo_ok = False
            slo_notes.append(f"conc{s.concurrency} ttft_p99 {s.ttft_p99:.1f}>"
                             f"{args.ttft_slo_ms}")
        if args.tpot_slo_ms is not None and s.tpot_p99 > args.tpot_slo_ms:
            slo_ok = False
            slo_notes.append(f"conc{s.concurrency} tpot_p99 {s.tpot_p99:.1f}>"
                             f"{args.tpot_slo_ms}")

    passed = ((total_degen == 0 or not REQUIRE_ZERO_DEGENERATE)
              and (total_err == 0 or not REQUIRE_ZERO_ERRORS)
              and slo_ok)
    print("=" * 72)
    print(f"totals: degenerate={total_degen} errors={total_err}")
    print(f"criteria: 0 errors AND 0 degenerate"
          + (f" AND SLO(ttft_p99<={args.ttft_slo_ms},tpot_p99<={args.tpot_slo_ms})"
             if (args.ttft_slo_ms or args.tpot_slo_ms) else " (no latency SLO set)"))
    if slo_notes:
        print("SLO violations: " + "; ".join(slo_notes))
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "api": args.api,
                       "max_tokens": args.max_tokens,
                       "prompt_seed": PROMPT_SEED,
                       "levels": [vars(s) for s in all_stats],
                       "total_degenerate": total_degen, "total_errors": total_err,
                       "slo": {"ttft_ms": args.ttft_slo_ms,
                               "tpot_ms": args.tpot_slo_ms, "ok": slo_ok},
                       "passed": passed}, fh, indent=2)
        print(f"wrote results: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
