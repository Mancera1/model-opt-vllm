#!/usr/bin/env python3
"""Omni audio Call-type-A parity + degeneracy check (Qwen3-Omni-30B-A3B).

============================ BLOCKED PROTOCOL ============================
REQUIRES A RUNNING GPU-SERVED vLLM-OMNI ENDPOINT. No GPU this cycle -> runnable
handoff only. NEVER fabricate outputs; run against a real endpoint or not at all.

HARDWARE / ENDPOINT NEEDED
    1x NVIDIA Hopper (H100/H200, sm90). Qwen/Qwen3-Omni-30B-A3B-Instruct served
    via vLLM-Omni (separate project) with `--omni`. This mirrors production Call
    type A: raw audio in ONE inference call -> audio feature extraction (the omni
    model; VL-Embedding has no audio tower). The split-call design intentionally
    bypasses use_audio_in_video cross-modal alignment -- do NOT "fix" it.

        # BF16 baseline (reference) serve:
        vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8801
        # Candidate: FP8 thinker-only quant (quant-offline/scripts/
        #   quantize_omni_thinker_fp8.py export), served the same way.

EXACT INVOCATION (two phases: create reference on BF16, then gate the candidate)
    python3 omni_audio_check.py --base-url http://localhost:8801 \
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct --save-reference omni_ref.json
    python3 omni_audio_check.py --base-url http://localhost:8801 \
        --model <candidate-model-id> --reference omni_ref.json --logprobs

EXPECTED-OUTPUT TEMPLATE (healthy candidate)
    item audio-000  exact-match=YES  ratio=1.000  degenerate=NO
    item audio-001  exact-match=YES  ratio=1.000  degenerate=NO
    ...
    SUMMARY items=4 exact=4 degenerate=0 logprob_maxabs=3.1e-04 errors=0
    RESULT: PASS

WHAT THIS GATES vs WHAT IT DOES NOT
    Gates: the fixed-prompt greedy OUTPUT DIFF (must be 0 at temp=0) and the
    `!!!!`/loop degenerate scan (ruling D2/R5; scope brief section 2). It does
    NOT compute the <=1pt TASK-EVAL delta -- that is a separate gate: run your
    audio task-eval suite on BF16 vs the FP8 thinker and require delta <= 1 pt
    before promoting the off-vendor-validated FP8 recipe. Raw encoder features
    are not exposed by the OpenAI API; --logprobs gives the accessible numeric
    signal (top-logprob diff per position).
=========================================================================
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence

import numpy as np

# import the shared degenerate detector (sibling module)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degenerate_detector as dd  # noqa: E402

# ---------------------------------------------------------------------------
# PASS / FAIL criteria
# ---------------------------------------------------------------------------
OMNI_REQUIRE_EXACT_MATCH = True   # temp=0 greedy output must be byte-identical
OMNI_MIN_CHAR_SIMILARITY = 1.0    # informational difflib ratio (gate is exact)
OMNI_LOGPROB_ABS_TOL = 1e-3       # if --logprobs: max abs top-logprob delta

SAMPLE_RATE = 16000
AUDIO_SECONDS = 4.0
BASE_SEED = 20260712
MAX_TOKENS = 128
TEMPERATURE = 0.0
SEED = 12345

HTTP_RETRIES = 4
HTTP_BACKOFF_S = 1.5
HTTP_TIMEOUT_S = 180

_SSL_CTX: Optional[ssl.SSLContext] = None


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


def http_post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last_exc: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S,
                                        context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code in (400, 404, 422):
                raise last_exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
        if attempt < HTTP_RETRIES:
            time.sleep(HTTP_BACKOFF_S * attempt)
    raise RuntimeError(f"POST failed after {HTTP_RETRIES} tries: {url}: {last_exc}")


# ---------------------------------------------------------------------------
# Deterministic audio synthesis (seeded sine + noise -> 16-bit PCM WAV)
# ---------------------------------------------------------------------------
def synth_wav(index: int, freq_hz: float) -> bytes:
    rng = np.random.default_rng(BASE_SEED + 7919 * index)
    n = int(SAMPLE_RATE * AUDIO_SECONDS)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    # a couple of deterministic partials + seeded low-amplitude noise
    sig = (0.6 * np.sin(2 * np.pi * freq_hz * t)
           + 0.2 * np.sin(2 * np.pi * (freq_hz * 2.0) * t)
           + 0.05 * rng.standard_normal(n))
    sig = np.clip(sig, -1.0, 1.0)
    pcm = (sig * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class AudioItem:
    id: str
    prompt: str
    freq: float


AUDIO_CORPUS = [
    AudioItem("audio-000", "Describe the audio you hear in one short sentence.", 220.0),
    AudioItem("audio-001", "Is the sound a tone or speech? Answer briefly.", 440.0),
    AudioItem("audio-002", "Roughly how many distinct tones are present?", 330.0),
    AudioItem("audio-003", "Summarize the character of this audio clip.", 550.0),
]


def build_request(item: AudioItem, model: str, logprobs: bool) -> dict:
    wav = synth_wav(int(item.id.split("-")[1]), item.freq)
    b64 = base64.b64encode(wav).decode("ascii")
    content = [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": item.prompt},
    ]
    req = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "stream": False,
    }
    if logprobs:
        req["logprobs"] = True
        req["top_logprobs"] = 1
    return req


def _extract_text(resp: dict) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some servers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _extract_token_logprobs(resp: dict) -> List[float]:
    """Best-effort: first-choice per-token top logprob (OpenAI logprobs schema)."""
    try:
        lp = resp["choices"][0]["logprobs"]["content"]
        return [float(tok["logprob"]) for tok in lp]
    except Exception:
        return []


def _audio_sha(item: AudioItem) -> str:
    wav = synth_wav(int(item.id.split("-")[1]), item.freq)
    return hashlib.sha256(wav).hexdigest()[:16]


def run_endpoint(base_url: str, model: str, logprobs: bool
                 ) -> Dict[str, dict]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    out: Dict[str, dict] = {}
    for item in AUDIO_CORPUS:
        try:
            resp = http_post_json(url, build_request(item, model, logprobs))
            out[item.id] = {
                "prompt": item.prompt,
                "audio_sha": _audio_sha(item),
                "text": _extract_text(resp),
                "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
                "usage": resp.get("usage"),
                "token_logprobs": _extract_token_logprobs(resp) if logprobs else [],
                "raw": resp,
            }
            print(f"  ran {item.id}: {len(out[item.id]['text'])} chars "
                  f"finish={out[item.id]['finish_reason']}")
        except Exception as exc:
            out[item.id] = {"error": str(exc)}
            print(f"  ERROR {item.id}: {exc}", file=sys.stderr)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--logprobs", action="store_true",
                   help="capture per-token top logprobs for a numeric diff")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save-reference", metavar="OUT.json")
    g.add_argument("--reference", metavar="REF.json")
    p.add_argument("--out", help="write comparison JSON report here")
    p.add_argument("--insecure", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _SSL_CTX
    args = build_arg_parser().parse_args(argv)
    _SSL_CTX = _make_ssl_context(args.insecure)

    print(f"omni Call-type-A corpus: {len(AUDIO_CORPUS)} seeded WAV items "
          f"({SAMPLE_RATE} Hz, {AUDIO_SECONDS}s, temp={TEMPERATURE}, seed={SEED})")
    cur = run_endpoint(args.base_url, args.model, args.logprobs)
    errors = [k for k, v in cur.items() if "error" in v]

    if args.save_reference:
        with open(args.save_reference, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "results": cur}, fh, indent=2)
        ok = not errors
        print(f"saved reference: {args.save_reference} ({len(cur)} items, "
              f"{len(errors)} errors)")
        print(f"RESULT: {'PASS' if ok else 'FAIL'} (reference creation)")
        return 0 if ok else 1

    # --- comparison mode ---
    with open(args.reference, "r", encoding="utf-8") as fh:
        ref = json.load(fh).get("results", {})

    n_exact = 0
    n_degen = 0
    logprob_maxabs = 0.0
    audio_mismatch: List[str] = []
    per_item = {}
    for item in AUDIO_CORPUS:
        cur_v = cur.get(item.id, {})
        ref_v = ref.get(item.id, {})
        if "error" in cur_v or not ref_v:
            per_item[item.id] = {"error": cur_v.get("error", "missing reference")}
            continue
        if cur_v.get("audio_sha") != ref_v.get("audio_sha"):
            audio_mismatch.append(item.id)  # corpus drift => reference invalid
        ctext, rtext = cur_v.get("text", ""), ref_v.get("text", "")
        exact = (ctext == rtext)
        ratio = SequenceMatcher(None, rtext, ctext).ratio()
        res = dd.analyze(ctext)
        n_exact += exact
        n_degen += res.degenerate
        # logprob diff (align by min length; greedy temp0 should match length too)
        lp_delta = 0.0
        clp, rlp = cur_v.get("token_logprobs", []), ref_v.get("token_logprobs", [])
        if clp and rlp:
            m = min(len(clp), len(rlp))
            if m:
                lp_delta = max(abs(clp[i] - rlp[i]) for i in range(m))
                logprob_maxabs = max(logprob_maxabs, lp_delta)
        per_item[item.id] = {"exact": exact, "ratio": round(ratio, 4),
                             "degenerate": res.degenerate, "reasons": res.reasons,
                             "logprob_maxabs": lp_delta}
        print(f"item {item.id}  exact-match={'YES' if exact else 'NO':3s}  "
              f"ratio={ratio:.3f}  degenerate={'YES' if res.degenerate else 'NO'}")
        for r in res.reasons:
            print(f"      degenerate: {r}")

    exact_gate = (n_exact == len(AUDIO_CORPUS)) if OMNI_REQUIRE_EXACT_MATCH else True
    degen_gate = (n_degen == 0)
    logprob_gate = (not args.logprobs) or (logprob_maxabs <= OMNI_LOGPROB_ABS_TOL)
    passed = (exact_gate and degen_gate and logprob_gate
              and not errors and not audio_mismatch)

    print("=" * 64)
    print(f"SUMMARY items={len(AUDIO_CORPUS)} exact={n_exact} "
          f"degenerate={n_degen} "
          f"logprob_maxabs={logprob_maxabs:.2e} errors={len(errors)}")
    print(f"CRITERIA exact-match all (temp0 greedy 0-diff) AND degenerate==0 "
          f"AND (--logprobs => maxabs<={OMNI_LOGPROB_ABS_TOL})")
    if audio_mismatch:
        print(f"AUDIO SHA MISMATCH (reference corpus drift!): {audio_mismatch}")
    print(f"NOTE: <=1pt task-eval is a SEPARATE gate (see BLOCKED header).")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"gates": {"exact_match": OMNI_REQUIRE_EXACT_MATCH,
                                 "logprob_abs_tol": OMNI_LOGPROB_ABS_TOL},
                       "per_item": per_item, "n_exact": n_exact,
                       "n_degenerate": n_degen, "logprob_maxabs": logprob_maxabs,
                       "audio_mismatch": audio_mismatch, "errors": errors,
                       "passed": passed}, fh, indent=2)
        print(f"wrote report: {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
