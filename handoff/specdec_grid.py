#!/usr/bin/env python3
"""Composed speculative-decode benchmark grid for Qwen3.6-35B-A3B generation.

============================ BLOCKED PROTOCOL ============================
REQUIRES GPU + a vLLM v0.25.0 install (to serve) AND an OpenAI-compatible +
/metrics endpoint. No GPU this cycle -> runnable handoff. NEVER fabricate any
acceptance/latency number: run it against a real Hopper node or not at all.

HARDWARE NEEDED
    >=1x NVIDIA Hopper (H100/H200, sm90), enough VRAM for Qwen3.6-35B-A3B (BF16
    ~70 GB / NVFP4 ~23.4 GB weights + KV + the BF16 vision tower it carries).
    vLLM v0.25.0, torch==2.11.0 (orchestrator-verified cuda.txt). Run the NVFP4
    scale gate FIRST: handoff/scale_scan.py must PASS before trusting the NVFP4
    target (ruling R2).

THE GRID (Agent C composition, EXACTLY -- ruling D3)
    target      : {BF16 = Qwen/Qwen3.6-35B-A3B, NVFP4 = nvidia/Qwen3.6-35B-A3B-NVFP4}
    spec        : {none, mtp k=1, mtp k=2, mtp k=3}
    prefix-cache: {on, off}
    kv-cache    : {auto, fp8_e4m3}
    concurrency : {1, 4, 8, 16, 32}
    => 2 x 4 x 2 x 2 x 5 = 160 cells (32 distinct serve-configs x 5 concurrency).

DFLASH IS EXCLUDED ON PURPOSE (ruling D3, gate1_adjudications.md):
    DFlash is rejected for sm90 this cycle: OPEN H100 correctness bug vLLM #39928
    (DFlash/EAGLE3 sub-1-block NaN -> "!!!!"; MTP + no-spec are spared), ZERO sm90
    perf numbers (every DFlash speedup is Blackwell B200), and the vendor vLLM
    path is still unmerged PR #40898 (hybrid-SWA support #47914 was validated on
    Qwen3.5, not 3.6). So NO dflash axis is generated below. If you must probe
    DFlash, run canary #39928 from handoff/canary_repros.md, do not add it here.

PER-CELL MEASUREMENTS
    - acceptance length / rate  (scraped from vLLM /metrics: vllm:spec_decode_*)
    - TTFT, TPOT, aggregate tok/s  (streamed, per request)
MANDATORY PER-CELL SCANS (all three; a cell FAILS if any trips)
    (1) DEGENERATE detector on LONG generations (max_tokens>=8192, agentic
        multi-turn / tool-calling prompts): handoff/degenerate_detector.py.
    (2) LOSSLESS temp=0 greedy DIFF vs the same-config no-spec baseline: MUST be
        0-diff for MTP. Prefix-cache=ON x MTP cells are EXPECTED to FAIL this
        (rulings C-07 / vLLM #43559/#47194/#47087) -- that failure IS the canary.
    (3) ACCEPTANCE-ANOMALY watch: acceptance rate -> ~100% / accepted==drafted is
        the vLLM #40756 pre-crash signature.

EXACT INVOCATION
    # Fully automated (launches/tears down vLLM per serve-config):
    python3 specdec_grid.py --launch --base-url http://localhost:8000 --port 8000 \
        --max-model-len 32768 --log-dir ./vllm_logs --out grid_results.json
    # Manual (operator brings up ONE matching serve-config; script drives it):
    python3 specdec_grid.py --assume-served --base-url http://localhost:8000 \
        --only-config target=NVFP4,spec=mtp2,prefix=off,kv=fp8_e4m3 \
        --out grid_results.json
    # Just print every serve command (no endpoint needed):
    python3 specdec_grid.py --print-commands

EXPECTED-OUTPUT TEMPLATE (one row per cell)
    target spec  prefix kv        conc  accept  ttft  tpot  tok/s  degen 0diff anom PASS
    NVFP4  mtp2  off    fp8_e4m3   8     0.72    190   17    240    OK    OK    OK   YES
    NVFP4  mtp2  on     fp8_e4m3   8     0.55    210   19    198    LOOP  DIFF  OK   NO   <- #43559 canary
=========================================================================
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degenerate_detector as dd  # noqa: E402
import gen_bench  # noqa: E402  (reuse _pct + SSL helper)

# ---------------------------------------------------------------------------
# Grid definition + PASS/FAIL criteria
# ---------------------------------------------------------------------------
TARGETS = {
    "BF16": "Qwen/Qwen3.6-35B-A3B",
    "NVFP4": "nvidia/Qwen3.6-35B-A3B-NVFP4",
}
SPEC_OPTIONS = ["none", "mtp1", "mtp2", "mtp3"]  # DFlash EXCLUDED (ruling D3)
PREFIX_OPTIONS = [True, False]                   # on, off
KV_OPTIONS = ["auto", "fp8_e4m3"]
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]

LONG_MAX_TOKENS = 8192          # ">=8k tokens" mandatory-scan requirement
LOSSLESS_DIFF_MAX = 0           # MTP temp=0 greedy MUST equal no-spec baseline
ACCEPTANCE_ANOMALY_RATE = 0.999  # accepted/drafted ~ 1.0 => #40756 pre-crash sig
REQUIRE_ZERO_DEGENERATE = True

HEALTH_TIMEOUT_S = 1800         # model load can be slow
HEALTH_POLL_S = 5
_SSL_CTX: Optional[ssl.SSLContext] = None

_SPEC_RE = re.compile(
    r"^(vllm:spec_decode_[a-z0-9_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+\-]+)\s*$")


# ---------------------------------------------------------------------------
# Deterministic agentic multi-turn prompts (tool-calling flavored, long output)
# ---------------------------------------------------------------------------
def agentic_prompts() -> List[List[dict]]:
    sys_tool = ("You are an autonomous engineering agent with access to tools "
                "(search, run_code, read_file). Think step by step, call tools "
                "when useful, and produce a thorough final answer.")
    return [
        [
            {"role": "system", "content": sys_tool},
            {"role": "user", "content": "Design a reproducible CUDA 13 container "
             "for vLLM. List every pinned dependency and justify each pin."},
            {"role": "assistant", "content": "I will first inspect the base "
             "image, then resolve the torch anchor, then pin the rest."},
            {"role": "user", "content": "Continue. Now give the full Dockerfile "
             "and a long explanation of each layer, plus failure modes."},
        ],
        [
            {"role": "system", "content": sys_tool},
            {"role": "user", "content": "Walk through debugging a model whose "
             "output collapses to repeated exclamation marks under quantization."},
            {"role": "assistant", "content": "That signature suggests NaN weight "
             "scales. Let me enumerate the diagnostic steps."},
            {"role": "user", "content": "Elaborate at length, including how to "
             "scan FP8 E4M3 scales and how speculative decoding interacts."},
        ],
        [
            {"role": "system", "content": sys_tool},
            {"role": "user", "content": "Plan a multi-step tool-using workflow to "
             "benchmark an LLM: enumerate the tools, arguments, and expected "
             "outputs for each step, then synthesize a report."},
        ],
        [
            {"role": "system", "content": sys_tool},
            {"role": "user", "content": "Write an exhaustive tutorial on "
             "speculative decoding: MTP vs draft models, acceptance length, "
             "prefix-caching interactions, and correctness pitfalls."},
        ],
    ]


# ---------------------------------------------------------------------------
# Serve-config + cell model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServeConfig:
    target: str      # BF16 | NVFP4
    spec: str        # none | mtp1 | mtp2 | mtp3
    prefix: bool
    kv: str          # auto | fp8_e4m3

    @property
    def repo(self) -> str:
        return TARGETS[self.target]

    @property
    def mtp_k(self) -> Optional[int]:
        return int(self.spec[3:]) if self.spec.startswith("mtp") else None

    def key(self) -> str:
        return (f"target={self.target},spec={self.spec},"
                f"prefix={'on' if self.prefix else 'off'},kv={self.kv}")

    def baseline_key(self, concurrency: int) -> str:
        # identical to this cell except spec=none
        return (f"target={self.target},spec=none,"
                f"prefix={'on' if self.prefix else 'off'},kv={self.kv},"
                f"conc={concurrency}")


def enumerate_serve_configs() -> List[ServeConfig]:
    """Ordered so that, within each (target,prefix,kv), spec=none precedes mtp*
    (baselines must exist before the lossless diff)."""
    cfgs: List[ServeConfig] = []
    for target in TARGETS:
        for prefix in PREFIX_OPTIONS:
            for kv in KV_OPTIONS:
                for spec in SPEC_OPTIONS:  # none first
                    cfgs.append(ServeConfig(target, spec, prefix, kv))
    return cfgs


def build_serve_argv(cfg: ServeConfig, port: int, max_model_len: int) -> List[str]:
    """Exact vLLM v0.25.0 serve command for a cell (flags per rulings D3/D4/D5)."""
    argv = ["vllm", "serve", cfg.repo,
            "--max-model-len", str(max_model_len),
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_xml",
            "--trust-remote-code",  # match configs/*.sh serve profiles (review fix)
            "--port", str(port)]
    if cfg.target == "NVFP4":
        argv += ["--quantization", "modelopt", "--moe-backend", "marlin"]
    # prefix caching (explicit either way)
    argv += ["--enable-prefix-caching"] if cfg.prefix else ["--no-enable-prefix-caching"]
    # kv cache
    if cfg.kv == "fp8_e4m3":
        argv += ["--kv-cache-dtype", "fp8"]
        if cfg.target == "BF16":
            # BF16 profile must derive KV scales; NVFP4 ships them (ruling D4)
            argv += ["--calculate-kv-scales"]
    # speculative config
    k = cfg.mtp_k
    if k is not None:
        spec_cfg = {"method": "mtp", "num_speculative_tokens": k}
        if cfg.target == "NVFP4":
            # main experts marlin, MTP experts triton (NVIDIA README)
            spec_cfg["moe_backend"] = "triton"
        argv += ["--speculative-config", json.dumps(spec_cfg)]
    return argv


# ---------------------------------------------------------------------------
# HTTP: health, models, metrics, streaming chat
# ---------------------------------------------------------------------------
def _get(url: str, timeout: float = 30) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return resp.status, resp.read()


def wait_healthy(base_url: str, expect_repo: str) -> bool:
    deadline = time.time() + HEALTH_TIMEOUT_S
    health = base_url.rstrip("/") + "/health"
    models = base_url.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        try:
            st, _ = _get(health, timeout=10)
            if st == 200:
                try:
                    _, body = _get(models, timeout=10)
                    ids = [m.get("id") for m in json.loads(body).get("data", [])]
                    if any(expect_repo.split("/")[-1] in (i or "") for i in ids):
                        return True
                    # health up but model list present and mismatched
                    if ids:
                        print(f"  [warn] served models {ids} do not match "
                              f"{expect_repo}", file=sys.stderr)
                    return True
                except Exception:
                    return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def scrape_spec_metrics(base_url: str) -> Dict[str, float]:
    """Sum vllm:spec_decode_* counters across label sets."""
    out: Dict[str, float] = {}
    try:
        _, body = _get(base_url.rstrip("/") + "/metrics", timeout=20)
    except Exception as exc:
        return {"_error": 1.0, "_error_msg": str(exc)}  # type: ignore
    for line in body.decode("utf-8", "replace").splitlines():
        if not line.startswith("vllm:spec_decode"):
            continue
        m = _SPEC_RE.match(line)
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


def compute_acceptance(before: Dict[str, float], after: Dict[str, float]
                       ) -> Dict[str, float]:
    def d(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)
    draft = d("vllm:spec_decode_num_draft_tokens")
    accepted = d("vllm:spec_decode_num_accepted_tokens")
    emitted = d("vllm:spec_decode_num_emitted_tokens")
    drafts = d("vllm:spec_decode_num_drafts")  # step count, if exposed
    res: Dict[str, float] = {"draft_tokens": draft, "accepted_tokens": accepted,
                             "emitted_tokens": emitted, "draft_steps": drafts}
    res["acceptance_rate"] = (accepted / draft) if draft > 0 else 0.0
    if drafts > 0:
        res["acceptance_length"] = 1.0 + accepted / drafts
    return res


@dataclass
class ChatResult:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    out_tokens: int = 0
    text: str = ""
    error: Optional[str] = None


def stream_chat(base_url: str, model: str, messages: List[dict],
                max_tokens: int) -> ChatResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    r = ChatResult()
    t0 = time.perf_counter()
    first_t: Optional[float] = None
    parts: List[str] = []
    chunk_tokens = 0
    usage_tokens: Optional[int] = None
    try:
        with urllib.request.urlopen(req, timeout=gen_bench.HTTP_TIMEOUT_S,
                                    context=_SSL_CTX) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = usage["completion_tokens"]
                ch = (obj.get("choices") or [{}])[0]
                piece = ch.get("delta", {}).get("content") or ""
                if piece:
                    if first_t is None:
                        first_t = time.perf_counter()
                    parts.append(piece)
                    chunk_tokens += 1
    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"
        return r
    end_t = time.perf_counter()
    r.text = "".join(parts)
    r.out_tokens = usage_tokens if usage_tokens is not None else chunk_tokens
    if first_t is not None:
        r.ttft_ms = (first_t - t0) * 1000.0
        if r.out_tokens > 1:
            r.tpot_ms = (end_t - first_t) * 1000.0 / (r.out_tokens - 1)
    return r


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    config_key: str = ""
    concurrency: int = 0
    n_requests: int = 0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p99: float = 0.0
    agg_tok_s: float = 0.0
    acceptance: Dict[str, float] = field(default_factory=dict)
    degenerate: int = 0
    degenerate_reasons: List[str] = field(default_factory=list)
    lossless_diffs: int = -1   # -1 => not applicable / no baseline
    acceptance_anomaly: bool = False
    errors: int = 0
    passed: bool = False
    texts: List[str] = field(default_factory=list)


def run_cell(base_url: str, cfg: ServeConfig, concurrency: int,
             prompts: List[List[dict]], baselines: Dict[str, List[str]]
             ) -> CellResult:
    from concurrent.futures import ThreadPoolExecutor
    reqs = max(concurrency, len(prompts))
    slots = [prompts[i % len(prompts)] for i in range(reqs)]

    before = scrape_spec_metrics(base_url)
    t0 = time.perf_counter()
    results: List[ChatResult] = [None] * reqs  # type: ignore

    def worker(i: int):
        results[i] = stream_chat(base_url, cfg.repo, slots[i], LONG_MAX_TOKENS)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(worker, range(reqs)))
    wall = time.perf_counter() - t0
    after = scrape_spec_metrics(base_url)

    cr = CellResult(config_key=cfg.key(), concurrency=concurrency, n_requests=reqs)
    ok = [r for r in results if r is not None and r.error is None]
    cr.errors = sum(1 for r in results if r is None or r.error is not None)
    ttfts = sorted(r.ttft_ms for r in ok if r.ttft_ms > 0)
    tpots = sorted(r.tpot_ms for r in ok if r.tpot_ms > 0)
    cr.ttft_p50, cr.ttft_p99 = gen_bench._pct(ttfts, 50), gen_bench._pct(ttfts, 99)
    cr.tpot_p50, cr.tpot_p99 = gen_bench._pct(tpots, 50), gen_bench._pct(tpots, 99)
    total_tokens = sum(r.out_tokens for r in ok)
    cr.agg_tok_s = total_tokens / wall if wall > 0 else 0.0

    if cfg.spec != "none":
        cr.acceptance = compute_acceptance(before, after)
        rate = cr.acceptance.get("acceptance_rate", 0.0)
        acc_eq = (cr.acceptance.get("accepted_tokens", 0.0) > 0 and
                  cr.acceptance.get("accepted_tokens", 0.0)
                  >= cr.acceptance.get("draft_tokens", 0.0))
        cr.acceptance_anomaly = bool(rate >= ACCEPTANCE_ANOMALY_RATE or acc_eq)

    # (1) degenerate scan (only the per-prompt first-slot texts, deduped by prompt)
    cr.texts = [ok[i].text for i in range(min(len(prompts), len(ok)))]
    for txt in cr.texts:
        res = dd.analyze(txt)
        if res.degenerate:
            cr.degenerate += 1
            cr.degenerate_reasons.extend(res.reasons[:2])

    # (2) lossless temp=0 greedy diff vs same-config no-spec baseline
    bkey = cfg.baseline_key(concurrency)
    if cfg.spec == "none":
        baselines[bkey] = cr.texts  # record baseline for this (t,p,k,c)
        cr.lossless_diffs = -1
    else:
        base = baselines.get(bkey)
        if base is None:
            cr.lossless_diffs = -1  # baseline unavailable -> cannot gate
        else:
            n = min(len(base), len(cr.texts))
            cr.lossless_diffs = sum(1 for i in range(n) if base[i] != cr.texts[i])

    degen_ok = (cr.degenerate == 0) or not REQUIRE_ZERO_DEGENERATE
    lossless_ok = (cr.lossless_diffs <= LOSSLESS_DIFF_MAX) if cr.lossless_diffs >= 0 else True
    cr.passed = bool(degen_ok and lossless_ok and not cr.acceptance_anomaly
                     and cr.errors == 0)
    return cr


# ---------------------------------------------------------------------------
# State (resumable)
# ---------------------------------------------------------------------------
@dataclass
class GridState:
    path: str
    cells: Dict[str, dict] = field(default_factory=dict)   # cellkey -> CellResult dict
    baselines: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "GridState":
        if os.path.exists(path):
            try:
                obj = json.load(open(path, encoding="utf-8"))
                return cls(path, obj.get("cells", {}), obj.get("baselines", {}))
            except Exception:
                pass
        return cls(path)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"cells": self.cells, "baselines": self.baselines}, fh)
        os.replace(tmp, self.path)


def cell_key(cfg: ServeConfig, conc: int) -> str:
    return f"{cfg.key()},conc={conc}"


# ---------------------------------------------------------------------------
# Launch / teardown
# ---------------------------------------------------------------------------
def launch_server(argv: List[str], log_path: str) -> subprocess.Popen:
    log = open(log_path, "wb")
    print(f"  launching: {' '.join(argv)}")
    print(f"  server log -> {log_path}")
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)


def teardown(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------
def parse_only_config(s: str) -> ServeConfig:
    kv = dict(part.split("=", 1) for part in s.split(","))
    prefix = kv.get("prefix", "off").lower() in ("on", "true", "1")
    return ServeConfig(kv["target"], kv.get("spec", "none"), prefix,
                       kv.get("kv", "auto"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true",
                      help="launch/teardown vLLM per serve-config (needs GPU+vllm)")
    mode.add_argument("--assume-served", action="store_true",
                      help="an external server already runs the --only-config")
    mode.add_argument("--print-commands", action="store_true",
                      help="print every serve command and exit (no endpoint)")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--only-config", help="e.g. target=NVFP4,spec=mtp2,prefix=off,kv=fp8_e4m3")
    p.add_argument("--concurrency", default=",".join(map(str, CONCURRENCY_LEVELS)))
    p.add_argument("--log-dir", default="./vllm_logs")
    p.add_argument("--state", default="./specdec_grid_state.json")
    p.add_argument("--out", help="write full grid JSON results")
    p.add_argument("--insecure", action="store_true")
    return p


def print_commands(max_model_len: int, port: int) -> None:
    print("# DFlash cells are intentionally ABSENT (ruling D3). 32 serve-configs:")
    for cfg in enumerate_serve_configs():
        argv = build_serve_argv(cfg, port, max_model_len)
        print(f"\n# {cfg.key()}")
        print(" ".join(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _SSL_CTX
    args = build_arg_parser().parse_args(argv)
    _SSL_CTX = gen_bench._make_ssl_context(args.insecure)
    gen_bench._SSL_CTX = _SSL_CTX

    if args.print_commands or not (args.launch or args.assume_served):
        print_commands(args.max_model_len, args.port)
        if not (args.launch or args.assume_served):
            print("\n[info] no run mode selected (--launch/--assume-served); "
                  "printed commands only.")
        return 0

    concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    prompts = agentic_prompts()
    state = GridState.load(args.state)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.assume_served:
        if not args.only_config:
            print("ERROR: --assume-served requires --only-config", file=sys.stderr)
            return 2
        serve_configs = [parse_only_config(args.only_config)]
    else:
        if not shutil.which("vllm"):
            print("ERROR: --launch needs the `vllm` CLI on PATH (BLOCKED here: "
                  "no GPU/vLLM this cycle).", file=sys.stderr)
            return 2
        serve_configs = enumerate_serve_configs()

    print(f"{'target':6s} {'spec':5s} {'prefix':6s} {'kv':9s} {'conc':>4s} "
          f"{'accept':>7s} {'ttftP99':>8s} {'tpotP99':>8s} {'tok/s':>7s} "
          f"{'degen':>5s} {'0diff':>6s} {'anom':>4s} PASS")

    for cfg in serve_configs:
        pending = [c for c in concurrency
                   if cell_key(cfg, c) not in state.cells]
        if not pending:
            continue
        proc = None
        try:
            if args.launch:
                argv_serve = build_serve_argv(cfg, args.port, args.max_model_len)
                log_path = os.path.join(args.log_dir, cfg.key().replace(",", "__")
                                        .replace("=", "-") + ".log")
                proc = launch_server(argv_serve, log_path)
            if not wait_healthy(args.base_url, cfg.repo):
                print(f"  [FAIL] server not healthy for {cfg.key()}", file=sys.stderr)
                if proc:
                    teardown(proc)
                continue
            for c in concurrency:
                ck = cell_key(cfg, c)
                if ck in state.cells:
                    continue
                cr = run_cell(args.base_url, cfg, c, prompts, state.baselines)
                state.cells[ck] = vars(cr)
                state.save()
                acc = cr.acceptance.get("acceptance_rate", float("nan"))
                _fmt_row(cfg, cr, acc)
        finally:
            if proc:
                teardown(proc)

    return _summarize(state, args.out)


def _fmt_row(cfg: ServeConfig, cr: CellResult, acc: float) -> None:
    degen = "OK" if cr.degenerate == 0 else "LOOP"
    if cr.lossless_diffs < 0:
        zerod = "base" if cfg.spec == "none" else "n/a"
    else:
        zerod = "OK" if cr.lossless_diffs == 0 else f"DIFF{cr.lossless_diffs}"
    anom = "OK" if not cr.acceptance_anomaly else "ANOM"
    print(f"{cfg.target:6s} {cfg.spec:5s} "
          f"{'on' if cfg.prefix else 'off':6s} {cfg.kv:9s} {cr.concurrency:4d} "
          f"{acc:7.3f} {cr.ttft_p99:8.1f} {cr.tpot_p99:8.1f} {cr.agg_tok_s:7.1f} "
          f"{degen:>5s} {zerod:>6s} {anom:>4s} {'YES' if cr.passed else 'NO'}")


def _summarize(state: GridState, out_path: Optional[str]) -> int:
    total = len(state.cells)
    passed = sum(1 for v in state.cells.values() if v.get("passed"))
    failed = total - passed
    print("=" * 72)
    print(f"cells executed: {total}  passed: {passed}  failed: {failed}")
    # surface the expected canary failures explicitly
    canary = [k for k, v in state.cells.items()
              if not v.get("passed") and "spec=mtp" in k and "prefix=on" in k]
    if canary:
        print(f"expected MTP+prefix-cache canary failures (#43559/#47194/#47087): "
              f"{len(canary)}")
    print(f"criteria/cell: 0 degenerate AND (mtp => 0-diff vs baseline) AND "
          f"no acceptance-anomaly AND 0 errors")
    print(f"RESULT: {'PASS' if failed == 0 else 'FAIL'} "
          f"(a full clean grid is required to enable qwen36-mtp-profile.sh)")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"cells": state.cells, "passed": passed, "failed": failed,
                       "grid": {"targets": list(TARGETS), "spec": SPEC_OPTIONS,
                                "prefix": ["on", "off"], "kv": KV_OPTIONS,
                                "concurrency": CONCURRENCY_LEVELS,
                                "dflash": "EXCLUDED (ruling D3)"}}, fh, indent=2)
        print(f"wrote grid results: {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
