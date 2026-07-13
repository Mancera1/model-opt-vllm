#!/usr/bin/env python3
"""Full FP8-scale integrity scan for NVFP4 / FP8 ModelOpt checkpoints.

NO GPU REQUIRED. Runs anywhere with network access (streams safetensors shard
HEADERS and the individual scale tensors via HTTP range reads -- it never
downloads a whole weight shard). Can also scan a LOCAL export directory.

WHAT / WHY
----------
The NVIDIA pre-quant `nvidia/Qwen3.6-35B-A3B-NVFP4` was produced with
nvidia-modelopt 0.44.0 -- the version carrying the documented bug where the
float32 -> FP8 E4M3 cast of PER-BLOCK WEIGHT SCALES can emit literal NaN bytes
(E4M3 encoding 0x7F for +NaN, 0xFF for -NaN). A single NaN in any weight_scale
collapses the served model output to "!!!!" (see handoff/degenerate_detector.py,
handoff/canary_repros.md). vLLM's Marlin W4A16 path on Hopper consumes those
per-block E4M3 weight scales, so the bug bites sm90 too. Two independent PARTIAL
scans came back clean (17 tensors incl. the full lm_head scale); this script is
the FULL 30,841-tensor promotion gate (ruling R2, gate1_adjudications.md).

CRITICAL INSTRUMENT LESSON (baked in -- do not "fix" it back)
-------------------------------------------------------------
The bug class is *per-block E4M3 weight scales* only. Filter on
    dtype == "F8_E4M3"  AND  name endswith ".weight_scale"
The checkpoint also carries ~61,942 F32 tensors, including per-TENSOR F32 scales
on the FP8 attention layers. Those F32 scales are NOT the E4M3 NaN bug class;
scanning them for the E4M3 0x7F/0xFF byte pattern is an instrument error (the
orchestrator hit exactly this). Default --mode nvfp4 selects only E4M3
weight_scale. --mode fp8 / all cover FP8 self-quant exports (see below).

E4M3 (float8_e4m3fn) BYTE SEMANTICS for a magnitude scale
---------------------------------------------------------
    0x00 .. 0x7E  valid non-negative  (0x7E == 448.0, the E4M3 normal ceiling)
    0x7F          +NaN                 -> FAIL (the bug signature)
    0x80 .. 0xFE  negative             -> FAIL (illegitimate for a magnitude scale)
    0xFF          -NaN                 -> FAIL
So for a weight_scale, any byte >= 0x7F is bad. (For an FP8 *weight* value, only
0x7F/0xFF are bad; negatives are legitimate -- handled in --mode fp8.)

PASS / FAIL (numeric criteria)
------------------------------
    EXPECTED_SCALE_TENSORS = 30841   (default repo; enforced only for it)
PASS  == zero bad bytes across all selected scale tensors AND, for the default
         NVIDIA repo in nvfp4 mode, exactly 30,841 selected tensors were found.
FAIL  == any bad byte, or (default repo) a scale-tensor count mismatch.

USAGE
-----
    # Full promotion-gate scan of the NVIDIA pre-quant (remote, ~2 GB ranged reads):
    python3 scale_scan.py --repo nvidia/Qwen3.6-35B-A3B-NVFP4 --out scan_report.json

    # Quick smoke test (first 200 scale tensors):
    python3 scale_scan.py --repo nvidia/Qwen3.6-35B-A3B-NVFP4 --max-tensors 200

    # Scan a local FP8 self-quant export directory:
    python3 scale_scan.py --local ./omni-thinker-fp8-export --mode fp8

The scan is resumable: a state file (default ./scale_scan_state.<slug>.json) is
checkpointed as groups complete; re-running skips already-scanned tensors.

EXPECTED-OUTPUT TEMPLATE (default repo, healthy checkpoint)
-----------------------------------------------------------
    files: 3 safetensors shard(s)
    selected scale tensors (F8_E4M3 *.weight_scale): 30841   [expected 30841 OK]
    total scale bytes scanned: ~2052.9 MB
    bad tensors: 0   (+NaN 0x7F: 0, -NaN 0xFF: 0, negative 0x80-0xFE: 0)
    RESULT: PASS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as _np  # optional fast path
except Exception:  # pragma: no cover - numpy is optional
    _np = None

# Populated by main(); None => default system verification.
_SSL_CTX: Optional[ssl.SSLContext] = None


def _make_ssl_context(insecure: bool) -> Optional[ssl.SSLContext]:
    """Build a TLS context. Honors SSL_CERT_FILE / REQUESTS_CA_BUNDLE, falls
    back to certifi if installed, and supports --insecure for self-signed
    proxies / constrained CI (loudly, never silently)."""
    if insecure:
        print("[warn] TLS verification DISABLED (--insecure)", file=sys.stderr)
        return ssl._create_unverified_context()
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and os.path.exists(cafile):
        return ssl.create_default_context(cafile=cafile)
    try:
        import certifi  # optional
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None  # system default

# ---------------------------------------------------------------------------
# Constants / numeric pass criteria
# ---------------------------------------------------------------------------
DEFAULT_REPO = "nvidia/Qwen3.6-35B-A3B-NVFP4"
EXPECTED_SCALE_TENSORS = 30841       # PASS criterion for the default repo (nvfp4)
EXPECTED_TOTAL_SCALE_MB = 2052.9     # informational (Agent B header measurement)

HF_ENDPOINT = "https://huggingface.co"
E4M3_VALID_MAX = 0x7E                # 448.0
E4M3_POS_NAN = 0x7F
E4M3_NEG_NAN = 0xFF

HTTP_RETRIES = 4
HTTP_BACKOFF_S = 1.5
HTTP_TIMEOUT_S = 60
USER_AGENT = "scale-scan/1.0 (+handoff; no-gpu; ranged-reads)"

# Coalesce adjacent scale tensors into one ranged read to cut request count.
COALESCE_MAX_GAP = 1 << 20           # merge if inter-tensor gap <= 1 MiB
COALESCE_MAX_SPAN = 64 << 20         # never let a merged read exceed 64 MiB
CHECKPOINT_EVERY_GROUPS = 25         # persist state every N merged groups

_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# HTTP (stdlib urllib; follows HF resolve -> CDN redirect, preserves Range)
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    return _SAFE_SLUG.sub("_", s).strip("_")


def _hf_token() -> Optional[str]:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v
    return None


def _request(url: str, headers: Dict[str, str]) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    for k, v in headers.items():
        req.add_header(k, v)
    tok = _hf_token()
    if tok and url.startswith(HF_ENDPOINT):
        # Only send auth to huggingface.co itself; urllib strips it on the
        # cross-host CDN redirect (the CDN URL is pre-signed). Fine for public
        # repos; required only for gated ones.
        req.add_header("Authorization", f"Bearer {tok}")
    return req


def http_get(url: str, *, expect_range: Optional[Tuple[int, int]] = None,
             max_full_bytes: int = 32 << 20) -> bytes:
    """GET url, optionally with a byte Range. Retries transient failures.

    If a Range was requested but the server answers 200 (Range ignored) with a
    body larger than max_full_bytes, aborts rather than pulling a whole shard.
    """
    headers: Dict[str, str] = {}
    if expect_range is not None:
        start, end = expect_range
        headers["Range"] = f"bytes={start}-{end}"
    last_exc: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            req = _request(url, headers)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S,
                                        context=_SSL_CTX) as resp:
                status = resp.status
                if expect_range is not None and status != 206:
                    clen = resp.headers.get("Content-Length")
                    if status == 200 and clen and int(clen) > max_full_bytes:
                        raise RuntimeError(
                            f"server ignored Range (200, {clen} bytes) on {url} "
                            f"-- refusing to download a full shard")
                data = resp.read()
                if expect_range is not None:
                    want = expect_range[1] - expect_range[0] + 1
                    if status == 200 and len(data) > want:
                        data = data[expect_range[0]:expect_range[1] + 1]
                return data
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, ConnectionError, RuntimeError) as exc:
            last_exc = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF_S * attempt)
            continue
    raise RuntimeError(f"GET failed after {HTTP_RETRIES} tries: {url}: {last_exc}")


def hf_resolve_url(repo: str, filename: str, revision: str) -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{filename}"


# ---------------------------------------------------------------------------
# Source abstraction: remote HF repo OR local directory
# ---------------------------------------------------------------------------
@dataclass
class Source:
    remote: bool
    repo: Optional[str] = None
    revision: str = "main"
    local_dir: Optional[str] = None

    def slug(self) -> str:
        return _slug(self.repo if self.remote else os.path.abspath(self.local_dir))

    def read_bytes(self, filename: str, start: int, end: int) -> bytes:
        """Inclusive byte range [start, end]."""
        if self.remote:
            url = hf_resolve_url(self.repo, filename, self.revision)
            return http_get(url, expect_range=(start, end),
                            max_full_bytes=COALESCE_MAX_SPAN + (1 << 20))
        path = os.path.join(self.local_dir, filename)
        with open(path, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)


def list_safetensors(src: Source) -> List[str]:
    """Enumerate *.safetensors shard filenames (excluding *.index.json)."""
    if not src.remote:
        files = [f for f in sorted(os.listdir(src.local_dir))
                 if f.endswith(".safetensors")]
        if not files:
            raise RuntimeError(f"no *.safetensors in {src.local_dir}")
        return files

    # Primary: HF tree API (paginated defensively).
    files: List[str] = []
    cursor_url = (f"{HF_ENDPOINT}/api/models/{src.repo}/tree/"
                  f"{src.revision}?recursive=1")
    seen_pages = 0
    while cursor_url and seen_pages < 50:
        try:
            raw = http_get(cursor_url)
        except Exception:
            break
        seen_pages += 1
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            break
        for e in entries:
            if isinstance(e, dict) and e.get("type") == "file":
                p = e.get("path", "")
                if p.endswith(".safetensors"):
                    files.append(p)
        # HF pagination is via a Link header; urllib exposes it only if we kept
        # the response. For our 3-shard models one page suffices, so stop here.
        cursor_url = None
    if files:
        return sorted(set(files))

    # Fallback: model.safetensors.index.json weight_map unique shard names.
    try:
        idx = http_get(hf_resolve_url(src.repo, "model.safetensors.index.json",
                                      src.revision))
        obj = json.loads(idx)
        wm = obj.get("weight_map", {})
        shards = sorted(set(wm.values()))
        if shards:
            return shards
    except Exception:
        pass

    # Last resort: single-file checkpoint.
    return ["model.safetensors"]


# ---------------------------------------------------------------------------
# safetensors header parsing
# ---------------------------------------------------------------------------
@dataclass
class TensorRef:
    filename: str
    name: str
    dtype: str
    abs_start: int   # absolute byte offset in the file
    abs_end: int     # inclusive
    check_kind: str  # E4M3_SCALE | E4M3_VALUE | F32_SCALE

    @property
    def nbytes(self) -> int:
        return self.abs_end - self.abs_start + 1

    def key(self) -> str:
        return f"{self.filename}::{self.name}"


def read_header(src: Source, filename: str) -> Tuple[dict, int]:
    """Return (header_json, data_section_start). 8-byte LE u64 len + JSON."""
    head8 = src.read_bytes(filename, 0, 7)
    if len(head8) < 8:
        raise RuntimeError(f"{filename}: short header length read")
    n = struct.unpack("<Q", head8)[0]
    if n <= 0 or n > (512 << 20):
        raise RuntimeError(f"{filename}: implausible header length {n}")
    hdr_bytes = src.read_bytes(filename, 8, 8 + n - 1)
    header = json.loads(hdr_bytes.decode("utf-8"))
    return header, 8 + n


def _classify(name: str, dtype: str, mode: str) -> Optional[str]:
    """Return a check_kind for this tensor under `mode`, or None to skip."""
    is_weight_scale = name.endswith(".weight_scale")
    is_any_scale = name.endswith("_scale") or name.endswith(".scale")
    if mode == "nvfp4":
        if dtype == "F8_E4M3" and is_weight_scale:
            return "E4M3_SCALE"
        return None
    if mode == "fp8":
        if dtype == "F8_E4M3" and name.endswith(".weight"):
            return "E4M3_VALUE"     # FP8 weight values (NaN-only check)
        if dtype in ("F32", "F64") and is_any_scale:
            return "F32_SCALE"      # per-tensor FP8 scales (float NaN/Inf check)
        return None
    if mode == "all":
        if dtype == "F8_E4M3":
            return "E4M3_SCALE" if is_weight_scale else "E4M3_VALUE"
        if dtype in ("F32", "F64") and is_any_scale:
            return "F32_SCALE"
        return None
    raise ValueError(f"unknown mode {mode!r}")


def select_tensors(src: Source, filename: str, header: dict, data_start: int,
                   mode: str) -> List[TensorRef]:
    refs: List[TensorRef] = []
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        dtype = meta.get("dtype", "")
        offs = meta.get("data_offsets")
        if not offs or len(offs) != 2:
            continue
        kind = _classify(name, dtype, mode)
        if kind is None:
            continue
        start = data_start + int(offs[0])
        end = data_start + int(offs[1]) - 1  # inclusive
        if end < start:
            continue
        refs.append(TensorRef(filename, name, dtype, start, end, kind))
    return refs


# ---------------------------------------------------------------------------
# Byte scanning
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    pos_nan: int = 0        # 0x7F
    neg_nan: int = 0        # 0xFF
    negative: int = 0       # 0x80..0xFE (E4M3 scale only)
    f32_nan: int = 0
    f32_inf: int = 0
    f32_nonpos: int = 0     # <=0 F32 scale (suspect, not fatal)

    @property
    def fatal(self) -> int:
        return self.pos_nan + self.neg_nan + self.negative + self.f32_nan + self.f32_inf

    def add(self, other: "Finding") -> None:
        self.pos_nan += other.pos_nan
        self.neg_nan += other.neg_nan
        self.negative += other.negative
        self.f32_nan += other.f32_nan
        self.f32_inf += other.f32_inf
        self.f32_nonpos += other.f32_nonpos


# translate table: bytes >= 0x7F -> 1, else 0 (for the pure-Python fast path)
_HIGH_TABLE = bytes(0 if b < 0x7F else 1 for b in range(256))


def _count_ge_7f(buf: bytes) -> int:
    if _np is not None:
        a = _np.frombuffer(buf, dtype=_np.uint8)
        return int((a >= 0x7F).sum())
    return len(buf) - buf.translate(_HIGH_TABLE).count(0)


def scan_e4m3_scale(buf: bytes) -> Finding:
    """Magnitude scale: valid 0x00..0x7E; any byte >= 0x7F is bad."""
    f = Finding()
    total_bad = _count_ge_7f(buf)
    if total_bad == 0:
        return f
    f.pos_nan = buf.count(0x7F)
    f.neg_nan = buf.count(0xFF)
    f.negative = total_bad - f.pos_nan - f.neg_nan
    return f


def scan_e4m3_value(buf: bytes) -> Finding:
    """Signed FP8 weight value: only 0x7F / 0xFF (NaN) are bad."""
    f = Finding()
    f.pos_nan = buf.count(0x7F)
    f.neg_nan = buf.count(0xFF)
    return f


def scan_f32_scale(buf: bytes) -> Finding:
    """Per-tensor F32 scale: NaN / Inf are fatal; <=0 is suspect."""
    f = Finding()
    n = len(buf) // 4
    if n == 0:
        return f
    if _np is not None:
        a = _np.frombuffer(buf[:n * 4], dtype="<f4")
        f.f32_nan = int(_np.isnan(a).sum())
        f.f32_inf = int(_np.isinf(a).sum())
        finite = a[_np.isfinite(a)]
        f.f32_nonpos = int((finite <= 0).sum())
        return f
    for (v,) in struct.iter_unpack("<f", buf[:n * 4]):
        if v != v:
            f.f32_nan += 1
        elif v in (float("inf"), float("-inf")):
            f.f32_inf += 1
        elif v <= 0:
            f.f32_nonpos += 1
    return f


_SCANNERS = {
    "E4M3_SCALE": scan_e4m3_scale,
    "E4M3_VALUE": scan_e4m3_value,
    "F32_SCALE": scan_f32_scale,
}


# ---------------------------------------------------------------------------
# Coalescing + orchestration
# ---------------------------------------------------------------------------
def coalesce(refs: List[TensorRef], max_gap: int, max_span: int
             ) -> List[List[TensorRef]]:
    """Group contiguous/nearby tensors (same file) into merged read spans."""
    groups: List[List[TensorRef]] = []
    by_file: Dict[str, List[TensorRef]] = {}
    for r in refs:
        by_file.setdefault(r.filename, []).append(r)
    for fname in sorted(by_file):
        items = sorted(by_file[fname], key=lambda r: r.abs_start)
        cur: List[TensorRef] = []
        span_start = span_end = 0
        for r in items:
            if not cur:
                cur = [r]
                span_start, span_end = r.abs_start, r.abs_end
                continue
            gap = r.abs_start - span_end - 1
            new_span = r.abs_end - span_start + 1
            if gap <= max_gap and new_span <= max_span:
                cur.append(r)
                span_end = r.abs_end
            else:
                groups.append(cur)
                cur = [r]
                span_start, span_end = r.abs_start, r.abs_end
        if cur:
            groups.append(cur)
    return groups


@dataclass
class ScanState:
    path: str
    source_slug: str
    mode: str
    scanned: Dict[str, dict] = field(default_factory=dict)  # key -> result
    done: bool = False

    @classmethod
    def load(cls, path: str, source_slug: str, mode: str) -> "ScanState":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if (obj.get("source_slug") == source_slug
                        and obj.get("mode") == mode):
                    return cls(path, source_slug, mode,
                               scanned=obj.get("scanned", {}),
                               done=obj.get("done", False))
            except Exception:
                pass
        return cls(path, source_slug, mode)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"source_slug": self.source_slug, "mode": self.mode,
                       "scanned": self.scanned, "done": self.done}, fh)
        os.replace(tmp, self.path)


def _fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def run_scan(src: Source, mode: str, state: ScanState, *,
             max_tensors: Optional[int], coalesce_gap: int, coalesce_span: int,
             progress_every_s: float = 3.0) -> Tuple[List[TensorRef], Finding, Dict[str, dict]]:
    files = list_safetensors(src)
    print(f"files: {len(files)} safetensors shard(s): {', '.join(files)}")

    all_refs: List[TensorRef] = []
    for fname in files:
        header, data_start = read_header(src, fname)
        refs = select_tensors(src, fname, header, data_start, mode)
        all_refs.extend(refs)
        print(f"  {fname}: {len(header) - ('__metadata__' in header)} tensors, "
              f"{len(refs)} selected ({mode})")

    all_refs.sort(key=lambda r: (r.filename, r.abs_start))
    if max_tensors is not None:
        all_refs = all_refs[:max_tensors]

    total_sel = len(all_refs)
    total_bytes = sum(r.nbytes for r in all_refs)
    print(f"selected scale tensors: {total_sel}  "
          f"(~{_fmt_mb(total_bytes)} to scan)")

    pending = [r for r in all_refs if r.key() not in state.scanned]
    if len(pending) != total_sel:
        print(f"resuming: {total_sel - len(pending)} already scanned, "
              f"{len(pending)} pending")

    groups = coalesce(pending, coalesce_gap, coalesce_span)
    agg = Finding()
    # fold already-scanned results into the aggregate
    for res in state.scanned.values():
        f = Finding(**{k: res.get(k, 0) for k in Finding().__dict__})
        agg.add(f)

    scanned_count = total_sel - len(pending)
    bytes_done = sum(r.nbytes for r in all_refs if r.key() in state.scanned)
    last_print = 0.0
    for gi, group in enumerate(groups):
        g_start = group[0].abs_start
        g_end = group[-1].abs_end
        fname = group[0].filename
        buf = src.read_bytes(fname, g_start, g_end)
        for r in group:
            sub = buf[r.abs_start - g_start: r.abs_end - g_start + 1]
            finding = _SCANNERS[r.check_kind](sub)
            state.scanned[r.key()] = {
                "dtype": r.dtype, "kind": r.check_kind, "nbytes": r.nbytes,
                **finding.__dict__,
            }
            agg.add(finding)
            scanned_count += 1
            bytes_done += r.nbytes
            if finding.fatal:
                print(f"  !! BAD {r.key()} [{r.check_kind}] "
                      f"+NaN={finding.pos_nan} -NaN={finding.neg_nan} "
                      f"neg={finding.negative} f32nan={finding.f32_nan} "
                      f"f32inf={finding.f32_inf}")
        now = time.time()
        if (gi + 1) % CHECKPOINT_EVERY_GROUPS == 0:
            state.save()
        if now - last_print >= progress_every_s:
            last_print = now
            print(f"  progress: {scanned_count}/{total_sel} tensors  "
                  f"{_fmt_mb(bytes_done)}  fatal-so-far={agg.fatal}")
    state.save()
    return all_refs, agg, state.scanned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full FP8-scale integrity scan (NaN weight-scale gate). "
                    "No GPU; ranged HTTP reads or local files.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--repo", help="HF repo id (remote)")
    grp.add_argument("--local", help="local export directory")
    p.add_argument("--revision", default="main", help="HF revision (remote)")
    p.add_argument("--mode", choices=("nvfp4", "fp8", "all"), default="nvfp4",
                   help="nvfp4: E4M3 *.weight_scale (the 0.44 bug class, default); "
                        "fp8: E4M3 *.weight + F32 *_scale; all: both")
    p.add_argument("--out", help="write full JSON report to this path")
    p.add_argument("--state", help="resumable state file "
                   "(default ./scale_scan_state.<slug>.json)")
    p.add_argument("--max-tensors", type=int, default=None,
                   help="scan only the first N selected tensors (smoke test)")
    p.add_argument("--coalesce-gap", type=int, default=COALESCE_MAX_GAP)
    p.add_argument("--coalesce-span", type=int, default=COALESCE_MAX_SPAN)
    p.add_argument("--no-coalesce", action="store_true",
                   help="one HTTP range read per tensor (slower, exact)")
    p.add_argument("--insecure", action="store_true",
                   help="disable TLS verification (self-signed proxy / CI only)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _SSL_CTX
    args = build_arg_parser().parse_args(argv)
    _SSL_CTX = _make_ssl_context(args.insecure)
    if not args.repo and not args.local:
        args.repo = DEFAULT_REPO
        print(f"[info] no --repo/--local given; defaulting to {DEFAULT_REPO}")

    if args.local:
        src = Source(remote=False, local_dir=args.local)
    else:
        src = Source(remote=True, repo=args.repo, revision=args.revision)

    coalesce_gap = 0 if args.no_coalesce else args.coalesce_gap
    coalesce_span = 0 if args.no_coalesce else args.coalesce_span
    state_path = args.state or f"./scale_scan_state.{src.slug()}.json"
    state = ScanState.load(state_path, src.slug(), args.mode)

    t0 = time.time()
    try:
        refs, agg, results = run_scan(
            src, args.mode, state,
            max_tensors=args.max_tensors,
            coalesce_gap=coalesce_gap, coalesce_span=coalesce_span)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    state.done = True
    state.save()

    total_sel = len(refs)
    total_bytes = sum(r.nbytes for r in refs)
    bad_tensors = sum(1 for r in refs
                      if Finding(**{k: results[r.key()].get(k, 0)
                                    for k in Finding().__dict__}).fatal)

    # count-expectation check (only meaningful for the default repo in nvfp4 mode)
    is_default = (src.remote and src.repo == DEFAULT_REPO and args.mode == "nvfp4"
                  and args.max_tensors is None)
    count_ok = True
    count_note = ""
    if is_default:
        count_ok = (total_sel == EXPECTED_SCALE_TENSORS)
        count_note = (f"  [expected {EXPECTED_SCALE_TENSORS} "
                      f"{'OK' if count_ok else 'MISMATCH'}]")

    passed = (agg.fatal == 0) and count_ok

    print("=" * 64)
    print(f"source: {'repo ' + src.repo if src.remote else 'local ' + src.local_dir}"
          f"  mode={args.mode}  revision={src.revision if src.remote else '-'}")
    print(f"selected scale tensors: {total_sel}{count_note}")
    print(f"total scale bytes scanned: {_fmt_mb(total_bytes)}")
    print(f"bad tensors: {bad_tensors}")
    print(f"  +NaN(0x7F)={agg.pos_nan}  -NaN(0xFF)={agg.neg_nan}  "
          f"negative(0x80-0xFE)={agg.negative}")
    print(f"  f32_nan={agg.f32_nan}  f32_inf={agg.f32_inf}  "
          f"f32_nonpositive(suspect)={agg.f32_nonpos}")
    print(f"criteria: fatal-bytes==0 AND "
          f"(default-repo => count=={EXPECTED_SCALE_TENSORS})")
    print(f"elapsed: {time.time() - t0:.1f}s   state: {state_path}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    if args.out:
        report = {
            "source": src.repo if src.remote else os.path.abspath(src.local_dir),
            "remote": src.remote, "revision": src.revision, "mode": args.mode,
            "selected_tensors": total_sel, "total_bytes": total_bytes,
            "expected_tensors": EXPECTED_SCALE_TENSORS if is_default else None,
            "count_ok": count_ok,
            "aggregate": agg.__dict__, "bad_tensors": bad_tensors,
            "passed": passed,
            "bad_detail": {k: v for k, v in results.items()
                           if Finding(**{kk: v.get(kk, 0)
                                         for kk in Finding().__dict__}).fatal},
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote report: {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
