#!/usr/bin/env python3
"""Embedding parity harness for Qwen3-VL-Embedding-8B (pooling runner).

============================ BLOCKED PROTOCOL ============================
REQUIRES A RUNNING GPU-SERVED ENDPOINT. There is no GPU in this cycle, so this
script cannot be exercised end-to-end here; it is a runnable handoff. NEVER
fabricate cosine numbers -- run it against a real endpoint or not at all.

HARDWARE / ENDPOINT NEEDED
    1x NVIDIA Hopper (H100/H200, sm90). Qwen/Qwen3-VL-Embedding-8B served on the
    vLLM POOLING runner with the parity-pinned preprocessing (brief S7 / ruling
    D5). Feature-parity vs the prod precomputed index is a HARD gate; the pins
    below are load-bearing configuration, not tunables:

        vllm serve Qwen/Qwen3-VL-Embedding-8B \
          --runner pooling \
          --pooler-config '{"pooling_type":"LAST","normalize":true}' \
          --mm-processor-kwargs '{"fps":2.0,"max_frames":64}' \
          --port 8000

EXACT INVOCATION (two phases)
    # 1) On the CURRENT/prod serving image, create the reference:
    python3 parity_embedding.py --base-url http://localhost:8000 \
        --model Qwen/Qwen3-VL-Embedding-8B --save-reference ref_embeddings.jsonl
    # 2) On the CANDIDATE image (new vLLM / quant), gate against it:
    python3 parity_embedding.py --base-url http://localhost:8000 \
        --model Qwen/Qwen3-VL-Embedding-8B --reference ref_embeddings.jsonl

EXPECTED-OUTPUT TEMPLATE (healthy)
    bucket   n   min_cos   mean_cos   gate     result
    text     12  0.99997   0.99999    >=0.999  PASS
    image     6  0.99912   0.99960    >=0.995  PASS
    video     4  0.99801   0.99880    >=0.995  PASS
    overall  22  p1=0.99801  p99=0.99999  gate p1>=0.995  PASS
    RESULT: PASS

PARITY CAVEAT (human amendment, gate1_adjudications.md section 0)
    Variant B (single vllm 0.24.0 image) ships NO torchcodec/PyNvVideoCodec
    (OpenCV-only video IO) vs Variant A/0.25.0 (torchcodec + PyNvVideoCodec).
    Production Call-type-B sends CLIENT-EXTRACTED frames, so this harness sends
    frames directly (default --video-mode frames-datauri-list), which is off the
    server-side video-file decode path. If you ever benchmark real video FILES,
    the two images can diverge -- document it, do not paper over it.
=========================================================================

NO audio here (VL-Embedding has no audio tower -- Call-type-A goes to the omni
model; see handoff/omni_audio_check.py).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image as _PILImage  # preferred encoder if present
except Exception:
    _PILImage = None

# ---------------------------------------------------------------------------
# PASS / FAIL gates (scope contract, brief section 2)
# ---------------------------------------------------------------------------
TEXT_COSINE_MIN = 0.999         # min cosine over text items
MULTIMODAL_COSINE_MIN = 0.995   # min cosine over image + video items
P99_COVERAGE_COSINE_MIN = 0.995  # 1st-percentile cosine (>=99% of items above)

IMG_H = 224
IMG_W = 224
VIDEO_FRAMES = 8
BASE_SEED = 20260712

HTTP_RETRIES = 4
HTTP_BACKOFF_S = 1.5
HTTP_TIMEOUT_S = 120

_SSL_CTX: Optional[ssl.SSLContext] = None


# ---------------------------------------------------------------------------
# TLS + HTTP
# ---------------------------------------------------------------------------
def _make_ssl_context(insecure: bool) -> Optional[ssl.SSLContext]:
    if insecure:
        print("[warn] TLS verification DISABLED (--insecure)", file=sys.stderr)
        return ssl._create_unverified_context()
    import os
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
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S,
                                        context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code in (400, 404, 422):
                raise last_exc  # request-shape error: don't retry blindly
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
        if attempt < HTTP_RETRIES:
            time.sleep(HTTP_BACKOFF_S * attempt)
    raise RuntimeError(f"POST failed after {HTTP_RETRIES} tries: {url}: {last_exc}")


# ---------------------------------------------------------------------------
# Deterministic media synthesis (numpy-seeded; PIL optional, stdlib PNG fallback)
# ---------------------------------------------------------------------------
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(arr: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to PNG. Uses PIL if available, else a
    dependency-free encoder (valid 8-bit RGB PNG)."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if _PILImage is not None:
        import io
        buf = io.BytesIO()
        _PILImage.fromarray(arr, "RGB").save(buf, format="PNG")
        return buf.getvalue()
    h, w = arr.shape[:2]
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (None)
        raw.extend(arr[y].tobytes())
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colortype 2=RGB
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _png_chunk(b"IEND", b""))


def _data_uri_png(arr: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(arr)).decode("ascii")


def synth_image(index: int) -> np.ndarray:
    """Deterministic structured image (gradient + colored block), not pure noise
    (pure noise embeds ambiguously)."""
    rng = np.random.default_rng(BASE_SEED + 101 * index)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    ang = (index % 6) * (np.pi / 6)
    grad = (np.cos(ang) * xx / IMG_W + np.sin(ang) * yy / IMG_H)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    img = np.zeros((IMG_H, IMG_W, 3), np.float32)
    for c in range(3):
        img[..., c] = grad * (0.4 + 0.2 * ((index + c) % 3))
    # a deterministic colored rectangle
    y0, x0 = rng.integers(10, IMG_H // 2), rng.integers(10, IMG_W // 2)
    color = rng.integers(60, 255, size=3)
    img[y0:y0 + 60, x0:x0 + 60, :] = color / 255.0
    # low-amplitude seeded texture
    img += 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def synth_video_frames(index: int, n_frames: int = VIDEO_FRAMES) -> List[np.ndarray]:
    """Deterministic frame sequence with a moving block (temporal signal so the
    interleaved-MRoPE / timestamp path is exercised)."""
    rng = np.random.default_rng(BASE_SEED + 997 * index)
    base = synth_image(index)
    frames = []
    bh, bw = 40, 40
    color = rng.integers(80, 255, size=3)
    for t in range(n_frames):
        f = base.astype(np.float32).copy()
        y = int((IMG_H - bh) * t / max(1, n_frames - 1))
        x = int((IMG_W - bw) * (1 - t / max(1, n_frames - 1)))
        f[y:y + bh, x:x + bw, :] = color
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


# Fixed deterministic text corpus.
TEXT_CORPUS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Quarterly revenue grew by twelve percent driven by cloud services.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose.",
    "The committee will reconvene on Thursday to finalize the budget.",
    "A distributed database replicates writes across multiple availability zones.",
    "She tuned the guitar carefully before the evening performance began.",
    "Gradient descent minimizes the loss by following the negative gradient.",
    "The recipe calls for two cups of flour and a pinch of sea salt.",
    "Satellites in low earth orbit complete a revolution in about ninety minutes.",
    "Please summarize the attached report and highlight the key risks.",
    "The museum's new exhibit features artifacts from the bronze age.",
    "Container images should be reproducible and pinned to exact versions.",
]


@dataclass
class CorpusItem:
    id: str
    kind: str  # text | image | video


def build_corpus() -> List[CorpusItem]:
    items: List[CorpusItem] = []
    for i in range(len(TEXT_CORPUS)):
        items.append(CorpusItem(f"text-{i:03d}", "text"))
    for i in range(6):
        items.append(CorpusItem(f"image-{i:03d}", "image"))
    for i in range(4):
        items.append(CorpusItem(f"video-{i:03d}", "video"))
    return items


# ---------------------------------------------------------------------------
# Request construction  (SINGLE place to adjust for your pooling server schema)
# ---------------------------------------------------------------------------
def build_request(item: CorpusItem, model: str, video_mode: str) -> dict:
    """vLLM embeddings request. Text -> EmbeddingCompletionRequest (`input`);
    image/video -> EmbeddingChatRequest (`messages` with multimodal content).
    If your deployed pooling server expects a different multimodal shape, adjust
    ONLY this function."""
    if item.kind == "text":
        idx = int(item.id.split("-")[1])
        return {"model": model, "input": [TEXT_CORPUS[idx]],
                "encoding_format": "float"}

    if item.kind == "image":
        idx = int(item.id.split("-")[1])
        uri = _data_uri_png(synth_image(idx))
        content = [{"type": "image_url", "image_url": {"url": uri}}]
        return {"model": model,
                "messages": [{"role": "user", "content": content}],
                "encoding_format": "float"}

    if item.kind == "video":
        idx = int(item.id.split("-")[1])
        frames = synth_video_frames(idx)
        frame_uris = [_data_uri_png(f) for f in frames]
        if video_mode == "frames-image-list":
            # Each frame as an image part (approximation; may not trigger the
            # temporal MRoPE path -- documented in the BLOCKED header).
            content = [{"type": "image_url", "image_url": {"url": u}}
                       for u in frame_uris]
        else:  # "frames-datauri-list" (default): frames-as-video, client-extracted
            content = [{"type": "video_url",
                        "video_url": {"url": frame_uris}}]
        return {"model": model,
                "messages": [{"role": "user", "content": content}],
                "encoding_format": "float"}

    raise ValueError(f"unknown kind {item.kind!r}")


def embed_one(base_url: str, path: str, item: CorpusItem, model: str,
              video_mode: str) -> np.ndarray:
    url = base_url.rstrip("/") + path
    resp = http_post_json(url, build_request(item, model, video_mode))
    data = resp.get("data")
    if not data:
        raise RuntimeError(f"{item.id}: no 'data' in response: "
                           f"{json.dumps(resp)[:300]}")
    emb = data[0].get("embedding")
    if emb is None:
        raise RuntimeError(f"{item.id}: no 'embedding' in response")
    return np.asarray(emb, dtype=np.float64)


# ---------------------------------------------------------------------------
# Cosine + gating
# ---------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_reference(path: str) -> Dict[str, np.ndarray]:
    ref: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref[obj["id"]] = np.asarray(obj["embedding"], dtype=np.float64)
    return ref


def gather_embeddings(base_url: str, path: str, model: str, video_mode: str,
                      items: Sequence[CorpusItem]
                      ) -> Tuple[Dict[str, np.ndarray], List[str]]:
    out: Dict[str, np.ndarray] = {}
    errors: List[str] = []
    for it in items:
        try:
            out[it.id] = embed_one(base_url, path, it, model, video_mode)
            print(f"  embedded {it.id} ({it.kind}) dim={out[it.id].shape[0]}")
        except Exception as exc:
            errors.append(f"{it.id}: {exc}")
            print(f"  ERROR {it.id}: {exc}", file=sys.stderr)
    return out, errors


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="http://host:port")
    p.add_argument("--model", required=True, help="served model id")
    p.add_argument("--embeddings-path", default="/v1/embeddings")
    p.add_argument("--video-mode",
                   choices=("frames-datauri-list", "frames-image-list"),
                   default="frames-datauri-list")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save-reference", metavar="OUT.jsonl",
                   help="query endpoint and SAVE embeddings as the reference")
    g.add_argument("--reference", metavar="REF.jsonl",
                   help="compare current endpoint against this reference")
    p.add_argument("--out", help="write comparison JSON report here")
    p.add_argument("--insecure", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _SSL_CTX
    args = build_arg_parser().parse_args(argv)
    _SSL_CTX = _make_ssl_context(args.insecure)
    items = build_corpus()

    print(f"corpus: {sum(i.kind=='text' for i in items)} text, "
          f"{sum(i.kind=='image' for i in items)} image, "
          f"{sum(i.kind=='video' for i in items)} video")
    cur, errors = gather_embeddings(args.base_url, args.embeddings_path,
                                    args.model, args.video_mode, items)

    if args.save_reference:
        with open(args.save_reference, "w", encoding="utf-8") as fh:
            for it in items:
                if it.id in cur:
                    fh.write(json.dumps({"id": it.id, "kind": it.kind,
                                         "dim": int(cur[it.id].shape[0]),
                                         "embedding": cur[it.id].tolist()}) + "\n")
        ok = not errors
        print(f"saved reference: {args.save_reference} "
              f"({len(cur)}/{len(items)} items, {len(errors)} errors)")
        print(f"RESULT: {'PASS' if ok else 'FAIL'} (reference creation)")
        return 0 if ok else 1

    # --- comparison mode ---
    ref = load_reference(args.reference)
    per_kind: Dict[str, List[float]] = {"text": [], "image": [], "video": []}
    per_item_cos: Dict[str, float] = {}
    all_cos: List[float] = []
    missing: List[str] = []
    dim_mismatch: List[str] = []
    for it in items:
        if it.id not in cur or it.id not in ref:
            missing.append(it.id)
            continue
        if cur[it.id].shape != ref[it.id].shape:
            dim_mismatch.append(it.id)
            continue
        c = cosine(cur[it.id], ref[it.id])
        per_kind[it.kind].append(c)
        per_item_cos[it.id] = c
        all_cos.append(c)

    def bucket_stats(vals: List[float]) -> Tuple[int, float, float]:
        if not vals:
            return 0, float("nan"), float("nan")
        return len(vals), min(vals), sum(vals) / len(vals)

    text_n, text_min, text_mean = bucket_stats(per_kind["text"])
    mm_vals = per_kind["image"] + per_kind["video"]
    mm_n, mm_min, mm_mean = bucket_stats(mm_vals)
    overall_p1 = float(np.percentile(all_cos, 1.0)) if all_cos else float("nan")
    overall_p99 = float(np.percentile(all_cos, 99.0)) if all_cos else float("nan")

    text_pass = text_n > 0 and text_min >= TEXT_COSINE_MIN
    mm_pass = mm_n > 0 and mm_min >= MULTIMODAL_COSINE_MIN
    p99_pass = bool(all_cos) and overall_p1 >= P99_COVERAGE_COSINE_MIN
    passed = (text_pass and mm_pass and p99_pass
              and not errors and not missing and not dim_mismatch)

    print("=" * 64)
    print(f"{'bucket':8s} {'n':>3s} {'min_cos':>9s} {'mean_cos':>9s} "
          f"{'gate':>9s}  result")
    print(f"{'text':8s} {text_n:3d} {text_min:9.5f} {text_mean:9.5f} "
          f"{'>=%.3f' % TEXT_COSINE_MIN:>9s}  {'PASS' if text_pass else 'FAIL'}")
    print(f"{'image':8s} {len(per_kind['image']):3d} "
          f"{bucket_stats(per_kind['image'])[1]:9.5f} "
          f"{bucket_stats(per_kind['image'])[2]:9.5f} {'':9s}")
    print(f"{'video':8s} {len(per_kind['video']):3d} "
          f"{bucket_stats(per_kind['video'])[1]:9.5f} "
          f"{bucket_stats(per_kind['video'])[2]:9.5f} {'':9s}")
    print(f"{'multimod':8s} {mm_n:3d} {mm_min:9.5f} {mm_mean:9.5f} "
          f"{'>=%.3f' % MULTIMODAL_COSINE_MIN:>9s}  {'PASS' if mm_pass else 'FAIL'}")
    print(f"{'overall':8s} {len(all_cos):3d}  p1={overall_p1:.5f}  "
          f"p99={overall_p99:.5f}  gate p1>={P99_COVERAGE_COSINE_MIN}  "
          f"{'PASS' if p99_pass else 'FAIL'}")
    if missing:
        print(f"MISSING items (no ref or no current): {missing}")
    if dim_mismatch:
        print(f"DIM MISMATCH items: {dim_mismatch}")
    if errors:
        print(f"REQUEST errors: {len(errors)}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "gates": {"text": TEXT_COSINE_MIN,
                          "multimodal": MULTIMODAL_COSINE_MIN,
                          "p99": P99_COVERAGE_COSINE_MIN},
                "text": {"n": text_n, "min": text_min, "mean": text_mean},
                "multimodal": {"n": mm_n, "min": mm_min, "mean": mm_mean},
                "overall": {"n": len(all_cos), "p1": overall_p1, "p99": overall_p99},
                "per_item": per_item_cos,
                "missing": missing, "dim_mismatch": dim_mismatch,
                "errors": errors, "passed": passed,
            }, fh, indent=2)
        print(f"wrote report: {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
