#!/usr/bin/env python3
"""FALLBACK self-quant of Qwen3.6-35B-A3B (nvidia-modelopt 0.45). NOT the primary path.

=============================== FALLBACK ONLY ============================
PRIMARY PATH IS THE NVIDIA PRE-QUANT: use `nvidia/Qwen3.6-35B-A3B-NVFP4`
AS-IS (ruling D2). It is MIXED_PRECISION (W4A16_NVFP4 experts + FP8 attention +
FP8 KV), sm90-servable via Marlin W4A16, ships the BF16 MTP head, and is gated by
handoff/scale_scan.py. Do NOT self-quantize unless the pre-quant's full scale
scan FAILS or a different format is explicitly required.

W4A16_NVFP4 CAVEAT (ruling D2 / B-22): the modelopt 0.45 changelog adds
first-class `mtq.W4A16_NVFP4_CFG` (--qformat w4a16_nvfp4, weight-only, no
calibration forward) but states "vLLM deployment support is in progress" for
SELF-QUANTIZED w4a16_nvfp4 exports. The NVIDIA pre-quant already carries the
W4A16_NVFP4 label and vLLM v0.25.0 serves IT fine -- but a fresh self-quant
export of this format may not yet deploy. If unsure, use --format fp8 (per-tensor
F32 scales, no NVFP4 per-block-E4M3 NaN exposure, ~2x VRAM instead of ~3x).

============================ BLOCKED PROTOCOL ============================
REQUIRES GPU. No GPU this cycle -> runnable handoff only. Never fabricate an
exported checkpoint. Run in the IMAGE 3 offline quant env (torch==2.11.0,
nvidia-modelopt[hf]==0.45.0, transformers in [4.56, 5.10); B-17/D2). modelopt is
quantize-time ONLY -- keep it OUT of the serve image.

EXCLUSIONS (resolved from the checkpoint at runtime, printed for audit)
    - MTP head (`*mtp*`): kept BF16 so native MTP spec-decode stays composable
      (the pre-quant does the same; modelopt also auto-excludes MTP in Python).
    - Vision tower (`*visual*`): Qwen3.6-35B-A3B is natively MULTIMODAL
      (Qwen3_5MoeForConditionalGeneration + vision_config; COR-1) -- the vision
      tower MUST stay BF16.
    - `*lm_head*` and the MoE ROUTER gate (module ending exactly `.mlp.gate`,
      dot-anchored so it never catches SwiGLU `gate_proj`; B-13).

EXACT INVOCATION
    python3 quantize_qwen36_fallback.py --model Qwen/Qwen3.6-35B-A3B \
        --format fp8 --output ./Qwen3.6-35B-A3B-FP8 --calib-data ./calib.jsonl
    python3 quantize_qwen36_fallback.py --model Qwen/Qwen3.6-35B-A3B \
        --format w4a16_nvfp4 --output ./Qwen3.6-35B-A3B-W4A16NVFP4   # weight-only

EXPECTED-OUTPUT TEMPLATE (healthy, --format w4a16_nvfp4)
    [audit] exclude *mtp* -> 19 ; *visual* -> 333 ; *lm_head* -> 1 ; router -> 40
    [audit] quantize targets (Linear): ~880
    quantized + exported to ./Qwen3.6-35B-A3B-W4A16NVFP4
    [scale-scan nvfp4] RESULT: PASS
    RESULT: PASS (export + scale-scan)
    REMINDER: w4a16_nvfp4 self-quant vLLM deploy support is "in progress" (B-22).
=========================================================================
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
CALIB_SIZE_DEFAULT = 32
CALIB_MAX_TOKENS = 512

EXCLUDE_WILDCARDS = ["*mtp*", "*visual*", "*lm_head*"]
ROUTER_SUFFIX = ".mlp.gate"      # exact router suffix (dot-anchored vs gate_proj)

FORMATS = {
    # name -> (modelopt cfg attribute, needs_calibration, scale_scan_mode)
    "w4a16_nvfp4": ("W4A16_NVFP4_CFG", False, "nvfp4"),
    "fp8": ("FP8_DEFAULT_CFG", True, "fp8"),
}


def _scale_scan_path(override: Optional[str]) -> str:
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "handoff", "scale_scan.py"))


def resolve_exclusions(model) -> Dict[str, List[str]]:
    import torch.nn as nn
    names = [n for n, _ in model.named_modules()]
    resolved: Dict[str, List[str]] = {}
    for pat in EXCLUDE_WILDCARDS:
        resolved[pat] = [n for n in names if fnmatch.fnmatch(n, pat)]
    routers = [n for n in names if n.endswith(ROUTER_SUFFIX)]
    resolved[ROUTER_SUFFIX] = routers
    bad = [n for n in routers if "gate_proj" in n or "experts" in n]
    if bad:
        raise RuntimeError(f"router resolution caught SwiGLU/experts modules: {bad}")
    excluded = set().union(*resolved.values()) if resolved else set()
    quant_targets = [n for n, m in model.named_modules()
                     if isinstance(m, nn.Linear) and n not in excluded]
    resolved["__quantize__"] = quant_targets
    return resolved


def print_audit(resolved: Dict[str, List[str]]) -> None:
    parts = [f"{p} -> {len(resolved.get(p, []))}" for p in EXCLUDE_WILDCARDS]
    print("[audit] exclude " + " ; ".join(parts)
          + f" ; router {ROUTER_SUFFIX} -> {len(resolved.get(ROUTER_SUFFIX, []))}")
    if resolved.get("*mtp*"):
        print(f"[audit] mtp modules (excluded, sample): {resolved['*mtp*'][:3]}")
    print(f"[audit] quantize targets (Linear): {len(resolved.get('__quantize__', []))}")


def build_quant_config(fmt: str, resolved: Dict[str, List[str]]):
    import copy
    import modelopt.torch.quantization as mtq
    cfg_attr = FORMATS[fmt][0]
    if not hasattr(mtq, cfg_attr):
        raise RuntimeError(f"modelopt has no {cfg_attr} (need nvidia-modelopt "
                           f"0.45+ for w4a16_nvfp4). Available FP8 fallback: "
                           f"--format fp8.")
    cfg = copy.deepcopy(getattr(mtq, cfg_attr))
    quant_cfg = cfg["quant_cfg"]
    for pat in EXCLUDE_WILDCARDS:
        quant_cfg[pat] = {"enable": False}
    for name in resolved.get(ROUTER_SUFFIX, []):
        quant_cfg[f"*{name}.*"] = {"enable": False}
    return cfg


def _synth_calibration_texts(n: int) -> List[str]:
    base = [
        "Explain the architecture of a mixture-of-experts language model.",
        "Write a detailed function to parse a configuration file.",
        "Summarize the trade-offs between FP8 and NVFP4 quantization.",
        "Describe how speculative decoding improves inference latency.",
        "Give a thorough overview of attention mechanisms in transformers.",
        "Draft a step-by-step plan to benchmark an LLM serving stack.",
    ]
    return [base[i % len(base)] for i in range(n)]


def load_calibration(tokenizer, args, device) -> List[dict]:
    rows: List[dict] = []
    if args.calib_data and os.path.exists(args.calib_data):
        with open(args.calib_data, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows = rows[:args.calib_size]
    else:
        print("[warn] no --calib-data; using synthesized TEXT calibration. Use "
              "representative production prompts for a real FP8 quant.",
              file=sys.stderr)
        rows = [{"text": t} for t in _synth_calibration_texts(args.calib_size)]
    samples = []
    for row in rows:
        inputs = tokenizer(row.get("text", ""), return_tensors="pt",
                           truncation=True, max_length=CALIB_MAX_TOKENS)
        samples.append({k: v.to(device) for k, v in inputs.items()
                        if hasattr(v, "to")})
    if not samples:
        raise RuntimeError("no usable calibration samples")
    print(f"[calib] built {len(samples)} calibration inputs")
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--format", choices=list(FORMATS), default="fp8")
    p.add_argument("--output", required=True)
    p.add_argument("--calib-data", help="JSONL {\"text\":..} per line (FP8 only)")
    p.add_argument("--calib-size", type=int, default=CALIB_SIZE_DEFAULT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--scale-scan-path")
    p.add_argument("--dry-run", action="store_true",
                   help="print exclusions/plan and STOP (no GPU needed)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, needs_calib, scan_mode = FORMATS[args.format]

    if args.dry_run:
        print(f"[dry-run] PRIMARY PATH is nvidia/Qwen3.6-35B-A3B-NVFP4 (D2); this "
              f"is FALLBACK ONLY.")
        print(f"[dry-run] format={args.format} needs_calibration={needs_calib} "
              f"scale_scan_mode={scan_mode}")
        print(f"[dry-run] EXCLUDE wildcards: {EXCLUDE_WILDCARDS}; router suffix "
              f"{ROUTER_SUFFIX} (dot-anchored, excludes gate_proj).")
        if args.format == "w4a16_nvfp4":
            print("[dry-run] CAVEAT: w4a16_nvfp4 self-quant vLLM deploy support is "
                  "'in progress' (B-22). Prefer the NVIDIA pre-quant.")
        print(f"[dry-run] scale-scan ({scan_mode}) will run on the export: "
              f"{_scale_scan_path(args.scale_scan_path)}")
        print("RESULT: DRY-RUN OK (no quantization performed)")
        return 0

    try:
        import torch
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
        from transformers import AutoTokenizer
        try:
            from transformers import AutoModelForCausalLM as _AutoModel
        except Exception:
            from transformers import AutoModel as _AutoModel
    except Exception as exc:
        print(f"ERROR: missing quant toolchain (torch/modelopt/transformers): {exc}",
              file=sys.stderr)
        print("This is a BLOCKED GPU job; run in the IMAGE 3 offline quant env.",
              file=sys.stderr)
        return 2

    print(f"[note] FALLBACK path. Primary = nvidia/Qwen3.6-35B-A3B-NVFP4 as-is (D2).")
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} (dtype={args.dtype}) ...")
    model = _AutoModel.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
        device_map=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)

    resolved = resolve_exclusions(model)
    print_audit(resolved)
    if not resolved.get("__quantize__"):
        print("ERROR: resolved ZERO quantize targets.", file=sys.stderr)
        return 2

    cfg = build_quant_config(args.format, resolved)

    if needs_calib:
        device = next(model.parameters()).device
        samples = load_calibration(tokenizer, args, device)

        def forward_loop(m):
            with torch.no_grad():
                for i, inp in enumerate(samples):
                    try:
                        m(**inp)
                    except Exception as exc:
                        print(f"[warn] calib forward {i} failed: {exc}",
                              file=sys.stderr)
        print(f"quantizing (format={args.format}, WITH calibration) ...")
        model = mtq.quantize(model, cfg, forward_loop)
    else:
        # W4A16_NVFP4 is weight-only: no calibration forward (sidesteps
        # activation-scale artifacts; B alternates note).
        print(f"quantizing (format={args.format}, weight-only, NO calibration) ...")
        model = mtq.quantize(model, cfg, forward_loop=None)
    mtq.print_quant_summary(model)

    os.makedirs(args.output, exist_ok=True)
    print(f"exporting to {args.output} ...")
    export_hf_checkpoint(model, export_dir=args.output)
    tokenizer.save_pretrained(args.output)

    scan = _scale_scan_path(args.scale_scan_path)
    print(f"[scale-scan {scan_mode}] {scan} --local {args.output} --mode {scan_mode}")
    scan_rc = 3
    try:
        scan_rc = subprocess.call([sys.executable, scan, "--local", args.output,
                                   "--mode", scan_mode])
    except Exception as exc:
        print(f"[scale-scan] ERROR launching scanner: {exc}", file=sys.stderr)

    export_ok = os.path.isdir(args.output) and any(
        f.endswith(".safetensors") for f in os.listdir(args.output))
    passed = export_ok and scan_rc == 0
    print("=" * 64)
    if args.format == "w4a16_nvfp4":
        print("REMINDER: w4a16_nvfp4 self-quant vLLM deployment support is "
              "'in progress' (B-22) -- validate it actually loads in vLLM v0.25.0 "
              "before relying on it; the NVIDIA pre-quant remains the primary path.")
    print(f"criteria: export produced safetensors AND scale-scan({scan_mode})==PASS")
    print(f"RESULT: {'PASS' if passed else 'FAIL'} (export + scale-scan)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
