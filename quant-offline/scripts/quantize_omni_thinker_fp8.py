#!/usr/bin/env python3
"""FP8 W8A8 thinker-only PTQ of Qwen3-Omni-30B-A3B-Instruct (nvidia-modelopt 0.45).

===================== OFF-VENDOR-VALIDATED-PATH ==========================
Rulings D2 / R5 (gate1_adjudications.md): vLLM-Omni documents ONLY an NVFP4
W4A4 thinker-only recipe as validated. This FP8 W8A8 thinker recipe is the
RECOMMENDED sm90 variant (native Hopper FP8 tensor cores, CUDA-graph-friendly,
and it SIDESTEPS the ModelOpt 0.44 NVFP4 per-block-E4M3 NaN weight-scale bug
entirely -- FP8 uses per-tensor F32 scales). But it is OFF the vendor-validated
path, so it is EVAL-GATED: promote ONLY after the omni <=1 pt task-eval gate AND
the fixed-prompt greedy 0-diff + degenerate scan (handoff/omni_audio_check.py)
both PASS. Omni default stays BF16 until then.

============================ BLOCKED PROTOCOL ============================
REQUIRES GPU. No GPU this cycle -> runnable handoff; it cannot be executed here.
Never fabricate an exported checkpoint or eval numbers.

HARDWARE / ENV NEEDED
    1x NVIDIA Hopper (H100/H200, sm90), enough VRAM to load Qwen3-Omni-30B-A3B
    in the OFFLINE quant image (IMAGE 3, gate1_adjudications.md section 2):
        nvidia/cuda:13.0.2-devel-ubuntu24.04, torch==2.11.0,
        nvidia-modelopt[hf]==0.45.0, transformers in [4.56, 5.10)
    (modelopt[hf] 0.45.0 pins transformers<5.10 -- do NOT install into the serve
     image; this is an offline job whose OUTPUT, not toolchain, ships. B-17.)

EXACT INVOCATION
    python3 quantize_omni_thinker_fp8.py \
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --output ./Qwen3-Omni-30B-A3B-Instruct-FP8-thinker \
        --calib-data ./omni_calib.jsonl        # optional; audio+text recommended
    # then auto-runs the scale-scan-equivalent on the export and prints the
    # <=1pt eval-gate instructions.

WHAT IT QUANTIZES / EXCLUDES  (exact module globs resolved at runtime, printed)
    QUANTIZE : the thinker MoE text body only (thinker.model.* attention +
               experts + shared_expert projections).
    EXCLUDE (keep BF16): audio_tower, visual, talker, code2wav, *lm_head*, and
               the MoE ROUTER gate (module ending exactly in `.mlp.gate`). The
               router glob is dot-anchored so it does NOT catch the SwiGLU
               `gate_proj` (B-13 -- the trap that would wrongly exclude experts).

EXPECTED-OUTPUT TEMPLATE (healthy)
    [audit] exclude *audio_tower* -> 525 modules ; *visual* -> 351 ; *talker* ->
            8037 ; *code2wav* -> 230 ; *lm_head* -> 1 ; router .mlp.gate -> 48
    [audit] quantize targets (thinker body Linear): ~1152
    quantized + exported to ./...-FP8-thinker
    [scale-scan fp8] RESULT: PASS
    NEXT (REQUIRED): run the <=1pt omni eval gate before promotion.
    RESULT: PASS (export + scale-scan)   [eval gate still pending]
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

# ---------------------------------------------------------------------------
# Constants / gates
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OMNI_EVAL_MAX_DELTA_PT = 1.0     # <=1 pt task-eval gate (run SEPARATELY, serving)
CALIB_SIZE_DEFAULT = 32
CALIB_MAX_TOKENS = 512

# Categories excluded from quantization (kept BF16). The router gate is handled
# separately with a DOT-ANCHORED pattern so it never matches `mlp.gate_proj`.
EXCLUDE_WILDCARDS = ["*audio_tower*", "*visual*", "*talker*",
                     "*code2wav*", "*lm_head*"]
ROUTER_SUFFIX = ".mlp.gate"      # exact module suffix of the MoE router


def _scale_scan_path(override: Optional[str]) -> str:
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "handoff", "scale_scan.py"))


# ---------------------------------------------------------------------------
# Module classification (resolved from the actual checkpoint, printed for audit)
# ---------------------------------------------------------------------------
def resolve_exclusions(model) -> Dict[str, List[str]]:
    """Walk named_modules() and return the resolved exclusion sets by category."""
    import torch.nn as nn
    names = [n for n, _ in model.named_modules()]
    resolved: Dict[str, List[str]] = {}
    for pat in EXCLUDE_WILDCARDS:
        resolved[pat] = [n for n in names if fnmatch.fnmatch(n, pat)]
    # Router: module name ends EXACTLY in `.mlp.gate` (NOT `.mlp.gate_proj`).
    routers = [n for n in names if n.endswith(ROUTER_SUFFIX)]
    resolved[ROUTER_SUFFIX] = routers
    # Sanity: prove the router set excludes no gate_proj / experts module.
    bad = [n for n in routers if "gate_proj" in n or "experts" in n]
    if bad:
        raise RuntimeError(f"router resolution caught SwiGLU/experts modules: {bad}")
    # Quantize targets: thinker-body Linear layers NOT in any exclusion set.
    excluded = set().union(*resolved.values()) if resolved else set()
    quant_targets = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n not in excluded
        and n.startswith(("thinker.model", "model.thinker.model"))
    ]
    resolved["__quantize__"] = quant_targets
    return resolved


def print_audit(resolved: Dict[str, List[str]]) -> None:
    parts = []
    for pat in EXCLUDE_WILDCARDS:
        got = resolved.get(pat, [])
        parts.append(f"{pat} -> {len(got)}")
    print("[audit] exclude " + " ; ".join(parts)
          + f" ; router {ROUTER_SUFFIX} -> {len(resolved.get(ROUTER_SUFFIX, []))}")
    if resolved.get(ROUTER_SUFFIX):
        sample = resolved[ROUTER_SUFFIX][:3]
        print(f"[audit] router modules (excluded, sample): {sample}")
    print(f"[audit] quantize targets (thinker body Linear): "
          f"{len(resolved.get('__quantize__', []))}")
    if resolved.get("__quantize__"):
        print(f"[audit] quantize sample: {resolved['__quantize__'][:3]}")


def build_quant_config(resolved: Dict[str, List[str]]):
    """FP8 W8A8 default config with the resolved exclusions disabled."""
    import copy
    import modelopt.torch.quantization as mtq
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg = cfg["quant_cfg"]
    # Category wildcards (unambiguous: none collide with thinker body).
    for pat in EXCLUDE_WILDCARDS:
        quant_cfg[pat] = {"enable": False}
    # Router: disable each resolved module's quantizers via a dot-anchored glob
    # derived from its EXACT name -> cannot match `gate_proj`.
    for name in resolved.get(ROUTER_SUFFIX, []):
        quant_cfg[f"*{name}.*"] = {"enable": False}
    return cfg


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _synth_calibration_texts(n: int) -> List[str]:
    base = [
        "Describe the audio you just heard in detail.",
        "Transcribe and summarize the following clip.",
        "What instruments or tones are present in this recording?",
        "Explain the emotional tone conveyed by the speaker.",
        "Provide a step-by-step analysis of the audio content.",
        "Answer the user's question about the sound thoroughly.",
    ]
    return [base[i % len(base)] for i in range(n)]


def load_calibration(model, tokenizer, processor, args, device) -> List[dict]:
    """Return a small list of model-input dicts. Prefers user --calib-data
    (text and, ideally, audio -- representative of production Call-type-A).
    Falls back to a synthesized TEXT set; a clear warning notes that AUDIO
    calibration is recommended for the audio workload."""
    samples: List[dict] = []
    rows: List[dict] = []
    if args.calib_data and os.path.exists(args.calib_data):
        with open(args.calib_data, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows = rows[:args.calib_size]
    else:
        print("[warn] no --calib-data; using synthesized TEXT calibration. For "
              "the audio workload, supply representative audio+text calibration.",
              file=sys.stderr)
        rows = [{"text": t} for t in _synth_calibration_texts(args.calib_size)]

    for row in rows:
        text = row.get("text", "")
        audio_path = row.get("audio")
        try:
            if audio_path and processor is not None:
                # Multimodal input via the omni processor (audio + text).
                import soundfile as sf  # available in the omni quant env
                audio, sr = sf.read(audio_path)
                messages = [{"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": text or "Describe the audio."}]}]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt, audio=[audio],
                                   sampling_rate=sr, return_tensors="pt")
            else:
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=CALIB_MAX_TOKENS)
            samples.append({k: v.to(device) for k, v in inputs.items()
                            if hasattr(v, "to")})
        except Exception as exc:
            print(f"[warn] calibration sample skipped ({exc})", file=sys.stderr)
    if not samples:
        raise RuntimeError("no usable calibration samples were built")
    print(f"[calib] built {len(samples)} calibration inputs "
          f"(audio-bearing: {sum('input_features' in s or 'input_audio' in s for s in samples)})")
    return samples


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", required=True, help="export directory")
    p.add_argument("--calib-data", help="JSONL: {\"text\":..,\"audio\":path?} per line")
    p.add_argument("--calib-size", type=int, default=CALIB_SIZE_DEFAULT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--scale-scan-path", help="override path to handoff/scale_scan.py")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve + print exclusions and quant config, then STOP "
                        "(no model load / no GPU needed for the plan only)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.dry_run:
        print("[dry-run] EXCLUDE wildcards:", EXCLUDE_WILDCARDS)
        print("[dry-run] router suffix (dot-anchored, excludes gate_proj):",
              ROUTER_SUFFIX)
        print("[dry-run] FP8 W8A8 thinker-only; encoders/talker/code2wav/lm_head/"
              "router kept BF16.")
        print("[dry-run] scale-scan (fp8 mode) will run on the export:",
              _scale_scan_path(args.scale_scan_path))
        print("RESULT: DRY-RUN OK (no quantization performed)")
        return 0

    # Heavy imports here so --dry-run and py_compile work without GPU/toolchain.
    try:
        import torch
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
        from transformers import AutoTokenizer
        try:
            from transformers import AutoProcessor
        except Exception:
            AutoProcessor = None
        try:
            from transformers import AutoModelForCausalLM as _AutoModel
        except Exception:
            from transformers import AutoModel as _AutoModel
    except Exception as exc:
        print(f"ERROR: missing quant toolchain (torch/modelopt/transformers): {exc}",
              file=sys.stderr)
        print("This is a BLOCKED GPU job; run it in the IMAGE 3 offline quant env.",
              file=sys.stderr)
        return 2

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} (dtype={args.dtype}) ...")
    model = _AutoModel.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
        device_map=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)
    processor = None
    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model, trust_remote_code=args.trust_remote_code)
        except Exception:
            processor = None

    # 1) resolve + audit exclusions
    resolved = resolve_exclusions(model)
    print_audit(resolved)
    if not resolved.get("__quantize__"):
        print("ERROR: resolved ZERO quantize targets -- check module prefixes "
              "(expected thinker.model.*).", file=sys.stderr)
        return 2

    # 2) build FP8 config with exclusions disabled
    cfg = build_quant_config(resolved)

    # 3) calibration forward loop
    device = next(model.parameters()).device
    samples = load_calibration(model, tokenizer, processor, args, device)

    def forward_loop(m):
        with torch.no_grad():
            for i, inp in enumerate(samples):
                try:
                    m(**inp)
                except Exception as exc:
                    print(f"[warn] calib forward {i} failed: {exc}", file=sys.stderr)

    print("quantizing (FP8 W8A8, thinker-only) ...")
    model = mtq.quantize(model, cfg, forward_loop)
    mtq.print_quant_summary(model)

    # 4) export HF checkpoint
    os.makedirs(args.output, exist_ok=True)
    print(f"exporting to {args.output} ...")
    export_hf_checkpoint(model, export_dir=args.output)
    tokenizer.save_pretrained(args.output)
    if processor is not None:
        try:
            processor.save_pretrained(args.output)
        except Exception:
            pass

    # 5) AUTO-RUN the scale-scan-equivalent on the export (FP8 mode: E4M3 weight
    #    NaN + F32 scale NaN/Inf). FP8 is NOT exposed to the NVFP4 per-block bug,
    #    but this is the fail-closed sanity gate on the export.
    scan = _scale_scan_path(args.scale_scan_path)
    print(f"[scale-scan fp8] {scan} --local {args.output} --mode fp8")
    scan_rc = 3
    try:
        scan_rc = subprocess.call([sys.executable, scan, "--local", args.output,
                                   "--mode", "fp8"])
    except Exception as exc:
        print(f"[scale-scan fp8] ERROR launching scanner: {exc}", file=sys.stderr)

    # 6) print the REQUIRED next-step eval gate (serving-dependent, not auto-run)
    print("=" * 64)
    print("NEXT (REQUIRED before promotion -- this recipe is OFF the vendor-")
    print("validated path, ruling D2/R5):")
    print("  a) Serve BF16 baseline and this FP8 export via vLLM-Omni (--omni).")
    print("  b) Create reference on BF16, gate the FP8 export:")
    print("       python3 handoff/omni_audio_check.py --base-url <bf16> \\")
    print("           --model Qwen/Qwen3-Omni-30B-A3B-Instruct --save-reference omni_ref.json")
    print("       python3 handoff/omni_audio_check.py --base-url <fp8> \\")
    print(f"           --model {os.path.basename(args.output)} --reference omni_ref.json --logprobs")
    print(f"  c) Run your audio TASK-EVAL suite; require delta <= {OMNI_EVAL_MAX_DELTA_PT} pt.")

    export_ok = os.path.isdir(args.output) and any(
        f.endswith(".safetensors") for f in os.listdir(args.output))
    passed = export_ok and scan_rc == 0
    print(f"criteria: export produced safetensors AND scale-scan(fp8)==PASS "
          f"(eval gate pending, run above)")
    print(f"RESULT: {'PASS' if passed else 'FAIL'} (export + scale-scan; "
          f"eval gate still REQUIRED)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
