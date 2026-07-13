#!/usr/bin/env python3
"""Degenerate-output detector for quantized / speculative-decode runs.

NO GPU, NO NETWORK REQUIRED. Pure Python standard library. Importable module
AND a command-line tool over a JSONL of generations.

WHY THIS EXISTS
---------------
The scope contract (Phase-1 brief section 2) mandates a "degenerate-output
detector (`!!!!`-class collapse) on every quantized/spec-decode run". Two
distinct failure modes must be caught:

  1. Repeated-token / repeated-char COLLAPSE ("!!!!!!!!!!..."), the classic
     signature of a ModelOpt 0.44 NaN weight-scale (E4M3 0x7F/0xFF) reaching a
     served GEMM (see handoff/scale_scan.py), or of a corrupted spec-decode
     draft (vLLM #39928 NaN -> "!!!!").
  2. N-GRAM LOOPS ("sorry, sorry, sorry ..." / agentic tool-call garbage
     loops), the signature of the native-MTP + prefix-caching correctness
     collapse on Qwen3.6-35B-A3B (vLLM #47087 "garbage token loops",
     #43559/#47194). See handoff/canary_repros.md.

The thresholds below are the numeric PASS/FAIL criteria; a run PASSES only if
ZERO generations are flagged degenerate.

USAGE (module)
--------------
    from degenerate_detector import analyze, analyze_tokens, DegenerateResult
    res = analyze("some generated text")
    if res.degenerate:
        print(res.reasons)

USAGE (CLI)
-----------
    python3 degenerate_detector.py --input generations.jsonl --field text
    python3 degenerate_detector.py --text "!!!!!!!!!!!!!!!!!!!!!!"
    cat gens.jsonl | python3 degenerate_detector.py --field output

Exit code 0 == PASS (no degenerate generations), 1 == FAIL, 2 == usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# PASS / FAIL numeric criteria (tune via CLI flags; these are the defaults the
# handoff pack gates on). A generation is DEGENERATE if ANY signal trips.
# ---------------------------------------------------------------------------
MAX_CHAR_RUN = 25          # longest run of one identical char; ">" => collapse
MAX_TOKEN_RUN = 10         # longest run of one identical whitespace token
NGRAM_N = 3                # n-gram size for loop detection
NGRAM_MAX_CONSECUTIVE = 5  # a block repeated back-to-back more than this => loop
NGRAM_MAX_FRACTION = 0.50  # single n-gram > this fraction of all n-grams => loop
DOMINANT_CHAR_FRACTION = 0.50  # one non-alnum char > this fraction => collapse
MIN_TOKENS_FOR_NGRAM = 12  # do not loop-scan generations shorter than this
MIN_NGRAMS_FOR_FRACTION = 20  # fraction test needs a reasonable population

# Characters that, when they dominate, indicate a numeric collapse rather than
# legitimate prose. Alphanumerics are excluded from the "dominant char" test so
# that e.g. a long code block of hex digits is not misflagged, but runs of any
# character (incl. alnum) still count toward MAX_CHAR_RUN.
_ALNUM_EXEMPT = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class DegenerateResult:
    """Structured verdict for one generation."""
    degenerate: bool
    reasons: List[str] = field(default_factory=list)
    num_chars: int = 0
    num_tokens: int = 0
    max_char_run: int = 0
    max_char_run_char: str = ""
    dominant_char: str = ""
    dominant_char_fraction: float = 0.0
    max_token_run: int = 0
    max_token_run_token: str = ""
    max_ngram_consecutive: int = 0
    ngram_consecutive_block: str = ""
    ngram_top_fraction: float = 0.0
    ngram_top_block: str = ""
    n_used: int = NGRAM_N

    def to_dict(self) -> dict:
        return asdict(self)


def _longest_run(seq: Sequence) -> Tuple[int, object]:
    """Longest run of identical adjacent elements. Returns (length, element)."""
    best_len = 0
    best_el = None
    cur_len = 0
    cur_el = object()
    for el in seq:
        if el == cur_el:
            cur_len += 1
        else:
            cur_el = el
            cur_len = 1
        if cur_len > best_len:
            best_len = cur_len
            best_el = el
    return best_len, best_el


def _dominant_char(text: str) -> Tuple[str, float]:
    """Most frequent non-whitespace character and its fraction of non-ws chars."""
    counts = Counter(c for c in text if not c.isspace())
    total = sum(counts.values())
    if total == 0:
        return "", 0.0
    ch, n = counts.most_common(1)[0]
    return ch, n / total


def max_ngram_consecutive(tokens: Sequence, n: int) -> Tuple[int, Optional[tuple]]:
    """Max number of times any n-gram repeats BACK-TO-BACK (period-n cycle).

    A clean "the cat sat the cat sat the cat sat" gives 3 for n=3. Runs are
    skipped past to keep this ~linear in len(tokens).
    """
    L = len(tokens)
    if n < 1 or L < 2 * n:
        return 0, None
    best = 1
    best_block: Optional[tuple] = None
    i = 0
    while i <= L - n:
        block = tuple(tokens[i:i + n])
        reps = 1
        j = i + n
        while j + n <= L and tuple(tokens[j:j + n]) == block:
            reps += 1
            j += n
        if reps > best:
            best = reps
            best_block = block
        i = j if reps > 1 else i + 1
    return best, best_block


def ngram_top_fraction(tokens: Sequence, n: int) -> Tuple[float, Optional[tuple]]:
    """Fraction of all n-grams accounted for by the single most common n-gram."""
    L = len(tokens)
    if n < 1 or L < n:
        return 0.0, None
    grams = Counter(tuple(tokens[i:i + n]) for i in range(L - n + 1))
    total = sum(grams.values())
    if total == 0:
        return 0.0, None
    block, cnt = grams.most_common(1)[0]
    return cnt / total, block


def analyze_tokens(
    text: str,
    tokens: Sequence,
    *,
    max_char_run: int = MAX_CHAR_RUN,
    max_token_run: int = MAX_TOKEN_RUN,
    ngram_n: int = NGRAM_N,
    ngram_max_consecutive: int = NGRAM_MAX_CONSECUTIVE,
    ngram_max_fraction: float = NGRAM_MAX_FRACTION,
    dominant_char_fraction: float = DOMINANT_CHAR_FRACTION,
    min_tokens_for_ngram: int = MIN_TOKENS_FOR_NGRAM,
    min_ngrams_for_fraction: int = MIN_NGRAMS_FOR_FRACTION,
) -> DegenerateResult:
    """Core analysis given both the raw text (for char signals) and a token
    sequence (for loop signals). Callers that only have text should use
    analyze(); callers holding real tokenizer ids should pass them here."""
    reasons: List[str] = []

    # --- character-level collapse signals ---------------------------------
    char_run, char_run_ch = _longest_run(text)
    char_run_ch = "" if char_run_ch is None else str(char_run_ch)
    if char_run > max_char_run:
        reasons.append(
            f"char-run: '{char_run_ch!r}' repeated {char_run} times "
            f"(> {max_char_run})"
        )

    dom_ch, dom_frac = _dominant_char(text)
    if (
        dom_frac >= dominant_char_fraction
        and dom_ch != ""
        and dom_ch not in _ALNUM_EXEMPT
    ):
        reasons.append(
            f"dominant-char: {dom_ch!r} is {dom_frac:.3f} of non-space chars "
            f"(>= {dominant_char_fraction})"
        )

    # --- token-level collapse / loop signals ------------------------------
    tok_run, tok_run_el = _longest_run(tokens)
    tok_run_el = "" if tok_run_el is None else str(tok_run_el)
    if tok_run > max_token_run:
        reasons.append(
            f"token-run: {tok_run_el!r} repeated {tok_run} times "
            f"(> {max_token_run})"
        )

    ng_consec = 0
    ng_consec_block: Optional[tuple] = None
    ng_frac = 0.0
    ng_frac_block: Optional[tuple] = None
    if len(tokens) >= min_tokens_for_ngram:
        ng_consec, ng_consec_block = max_ngram_consecutive(tokens, ngram_n)
        if ng_consec > ngram_max_consecutive:
            reasons.append(
                f"ngram-loop: {list(ng_consec_block or [])} repeated "
                f"{ng_consec}x consecutively (> {ngram_max_consecutive})"
            )
        ng_frac, ng_frac_block = ngram_top_fraction(tokens, ngram_n)
        n_ngrams = max(0, len(tokens) - ngram_n + 1)
        if ng_frac >= ngram_max_fraction and n_ngrams >= min_ngrams_for_fraction:
            reasons.append(
                f"ngram-dominance: {list(ng_frac_block or [])} is {ng_frac:.3f} "
                f"of all {ngram_n}-grams (>= {ngram_max_fraction})"
            )

    return DegenerateResult(
        degenerate=bool(reasons),
        reasons=reasons,
        num_chars=len(text),
        num_tokens=len(tokens),
        max_char_run=char_run,
        max_char_run_char=char_run_ch,
        dominant_char=dom_ch,
        dominant_char_fraction=round(dom_frac, 6),
        max_token_run=tok_run,
        max_token_run_token=tok_run_el,
        max_ngram_consecutive=ng_consec,
        ngram_consecutive_block=" ".join(map(str, ng_consec_block or [])),
        ngram_top_fraction=round(ng_frac, 6),
        ngram_top_block=" ".join(map(str, ng_frac_block or [])),
        n_used=ngram_n,
    )


def analyze(text: str, **kwargs) -> DegenerateResult:
    """Analyze a text generation using whitespace tokenization for loop signals.

    For char-level `!!!!`-class collapse this is exact. For n-gram loops on real
    model output, whitespace tokens are a robust proxy; pass real tokenizer ids
    to analyze_tokens() if you have them."""
    if text is None:
        text = ""
    tokens = text.split()
    return analyze_tokens(text, tokens, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _iter_jsonl(fh, field_name: str):
    for lineno, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            yield lineno, None, f"invalid JSON: {exc}"
            continue
        if isinstance(obj, str):
            yield lineno, obj, None
            continue
        if not isinstance(obj, dict):
            yield lineno, None, f"line is not an object or string: {type(obj)}"
            continue
        if field_name not in obj:
            yield lineno, None, f"missing field {field_name!r}; keys={list(obj)}"
            continue
        yield lineno, obj[field_name], None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Degenerate-output detector (repeated-token / n-gram loop / "
                    "!!!!-collapse). Exit 0 = PASS (none degenerate), 1 = FAIL.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", help="JSONL file of generations (default: stdin)")
    src.add_argument("--text", help="analyze a single string and exit")
    p.add_argument("--field", default="text",
                   help="JSON field holding the generation text (default: text)")
    p.add_argument("--max-char-run", type=int, default=MAX_CHAR_RUN)
    p.add_argument("--max-token-run", type=int, default=MAX_TOKEN_RUN)
    p.add_argument("--ngram-n", type=int, default=NGRAM_N)
    p.add_argument("--ngram-max-consecutive", type=int, default=NGRAM_MAX_CONSECUTIVE)
    p.add_argument("--ngram-max-fraction", type=float, default=NGRAM_MAX_FRACTION)
    p.add_argument("--dominant-char-fraction", type=float,
                   default=DOMINANT_CHAR_FRACTION)
    p.add_argument("--json", action="store_true",
                   help="emit per-generation results as JSONL to stdout")
    p.add_argument("--quiet", action="store_true",
                   help="only print the final PASS/FAIL summary")
    return p


def _analyze_kwargs(args) -> dict:
    return dict(
        max_char_run=args.max_char_run,
        max_token_run=args.max_token_run,
        ngram_n=args.ngram_n,
        ngram_max_consecutive=args.ngram_max_consecutive,
        ngram_max_fraction=args.ngram_max_fraction,
        dominant_char_fraction=args.dominant_char_fraction,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kwargs = _analyze_kwargs(args)

    total = 0
    degenerate = 0
    errors = 0

    def report(idx, text, res: DegenerateResult):
        if args.json:
            out = {"index": idx, **res.to_dict()}
            print(json.dumps(out, ensure_ascii=False))
        elif not args.quiet:
            tag = "DEGENERATE" if res.degenerate else "ok"
            preview = (text or "")[:60].replace("\n", "\\n")
            print(f"[{idx}] {tag:11s} chars={res.num_chars} "
                  f"tokens={res.num_tokens} :: {preview!r}")
            for r in res.reasons:
                print(f"        - {r}")

    if args.text is not None:
        total = 1
        res = analyze(args.text, **kwargs)
        degenerate += res.degenerate
        report(0, args.text, res)
    else:
        if args.input:
            try:
                fh = open(args.input, "r", encoding="utf-8")
            except OSError as exc:
                print(f"ERROR opening {args.input}: {exc}", file=sys.stderr)
                return 2
        else:
            fh = sys.stdin
        with fh:
            for lineno, text, err in _iter_jsonl(fh, args.field):
                if err is not None:
                    errors += 1
                    if not args.quiet:
                        print(f"[{lineno}] ERROR {err}", file=sys.stderr)
                    continue
                total += 1
                res = analyze(text, **kwargs)
                degenerate += res.degenerate
                report(lineno, text, res)

    ok = degenerate == 0 and errors == 0
    print("-" * 60)
    print(f"SUMMARY  analyzed={total}  degenerate={degenerate}  "
          f"read_errors={errors}")
    print(f"CRITERIA max_char_run>{args.max_char_run}  "
          f"max_token_run>{args.max_token_run}  "
          f"{args.ngram_n}-gram_consecutive>{args.ngram_max_consecutive}  "
          f"ngram_fraction>={args.ngram_max_fraction}  "
          f"dominant_char>={args.dominant_char_fraction}")
    print(f"RESULT   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
