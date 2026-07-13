#!/usr/bin/env python3
"""Capacity probe: binary-search a vLLM serving parameter's ceiling on THIS host.

============================ BLOCKED PROTOCOL ============================
REQUIRES A GPU HOST (NVIDIA Hopper sm90) + a built serving image + docker with
the NVIDIA Container Toolkit. No GPU this cycle -> runnable handoff. NEVER
fabricate a capacity number -- each probe boots a real server (and, by default,
fires a real load burst) or the probe does not count.

WHAT IT DOES
    Finds the highest value of a capacity parameter that this machine can serve
    for a given model, by bisection. Each probe: launch container -> wait for
    /health -> (default) concurrent load burst -> verify the container survived
    -> tear down. Monotonicity assumed: if value V fails, V' > V is not tried.

    Parameters supported:
        max-num-batched-tokens   (vLLM --max-num-batched-tokens; integer grid)
        gpu-memory-utilization   (vLLM --gpu-memory-utilization; 0.xx grid)

    Rulings context: configs/* deliberately ship vLLM DEFAULTS for these knobs
    (Gate-1 rejection log: no tuning without a real load test). This tool IS
    that load test -- run it per host+model, record the ceiling, and set
    per-deployment overrides deliberately. It never edits the profiles.

    Pattern credit: the search-the-ceiling idea mirrors AWS DJL-LMI's mnttool
    (Apache-2.0). No code reused -- mnttool rebuilds a TensorRT engine per
    probe; vLLM only needs a serve restart, so this is a fresh implementation.

EXACT INVOCATION (example: NVFP4 profile stack, batched-tokens ceiling)
    python3 capacity_probe.py \
        --image model-opt-vllm/serve-main:vllm-0.25.0 \
        --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
        --extra-args "--quantization modelopt --kv-cache-dtype fp8 --moe-backend marlin --max-model-len 32768" \
        --param max-num-batched-tokens --lo 2048 --hi 131072 \
        --out capacity_mnbt.json

EXPECTED-OUTPUT TEMPLATE (healthy)
    probe  value   boot     load     verdict   note
      1    66560   ok       ok       PASS
      2    99328   ok       FAIL     FAIL      container died during burst (OOM in logs)
      ...
    RESULT: max-num-batched-tokens ceiling = 66560 on this host/model
    (exit 0 when a ceiling >= --lo is found; exit 1 if even --lo fails)
=========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PARAMS = {
    "max-num-batched-tokens": {"flag": "--max-num-batched-tokens", "lo": 2048, "hi": 131072,
                               "step": 1024, "kind": "int"},
    "gpu-memory-utilization": {"flag": "--gpu-memory-utilization", "lo": 0.70, "hi": 0.98,
                               "step": 0.01, "kind": "float"},
}
OOM_SIGNS = ("CUDA out of memory", "No available memory", "OutOfMemoryError",
             "KV cache", "insufficient", "engine core initialization failed")
CONTAINER = "capacity-probe"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def container_running() -> bool:
    r = sh(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER])
    return r.returncode == 0 and r.stdout.strip() == "true"


def container_logs(tail: int = 40) -> str:
    r = sh(["docker", "logs", "--tail", str(tail), CONTAINER])
    return (r.stdout or "") + (r.stderr or "")


def teardown() -> None:
    sh(["docker", "rm", "-f", CONTAINER])
    time.sleep(5)  # let the driver release GPU memory before the next probe


def http_get(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def http_post_json(url: str, payload: dict, timeout: float) -> int:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def fmt_value(v, kind: str) -> str:
    return f"{v:.2f}" if kind == "float" else str(int(v))


def build_docker_cmd(a, value_str: str) -> list[str]:
    cmd = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "--ipc=host",
           "-p", f"{a.port}:8000",
           "-v", f"{a.hf_home}:/hf-cache", "-e", "HF_HOME=/hf-cache"]
    if os.environ.get("HF_TOKEN"):
        cmd += ["-e", f"HF_TOKEN={os.environ['HF_TOKEN']}"]
    cmd += ["--entrypoint", "vllm", a.image,
            "serve", a.model, "--port", "8000", "--trust-remote-code",
            PARAMS[a.param]["flag"], value_str]
    cmd += shlex.split(a.extra_args)
    return cmd


def load_burst(a) -> tuple[bool, str]:
    """Concurrent completions burst; capacity failures usually surface under load."""
    prompt = "capacity probe token " * a.load_prompt_words
    payload = {"model": a.model, "prompt": prompt, "max_tokens": a.load_max_tokens,
               "temperature": 0.0}
    url = f"http://localhost:{a.port}/v1/completions"
    codes: list[int] = []
    lock = threading.Lock()

    def worker():
        code = http_post_json(url, payload, timeout=a.load_timeout)
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=worker) for _ in range(a.load_concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [c for c in codes if c != 200]
    if bad:
        return False, f"{len(bad)}/{len(codes)} burst requests failed (codes {sorted(set(bad))})"
    if not container_running():
        return False, "container died during burst"
    if http_get(f"http://localhost:{a.port}/health") != 200:
        return False, "unhealthy after burst"
    return True, ""


def probe(a, value, kind: str) -> tuple[bool, str, str]:
    """Returns (ok, boot_status, note)."""
    value_str = fmt_value(value, kind)
    teardown()
    r = sh(build_docker_cmd(a, value_str))
    if r.returncode != 0:
        return False, "launch-fail", r.stderr.strip()[-200:]
    deadline = time.time() + a.startup_timeout
    while time.time() < deadline:
        if not container_running():
            logs = container_logs()
            oom = next((s for s in OOM_SIGNS if s.lower() in logs.lower()), "")
            note = f"container exited during boot ({'OOM: ' + oom if oom else 'see logs'})"
            teardown()
            return False, "boot-fail", note
        if http_get(f"http://localhost:{a.port}/health") == 200:
            break
        time.sleep(5)
    else:
        teardown()
        return False, "boot-timeout", f"no /health 200 within {a.startup_timeout}s"
    if a.skip_load_check:
        teardown()
        return True, "ok", "boot only (--skip-load-check)"
    ok, note = load_burst(a)
    if not ok:
        logs = container_logs()
        oom = next((s for s in OOM_SIGNS if s.lower() in logs.lower()), "")
        if oom:
            note += f" (OOM in logs: {oom})"
    teardown()
    return ok, "ok", note


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", required=True, help="built serving image (variant A or B)")
    p.add_argument("--model", required=True, help="HF model id to serve")
    p.add_argument("--param", choices=sorted(PARAMS), default="max-num-batched-tokens")
    p.add_argument("--lo", type=float, default=None, help="search floor (default per param)")
    p.add_argument("--hi", type=float, default=None, help="search ceiling (default per param)")
    p.add_argument("--resolution", type=float, default=None, help="grid step (default per param)")
    p.add_argument("--extra-args", default="", help="extra vllm serve flags, one quoted string")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--hf-home", default=os.environ.get("HF_HOME",
                   os.path.expanduser("~/.cache/huggingface")))
    p.add_argument("--startup-timeout", type=int, default=1200,
                   help="seconds to wait for /health per probe (weights may download)")
    p.add_argument("--skip-load-check", action="store_true",
                   help="boot-only probes (weaker: OOM often appears under load)")
    p.add_argument("--load-concurrency", type=int, default=8)
    p.add_argument("--load-max-tokens", type=int, default=64)
    p.add_argument("--load-prompt-words", type=int, default=1024)
    p.add_argument("--load-timeout", type=float, default=300.0)
    p.add_argument("--out", help="write JSON report here")
    p.add_argument("--dry-run", action="store_true",
                   help="print the probe grid + docker command, execute nothing")
    a = p.parse_args()

    spec = PARAMS[a.param]
    kind = spec["kind"]
    lo = spec["lo"] if a.lo is None else a.lo
    hi = spec["hi"] if a.hi is None else a.hi
    step = spec["step"] if a.resolution is None else a.resolution
    grid = []
    v = lo
    while v <= hi + 1e-9:
        grid.append(round(v, 4) if kind == "float" else int(v))
        v += step
    if not grid:
        print(f"ERROR: empty grid (lo={lo} hi={hi} step={step})", file=sys.stderr)
        return 2

    if a.dry_run:
        print(f"[dry-run] param={a.param} grid: {len(grid)} candidates "
              f"{fmt_value(grid[0], kind)}..{fmt_value(grid[-1], kind)} step {step}")
        print(f"[dry-run] worst-case probes (bisection): ~{max(1, len(grid)).bit_length()}")
        print("[dry-run] docker command for the first probe:")
        print("   ", " ".join(shlex.quote(c) for c in build_docker_cmd(a, fmt_value(grid[0], kind))))
        return 0

    if os.environ.get("SKIP_GPU_PREFLIGHT", "0") != "1":
        if sh(["nvidia-smi", "-L"]).returncode != 0:
            print("BLOCKED: no NVIDIA GPU visible (nvidia-smi failed). This probe boots real "
                  "servers; run it on the sm90 host. SKIP_GPU_PREFLIGHT=1 overrides.", file=sys.stderr)
            return 3

    results = []
    lo_i, hi_i, best_i = 0, len(grid) - 1, -1
    n = 0
    # probe the floor first: if lo fails, nothing to search
    while lo_i <= hi_i:
        mid = (lo_i + hi_i) // 2 if best_i >= 0 or n > 0 else 0
        value = grid[mid]
        n += 1
        ok, boot, note = probe(a, value, kind)
        results.append({"probe": n, "value": value, "boot": boot, "ok": ok, "note": note})
        print(f"probe {n:>2}  {a.param}={fmt_value(value, kind):>8}  boot={boot:<12} "
              f"verdict={'PASS' if ok else 'FAIL'}  {note}")
        if ok:
            best_i = max(best_i, mid)
            lo_i = mid + 1
        else:
            hi_i = mid - 1

    report = {"param": a.param, "image": a.image, "model": a.model,
              "grid": {"lo": grid[0], "hi": grid[-1], "step": step},
              "load_check": not a.skip_load_check,
              "probes": results,
              "ceiling": grid[best_i] if best_i >= 0 else None}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
        print(f"report -> {a.out}")
    if best_i < 0:
        print(f"RESULT: FAIL — even {a.param}={fmt_value(grid[0], kind)} does not serve on this host")
        return 1
    print(f"RESULT: {a.param} ceiling = {fmt_value(grid[best_i], kind)} on this host/model "
          f"({'boot+load' if not a.skip_load_check else 'boot-only'} criterion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
