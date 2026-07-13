# Open Risk Register — status as of 2026-07-13 (post-Gate-3)

| # | Risk | Status | Disposition / evidence |
|---|------|--------|------------------------|
| R1 | No GPU anywhere this cycle — all runtime behavior is code/doc-verified only | **OPEN (structural)** | Every GPU-dependent test ships as a runnable BLOCKED handoff (`handoff/`); zero fabricated numbers. First GPU session executes: smoke per profile, parity procedure, FA3 inspection, MTP grid |
| R2 | modelopt-0.44 NaN-scale bug; NVIDIA checkpoint produced with 0.44.0; 0.45 changelog silent on a fix | **CLOSED-PASS (this checkpoint)** | Full scan executed 2026-07-12: 30,841/30,841 F8_E4M3 scale tensors clean (`handoff/results/scale_scan_result.2026-07-12.json`). Prod re-runs on its own copy; any FUTURE modelopt export needs a fresh scan (bug unfixed upstream) |
| R3 | MTP × prefix-caching correctness collapse (open upstream: #43559/#47194/#47087; 5 fix PRs unmerged) | **OPEN (mitigated)** | MTP profile hard-disables prefix caching; default profile keeps MTP off; canary repros gate enablement |
| R4 | FA3-active-on-cu13-wheel unverified — affects default FLASH_ATTN backend + fp8_e4m3 KV | **OPEN (mitigation VERIFIED)** | Gate-3: FlashInfer backend supports fp8/fp8_e4m3 KV with no capability gate (code-read at tag) → the FP8-KV lever survives backend fallback. Residual: wheel-contents inspection + GPU smoke (Gate-2 items) |
| R5 | Omni FP8-thinker recipe off vendor-validated path (vendor documents only NVFP4 W4A4) | **OPEN (contained)** | Omni default is BF16; the FP8 recipe is an offline script gated on the ≤1 pt omni eval |
| R6 | vllm-omni × vLLM 0.25.0 untested upstream (0.25.0rc1 still bases on 0.24.0) | **OPEN (designed around)** | Omni isolated on the 0.24.x line in both variants; watch upstream for the 0.25 rebase |
| R7 | Internal 4-segment wheels may carry required patches (private index unreachable) | **OPEN (prod-side)** | Wheel-diff reconciliation procedure in `docs/migration-notes.md`; owner sign-off before cutover |
| R8 | Parity vs the production precomputed index untestable here | **OPEN (procedure shipped)** | Frozen preprocessing pins in every profile; `--save-reference` two-phase procedure + stored-index sampling in migration notes; #33204 documented |
| R9 | `evs.py` carries an NVIDIA-proprietary copyright header inside Apache-2 vLLM | **NOTED** | EVS unused (and a parity violation anyway); flag to legal only if prod redistributes vLLM differently than upstream ships it |
| R10 | Driver requirement is an enumerated branch list (535/550/565/570/575), not "≥535" | **SHARPENED (Gate-3)** | Host check added to migration notes; 545.x/560.x-class drivers fail `NVIDIA_REQUIRE_CUDA` |
| R11 | MTP can be net-slower at low batch (#47277/#39680) and corrupt at conc≥4 (#35288); MTP+FP8 long-seq crash at v0.19.1 (#40756, unretested at 0.25.0) | **OPEN (gated)** | All four issues verified open 2026-07-12/13; each is a mandatory cell/scan in `handoff/specdec_grid.py` + `canary_repros.md`; MTP ships off-by-default |
| R12 | Local Gate-2 builds blocked: Docker Desktop VM disk 60 G (98% full); authoring host is arm64/no GPU | **OPEN (environment, not design)** | Builds + in-image checks + negative-index test pending: raise Docker disk to ≥160 G or run README §2 on any linux/amd64 host. Compensating executed checks: uv cross-platform resolution of all four graphs (zero drift, `*/resolved-lock.txt`) |
| R13 | apt system packages unpinned (Ubuntu archive drift makes hard pins brittle) | **ACCEPTED** | As-built versions recorded in `/etc/image-manifest.json` per image; treat manifest as record of truth per build |

Residual UNVERIFIED claims (all acknowledged, none load-bearing for the build): B-19 (GB300 accuracy
table's transfer to the Hopper W4A16 path), B-21 (vllm-omni raw-W4A4 on Hopper — moot, omni ships
BF16), C-13 (MTP+FP8 stability at 0.25.0 — canary #40756), C-21 (fp8-KV-recovers-accuracy,
single-reporter — grid axis), D-17 (FA3 on cu13 — see R4), D-18 (modelopt fp8_cast KV-scale loading —
moot for the pre-quant). Full dispositions: `docs/claim-ledger.md` §6.
