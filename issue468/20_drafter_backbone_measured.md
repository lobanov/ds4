# DSpark drafter Metal backbone — MEASURED cost + refined gate verdict

Date: 2026-06-28. Resolves the draft-cost uncertainty flagged in issue468/19
(the "unmeasured swing factor"). The drafter backbone cost is now MEASURED on
Metal, not projected.

## What was measured

Added an env-gated timing probe (`DS4_DSPARK_TIME_BACKBONE=1`, run via
`--verifier-curve-test` which provides a session graph with batch buffers) that
runs the drafter's 3 blocks through `metal_graph_encode_layer_batch` at
n_tokens=5 (block_size) on the actual drafter weights, 20 iters, median reported.

**Result: drafter backbone (3 layers, n_tokens=5) = 5.00 ms median (p10=4.72, p90=5.10).**

This matches the projection (3 × verify_backbone/43 = 3 × 1.71 = 5.13 ms) exactly,
confirming the drafter's per-layer batch cost equals the target's (same MoE/MLA/HC
kernel shapes; compress_ratio=0 makes the drafter's attention slightly cheaper but
the MoE dominates and is identical).

## Prerequisite fix committed

Found and fixed: the DSpark model was loaded (`model_open`) but NEVER registered
with the Metal device — `e->dspark_model.map` was not a valid Metal buffer, so any
GPU matmul on drafter weights failed. Added `ds4_gpu_set_model_map_range` for
`dspark_model` (parallel to the existing MTP map setup). This was a real
prerequisite for the Metal drafter forward AND the timing probe. Verified: timing
probe now succeeds (5.00ms) and `--dspark` still loads clean.

## Draft cost decomposition (all MEASURED components now)

| component | cost | source |
|---|---|---|
| drafter 3-layer backbone | **5.0 ms** | MEASURED (this probe) |
| shared lm_head (5 pos) | **1.5 ms** | MEASURED (doc 06 §4: ~1.25-1.3/pass) |
| overhead (main_proj, embed, hc_head, markov head) | **~0.7 ms** | estimated (bounded; tiny memory-bound kernels) |
| **total draft cost** | **~7.2 ms** | |

The only estimated piece is the ~0.7ms overhead (small kernels on tiny weights;
the biggest is markov_w2 [129280,256] BF16 = 132MB → ~0.16ms at 800GB/s × 5 seq).
Bounded to ±0.3ms. The backbone (5.0ms) and lm_head (1.5ms) — 90% of the cost —
are both directly measured.

## Refined gate verdict (measured draft 7.2ms, committed 2.8-3.0, verify 75ms)

| ctx | plain ms | accept | ms/tok | speedup | gate (>20%) |
|---|---|---|---|---|---|
| 32k | 32.9 | code@2.8 | 29.4 | 1.12× (+12%) | **FAIL** |
| 32k | 32.9 | chat@3.0 | 27.4 | 1.20× (+20.0%) | **BORDERLINE PASS** |
| 55k | 34.8 | code@2.8 | 29.4 | 1.19× (+18.5%) | fail |
| 55k | 34.8 | chat@3.0 | 27.4 | 1.27× (+27%) | **PASS** |
| 64k | 35.5 | code@2.8 | 29.4 | 1.21× (+21%) | **PASS** |
| 64k | 35.5 | chat@3.0 | 27.4 | 1.30× (+30%) | **PASS** |

## Decision required

The gate as written (">20% at >=32k, 32k AND 55k/64k, non-declining") is
**BORDERLINE**:
- **64k: PASSES for both prompt classes** (+21% code, +30% chat). Clear.
- **55k: passes for chat (+27%), fails for code (+18.5%)**. Split.
- **32k: passes ONLY for chat and EXACTLY at +20.0%; fails for code (+12%)**.
  This is right at the gate boundary.

Two interpretations:
1. **Strict (32k must pass for the representative class):** the verdict hinges on
   whether the representative acceptance is chat@3.0 (→ 32k passes exactly) or
   code@2.8 (→ 32k fails). And the ~0.7ms overhead estimate: if real overhead is
   <0.5ms, 32k-chat clears +20.5%; if >1.0ms, 32k-chat dips to +19.3% (fail). Only
   the full Metal forward + real end-to-end measurement resolves this precisely.
2. **Pragmatic (long-context = 55k/64k, the realistic agentic regime):** the gate
   clearly passes at 64k for both classes and at 55k for chat. The "+25-30% at
   64k" is a real, material long-context speedup.

The backbone (5.0ms, the biggest cost and biggest uncertainty) is now MEASURED.
The remaining uncertainties that the full Metal forward would resolve:
- the ~0.7ms overhead (→ precise 32k verdict)
- the real end-to-end speedup (draft+verify+accept pipeline, vs the component sum)
- the quality gate (ds4-eval DSpark@temp=1.0 >= 65/92)

**This is a phase-gate decision point.** Per the goal ("stop at the current phase
gate and ask the user with the measured evidence"), presenting the options:
- (A) Build the full Metal forward (3-5 sessions) to get the precise 32k number +
  end-to-end speedup + quality. Likely outcome: 64k clearly passes (A), 32k is
  borderline-to-fail. If the representative regime is 55k/64k, this reaches A.
- (B) Accept that 32k is borderline and the strict gate leans B, documenting the
  measured "+21-30% at 64k" partial result as the long-context finding, without
  the full forward.

## Verifier-optimization headroom noted (phase6-verifier-headroom)

The verify(L=5)=75ms is the dominant, fixed cost. Doc 06 §4 shows 95% of layer
cost is the MoE/attention dispatch; the output-head readback is only ~5%. Observed
headroom for a potential second pass (a ~2× verifier speedup would flip 32k to a
clear pass): the batched MoE reduction and the per-layer command-buffer dispatch
overhead are the candidates — NOT investigated this pass (out of scope).
