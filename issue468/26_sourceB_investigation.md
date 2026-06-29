# Source B pinpointed — NOT mid precision; it's the mv-path F16 activations

Date: 2026-06-29. Sharpens Assignment 2's target (issue468/25 exit).

## What was ruled out: mid precision

The independent review (issue468/24) and the moe.metal comment ("stores the down-
projection input in half precision") suggested the SwiGLU `mid` rounding to F16 was
Source B. Tested directly:

- `ds4_gpu_routed_moe_batch_tensor` has `request_mid_f16 = !g_quality_mode && ...`.
- The drafter (n_tokens=5, Q4_K, 256 experts) takes the catch-all **"mv"** path
  (not mm_id which needs n_tokens>=32; not tiny_pair which needs n_tokens<=4; not
  q4_table which needs n_total_expert==384).
- DS4_METAL_MOE_STAGE_PROFILE confirms: WITHOUT `--quality`, drafter MoE = `path=mv
  mid=f16`; WITH `--quality`, drafter MoE = `path=mv mid=f32`.
- BUT acceptance is IDENTICAL with/without `--quality` (greedy 1.53, same histogram;
  B2 committed 1.31). So **mid f16→f32 has NO effect on acceptance.** Mid is ruled out.

## What Source B actually is: F16 activations in the gate/up matmul

The same moe.metal comment: "The grouped routed-MoE matmul loads **activation tiles
as half** before using SIMD-group MMA." The gate/up projections of each routed expert
take the drafter's F32 activation and round it to F16 for the grouped matmul. The
oracle (F32) keeps activations in F32 throughout. This activation rounding is the
dominant precision loss — it's applied to the FULL [4096]-dim activation at every
expert op, and compounds through silu + down-proj + the 3-layer sequential chain.

## Implication for Assignment 2

The gathered-dense path (the review's Assignment 2) is the correct fix because it
uses `ds4_gpu_matmul_f16_f32` which takes **F32 activations** (not F16) with F16
weights and F32 accumulation. This bypasses the mv path's activation-to-half
conversion entirely. Flipping `--quality`/`mid` does NOT suffice (ruled out above);
the gathered-dense rewrite is necessary.

Remaining precision delta after gathered-dense: the dequanted expert weights will be
F16 (via Q4_K→F16), whereas the oracle uses F32 weights. F16-weight rounding is
~0.5%/element (small vs the activation rounding), so gathered-dense should capture
MOST of the oracle's advantage. If it falls short, an F32-weight path (harder —
ds4_gpu_matmul_f32_tensor is single-token) would be needed.

## Files
- DS4_METAL_MOE_STAGE_PROFILE output (above) confirms path=mv + mid flip.
- moe.metal:171-174 documents the F16-activation load.

## UPDATE: F16 activations ALSO ruled out (2026-06-29)

Tested by monkeypatching moe.swiglu_expert to round the activation x to F16 before
the gate/up matmul (simulating the Metal mv-path F16 activation load), then running
the validated measure_b2_acceptance.py:

- Oracle F32 (baseline): B2 committed 2.801
- Oracle F16-ACTIVATIONS: B2 committed 2.801 (IDENTICAL)
- Per-position accept [0.76, 0.691, 0.632, 0.551, 0.425] — same as F32.

(Greedy prefix dipped 2.79→2.15, but B2 committed is unchanged — B2 is robust to
F16 activations because acceptance depends on distribution overlap, not argmax.)

**So both F16-mid AND F16-activations are ruled out as Source B.** Neither
--quality (mid) nor F16-activation rounding explains the Metal 1.31 vs oracle 2.80 gap.

## Remaining Source B candidate: Q4_K weight dequant algorithm difference

The ONLY remaining explanation for the 0.979/block FFN corr (Metal vs oracle, both
reading the SAME Q4_K GGUF bytes) is a difference in the Q4_K DEQUANT algorithm:
the oracle's ported dequant_q4_k (from ggml-quants.c) vs ds4's Metal hardware
dequant produce different F32 weight values. This compounds through 6 experts ×
silu × down-proj × 3-layer chain → B2 collapses 2.80→1.31.

**Implication for Assignment 2:** the gathered-dense path may NOT fix Source B if
it uses Metal's dequant. The fix depends on whether Metal's Q4_K dequant is
CORRECT (matching the reference) or has a subtle bug/difference. Next step:
directly compare oracle's Q4_K expert dequant to a reference (or to Metal's) for
one expert. Phase 3 crosscheck validated F32/BF16 (47/47 byte-exact) but never
validated Q4_K (lossy, no byte-compare possible).

This is a higher-value finding than expected: Source B may be a dequant CORRECTNESS
issue, not a precision tradeoff — which would mean the fix is fixing the dequant,
not the accumulation path.
