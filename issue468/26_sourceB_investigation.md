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
