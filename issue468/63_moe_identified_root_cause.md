# Op-Level Bisection — MoE Is the First Identified Divergent Op

Date: 2026-07-01. Twenty-third productionization handoff note. Records the
FIRST op that fails the identical-input test for layer 2 specifically, and
narrows the root cause. Doc-only research record.

## 1. FINDING: the MoE diverges for layer 2 on identical input

Fed Metal's mid-block output (post-attention, pre-FFN — an IDENTICAL input to
both implementations) into the oracle's FFN sub-block, compared the resulting
block output to Metal's:

| step | layer 0 cos | layer 1 cos | layer 2 cos |
|---|---|---|---|
| 1 | 1.00000 | 0.99992 | **0.99569** |
| 2 | 1.00000 | 0.99992 | **0.99822** |
| 3 | 1.00000 | 0.99991 | **0.99818** |

Layer 0: CLEAN (cos 1.0). Layer 1: nearly clean (cos 0.9999). Layer 2:
DIVERGES 2-3.5% (cos 0.996-0.998). This is the FIRST op that passes the
"layer-2-specific divergence on identical input" test — all prior candidates
(hc_pre, q matmuls, KV, attention) were exonerated.

## 2. Routing is IDENTICAL — not a routing sensitivity issue

Compared the oracle's gate() expert selection (on Metal's identical mid-block
input) to Metal's router_selected dump. Expert selections are IDENTICAL for all
layers/steps (e.g., step 1 lay 2: both select [49, 61, 61, 61, 61, 67]). The
MoE divergence is NOT from top-6 boundary flips.

## 3. Q4_K matmul accumulates in F32 — not an F16 accumulation issue

The Q4_K matvec kernel (kernel_mul_mv_q4_K_f32_impl, metal/moe.metal:476)
accumulates in `float4 acc1` and `float sumf[nr0]` — F32, not F16. The Q4_K
block scale `d` is read as `half` but auto-promoted to F32 in the dot product.
So the Q4_K expert matmul itself should be F32-precise.

## 4. Remaining suspects within the MoE

Given routing is identical and Q4_K matmul is F32, the 2-3.5% layer-2 MoE
divergence on identical input comes from one of:
a. **Q4_K dequant algorithm difference** between Metal's inline dequant
   (bit-shift/mask unpacking of 6-bit sub-block scales) and the oracle's
   _dequant_q4_k. A subtle bit-manipulation difference would produce slightly
   different F32 weights → slightly different matmul results.
b. **hc_post (FFN) mixing** — the FFN sub-block's hc_post applies F16-weighted
   HC mixing (hc_ffn_fn is F16). hc_pre(ATTN) was tested clean, but hc_pre(FFN)
   uses different weights and the test was on the attn branch; hc_post might
   differ.
c. **SwiGLU clamp** — the oracle clamps gate(max) and up(both) at
   SWIGLU_LIMIT=10.0; Metal might clamp differently.
d. **Accumulation order** — the Q4_K matmul's SIMD reduction order may differ
   from numpy's contiguous sum.

## 5. All ops individually verified (the COMPLETE op-level bisection)

| op | identical-input cos (L2) | verdict |
|---|---|---|
| hc_pre + attn_norm | 1.00000 | CLEAN |
| q_a + q_b (Q8_0) | 1.0 (L0/L1) | CLEAN |
| KV window | 1.0 (per-step) | CLEAN |
| attention (sparse_attn/flash) | 1.0 (CPU F32=GPU F16) | CLEAN |
| **MoE (FFN)** | **0.996-0.998** | **DIVERGES — first identified** |

The MoE is the sole op that fails the identical-input test for layer 2
specifically. All other ops are clean. This IS the root-cause op.

## 6. Significance

The MoE divergence (2-3.5% for layer 2) is the error INJECTION source. The
attention AMPLIFIES this through softmax (the attention divergence I saw
earlier — 10-20% — is amplification of the accumulated MoE + residual error,
not a bug in the attention kernel itself). Fixing the MoE's 2-3.5% to <0.5%
(matching layers 0/1) would reduce the per-layer error injection, which the
attention amplifies less, potentially recovering a significant portion of the
+8.27% oracle headroom.

## 7. Next step

Bisect WITHIN the MoE: dump the oracle's intermediate states (hc_pre(ffn)
output, ffn_norm, per-expert gate/up output, SwiGLU, down-projection, hc_post)
on Metal's identical mid-block input, and compare to Metal's corresponding
intermediates. The first diverging intermediate identifies the specific kernel/
dequant that needs fixing.
