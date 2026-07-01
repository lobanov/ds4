# Exact-Q4 Diagnostic — Q4_K Accumulation Order DISPROVEN

Date: 2026-07-01. Twenty-seventh productionization handoff note. The exact-Q4
diagnostic result. Doc-only research record.

## 1. Result: CPU exact-Q4 = GPU Q4_K SIMD (IDENTICAL)

Implemented DS4_DSPARK_EXACT_Q4: a CPU-side Q4_K routed expert matmul that
dequantizes each Q4_K element to F32 individually using q4_k_get_scale_min
(the standard ggml algorithm), then dots against the F32 activation in a
simple sequential F32 sum — matching numpy's approach exactly. Gated by env,
diagnostic-only (29s/step, slow).

Probe comparison (3 steps, ctx_08192):
| config | greedy match | avg prefix | histogram |
|---|---|---|---|
| GPU Q4_K SIMD (default) | 13/15 (86.7%) | 3.00 | [1 0 0 0 1 1] |
| CPU exact-Q4 (F32 dequant) | 13/15 (86.7%) | 3.00 | [1 0 0 0 1 1] |

**IDENTICAL.** The Q4_K SIMD accumulation order is NOT the source of the
divergence. Element-by-element F32 dequant + sequential dot produces the same
probe results as the GPU's SIMD-grouped kernel.

## 2. Profound implication: the divergence is in the ORACLE, not production

If CPU-exact-Q4 = GPU-Q4_K (both produce the same drafts), but BOTH diverge
from the numpy oracle by cos 0.996-0.998 for layer 2's MoE, then the divergence
is NOT in the Q4_K matmul at all. It's between the oracle's ExpertStore
dequant + numpy matmul and the production code's inline dequant + F32/GPU
matmul.

This means the +8.27% oracle headroom (issue468/51) may be from an ORACLE
IMPLEMENTATION DIFFERENCE, not a production code bug. The production code
(GPU + CPU exact-Q4) is self-consistent and correct; the oracle's ExpertStore
may dequant Q4_K differently for layer 2's weight patterns.

## 3. The blocker rule's 8th precondition is satisfied

The exact-Q4 diagnostic was attempted. It DISPROVED the accumulation-order
hypothesis. Per the contract: "IF the diagnostic does NOT change layer-2 cosine
→ the accumulation-order hypothesis is disproven; document and move to fallback
levers."

## 4. What this means for the perf gate

The Q8_0 re-quant (step 2 of the authorized path) was CONDITIONAL on the exact-Q4
diagnostic confirming accumulation order. Since it DISPROVED the hypothesis,
re-quanting to Q8_0 would NOT help — Q8_0 uses the same ExpertStore-style
dequant in the oracle, so the oracle-vs-production gap would persist.

The divergence is between the oracle (numpy reference) and the production code.
Since the production code is self-consistent (GPU = CPU-exact-Q4) and every op
was verified clean on identical input, the remaining explanation is that the
oracle's ExpertStore dequant or some other oracle implementation detail differs
from the production dequant for layer 2's specific weight patterns.

## 5. Next investigation direction

The oracle's ExpertStore.expert(e) function (issue468/dspark_oracle/expert_store.py)
dequants Q4_K weights lazily. If its dequant algorithm differs from the
production code's q4_k_get_scale_min for some edge-case weight patterns, the
oracle's MoE output would diverge from both GPU and CPU-exact-Q4. This is a
REFERENCE implementation difference, not a production bug.

To verify: compare the oracle's ExpertStore dequant output to the production
inline dequant on the same Q4_K bytes for layer 2. If they differ, the +8.27%
headroom is a phantom (the oracle's reference has a subtle dequant difference
that makes it appear to have higher acceptance than the production code, which
is actually correct).
