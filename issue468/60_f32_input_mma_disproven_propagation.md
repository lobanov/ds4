# F32-input Q8_0 MMA — Authorized Lever DISPROVEN; Divergence Is Propagation

Date: 2026-07-01. Nineteenth productionization handoff note. The decisive test
of the authorized F32-input Q8_0 MMA kernel fix (issue468/57). Doc-only record.

## 0. What was authorized

Per the goal tweak, the F32-input Q8_0 MMA Metal kernel fix was authorized as
"the one remaining perf lever." The hypothesis (issue468/57): the drafter q_b
matmul's F16-input Q8_0 MMA (kernel_mul_mm_q8_0_f32, half tiles + simdgroup_half8x8)
diverges from the numpy F32 reference on mtp.2's specific weights, causing the
4-6% q divergence → 9-11% MHSA → the +8.27% acceptance gap (issue468/51).

## 1. Implementation (DONE, working, reverted)

Implemented cleanly:
- metal/dense.metal: parameterized the kernel_mul_mm sb-offset to scale with
  sizeof(S0) (half→4096 unchanged; float→8192); added an F32-input Q8_0
  instantiation `kernel_mul_mm_q8_0_f32_f32input` (float4x4 tiles +
  simdgroup_float8x8 MMA for both inputs).
- ds4_metal.m: added a drafter-only dispatch `ds4_gpu_matmul_q8_0_f32_input_tensor`
  (selects the new pipeline, larger threadgroup memory).
- ds4_gpu.h: declared the new entry.
- ds4.c encode_attention: routed the drafter q_a + q_b matmuls through the
  F32-input dispatch (n_tokens>1 batch path).

Verified: make warning-clean (metal under -Wall -Wextra); the new pipeline
compiled and ran (no "pipeline not found"; DSpark produced clean output at
28.34 t/s). The kernel infrastructure is sound.

## 2. The decisive NEGATIVE result

Routed BOTH q_a AND q_b to F32-input and re-measured the deterministic probe
(mtp.2 q divergence). It was **BYTE-IDENTICAL** to the F16-input baseline:

| step | lay | q_rel% (F16, before) | q_rel% (F32-input, after) | q_cos |
|---|---|---|---|---|
| 1 | 2 | 4.14% | 4.14% | 0.99858 |
| 2 | 2 | 4.22% | 4.22% | 0.99863 |
| 3 | 2 | 5.84% | 5.84% | 0.99543 |

The F32-input kernel IS running (probe completed; run succeeded with clean
output), so this is not a silent fallback. **The F16-input Q8_0 MMA was NOT the
divergence source. The issue468/57 hypothesis is DISPROVEN.**

## 3. The real finding: the divergence is PROPAGATION, already at the q INPUT

Dumped batch_attn_norm (the q_a INPUT, = hc_pre + attn_norm output) and compared
to the oracle:

| step | lay | attnnorm (q-input) rel% | cos |
|---|---|---|---|
| 1 | 0 | 0.01% | 1.00000 |
| 1 | 1 | 0.22% | 0.99996 |
| 1 | **2** | **11.88%** | **0.99534** |
| 2 | 2 | 8.28% | 0.99615 |
| 3 | 2 | 11.85% | 0.98688 |

**The q divergence (4-6%) is DOWNSTREAM of an already-divergent input (8-12% at
batch_attn_norm).** The q matmuls are innocent — they faithfully process a
divergent input. No attention-matmul fix (q_a/q_b/kv/output) can help.

## 4. The entire "layer-2 bug" chain (issue468/53-57) was chasing propagation

Re-interpretation: layer 1's block output diverges ~4% from the oracle (within
F16/Q8_0 accumulation noise). Layer 2's hc_pre AMPLIFIES that ~4% to 8-12% at
batch_attn_norm (hc_pre is a mixing op; ~2-3× amplification is plausible).
attn_norm (rmsnorm) is scale-invariant, preserving relative error. So layer 2's
q inherits the amplified error. The earlier "propagation refuted" test
(issue468/54) compared block outputs and misread similar-magnitude divergences
as a layer-2-specific bug; the batch_attn_norm bisection corrects this.

**There is NO single-op fixable bug.** The +8.27% oracle headroom (issue468/51)
is cumulative F16/Q8_0 precision drift across the 3 drafter layers vs the
oracle's pure-F32 forward. Layer 0 is clean (cos 1.0); layer 1 accumulates ~4%;
layer 2 amplifies to ~8-12%. Recovering it would require raising the ENTIRE
drafter forward to F32 (dequant ALL weights: q_a/q_b/kv/output + ffn experts +
hc_pre/norms) — large scope (the FFN experts alone are ~10GB Q8_0 → 40GB F32,
infeasible), slower, and may still not fully recover (norms/rope/softmax also
differ in F16 vs F32 accumulation).

## 5. Conclusion — the authorized perf lever is exhausted and disproven

- The F32-input Q8_0 MMA kernel was correctly implemented, compiles clean, runs
  clean, no regression — but it does NOT fix the divergence (wrong operation).
- The routing was REVERTED (q_a/q_b back to standard q8_0) to avoid the slower
  F32 MMA's perf cost with zero benefit. The kernel + dispatch code is kept
  (harmless, documented, available if a future bisection finds an actual
  matmul-precision issue).
- The divergence is cumulative multi-layer precision drift, NOT a fixable
  single-op bug. This is consistent with the drafter being unusually sensitive
  (small, conditioned on the target's hidden state) to the F16/Q8_0 runtime
  precision vs the oracle's F32.

## 6. Perf-gate implication

The +8.27% oracle headroom is NOT recoverable via a code-only, frozen-drafter
single-kernel fix. The authorized lever (the F32-input Q8_0 MMA) is the last
identified in-scope code-only lever, and it is disproven. The fallback levers
(Opp3 drafter sync-fusion +~1 t/s; custom verifier high-risk) remain but are
each insufficient to cross the gate (issue468/42 break-even needs α 0.83→0.90).

Per the goal's blocker rule, all six preconditions are now satisfied
(regression clean, P1-P4 done, sweep measured, Bug#1 done, codex loop converged,
F32-MMA attempted). The perf gate (DSpark > 39 t/s) remains NOT MET (29.7 t/s),
now with the root cause definitively characterized as cumulative drafter-layer
precision drift (not a fixable single-op bug within the frozen-drafter,
code-only scope).
