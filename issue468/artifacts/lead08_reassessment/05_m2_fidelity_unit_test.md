# Lead 08 — M=2 fused routed-expert fidelity (unit test)

**Date:** 2026-07-16
**Harness:** `metal_graph_test_m2_fidelity_unit` (ds4.c), env `DS4_M2_FIDELITY_TEST`, invoked from
`ds4-spec-bench` after `ds4_engine_open` (runs under the target model; drafter-independent).
**Why a unit test, not the decode2_exact capture:** decode2_exact is MTP-only (the dspark section
returns before it; needs a `--mtp` model we don't have). An encode_decode_layer capture hits a
command-buffer wall: the M=2 orchestrator needs the 2nd token's router selections/weights on CPU
(for the union), but those live in the *current uncommitted* buffer. The self-contained unit test
(CPU-known inputs) sidesteps both.

## Method
For each of 5 layers (0, n/4, n/2, 3n/4, n-1), on identical deterministic inputs (random ffn_norm,
router selections **with overlap** selA={0..5} selB={3..8} → union {0..8}, random weights), run in
one command buffer:
- **M=1 reference:** `ds4_gpu_routed_moe_one_tensor` ×2 (one per token) → ref routed_out + ref mid.
- **M=2 fused:** `ds4_gpu_routed_moe_pair_tensor` (gate+up fused over the union + down×2) → m2 out + m2 mid.

Diff routed_out AND the gate+up mid (the mid localizes the residual source).

## Result (DeepSeek-V4-Flash-IQ2XXS, M5 Max, warm)
```
il=0  routed_out max_abs=3.725e-08 l1_rel=3.3e-07 argmax_flip=0 | gate+up(mid) max_abs=2.98e-08
il=10 routed_out max_abs=3.353e-08 l1_rel=3.0e-07 argmax_flip=0 | gate+up(mid) max_abs=7.45e-08
il=21 routed_out max_abs=3.353e-08 l1_rel=3.1e-07 argmax_flip=0 | gate+up(mid) max_abs=2.24e-08
il=32 routed_out max_abs=3.353e-08 l1_rel=3.2e-07 argmax_flip=0 | gate+up(mid) max_abs=3.73e-08
il=42 routed_out max_abs=4.843e-08 l1_rel=3.2e-07 argmax_flip=0 | gate+up(mid) max_abs=5.96e-08
SUMMARY worst_routed_out=4.8e-08 worst_gateup_mid=7.5e-08 argmax_flip=0 -> CLOSE_FASTMATH_NOISE(loc gate+up)
```

## Verdict (fidelity, the exactness-secondary objective)
- **NOT bit-exact** (max_abs ≈ 1e-8, not 0). Strict max_abs==0 gate: FAIL.
- **Meets the user-granted relaxed bar**: distribution-close (l1_rel ≈ 3e-7, sub-ULP) + argmax-stable
  (0 flips across all layers) → score-neutral, the M3 batched-verify precedent.
- **Localized to the fused gate+up kernel** (mid residual ≈ routed_out residual; the down/sum6 is the
  SAME kernel for M=1 and M=2, so it adds no residual beyond propagating the mid's). Root cause: the
  M=2 gate+up kernel (`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32_m2`) has 2× interleaved accumulators
  (a+b) vs M=1's single set → different register pressure / FMA-contraction under Metal fast-math →
  ~1e-8 FP-reordering noise. Inherent to a distinct fused kernel; not a fixable reduction bug.

## Status
Fidelity-gate: relaxed bar MET (bit-exact elusive — fast-math). Pending codex confirmation (gate B)
that this is fast-math noise + not a latent bug, and the cost gate (the primary question).
