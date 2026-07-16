# Lead 08 — M=2 fused routed-expert COST (cold all-layers sweep)

**Date:** 2026-07-16
**Harness:** `metal_graph_test_m2_fidelity_unit` cost section (ds4.c), env `DS4_M2_FIDELITY_TEST`.
**Method:** sweep ALL layers (DS4_N_LAYER=43) × 3 passes, dispatching per layer either M=2
(`ds4_gpu_routed_moe_pair_tensor`, fused gate+up over the 9-expert union + down×2) or M=1×2
(`ds4_gpu_routed_moe_one_tensor` ×2, per-token, no de-dup = the batch-verifier-equivalent routed
MoE). Each layer's experts are fresh (working set n_layer×6×~0.3GiB ≫ GPU L2) → DRAM, the
production regime. (A tight single-layer loop is CACHED — iters 2..N re-hit L2 at ~0.4ms and hide
the de-dup; useless for cost. The probe's pair-vs-unique already showed suffix_tops routed cost
tracks PHYSICAL pairs = DRAM cache-miss.) Same fixed overlapping selections selA={0..5} selB={3..8}
→ n_union=9 of 12 physical pairs.

## Result (cold, M5 Max, warm model)
```
COST_COLD passes=3 layers=43 selA=[0..5] selB=[3..8] (n_union=9 of 12)
  M2   = 13.334 ms/layer   (1.48 ms/expert over 9)
  M1x2 =  3.039 ms/layer   (0.25 ms/expert over 12)
  saving = -10.295 ms  (-338%)   -> M=2 is 4.4x SLOWER
```

## Verdict (cost, the primary objective)
**NO-GO.** The M=2 fused kernel is **4.4× slower** than per-token M=1×2 (both cold). The de-dup
(saving 3 of 12 expert loads) is completely overwhelmed by the fused kernel's per-expert overhead.

## Diagnosis (the ~6× per-expert slowdown)
- Expected work: M=2 processes 9 union experts; the 3 shared experts do 2 tokens' MAC, the 6
  single-selection do 1 → 3·2 + 6·1 = 12 expert-token units = SAME compute as M=1×2's 12, PLUS
  M=2 saves 3 expert LOADS (dequant). So an efficient M=2 should be FASTER (same compute, fewer loads).
- Measured: M=2 is 6× slower PER EXPERT (1.48 vs 0.25 ms). The extra ~3× (beyond the expected 2×
  for two tokens) is **register pressure**: the M=2 kernel holds 2× the per-thread state
  (`yl_a[32]`, `yl_b[32]`, `sumg_a/u_a`, `sumg_b/u_b` ≈ 68 floats vs M=1's `yl[32]`+2 accumulators
  ≈ 34) → low GPU occupancy → the slowdown.
- This is plausibly fixable (process the two tokens with less per-thread state / sequential reuse
  of the yl + accumulator registers), but closing a 4.4× gap to beat the batch verifier is a large
  kernel rework — well beyond a bounded bug-hunt, and the goal's hard-exit gate says don't grind.

## Cross-check vs the probe
The cold M=1×2 (3.0 ms/layer, 2 tokens) vs the probe's full-cycle verify_ms(2)=37.8ms/43layers ≈
0.88ms/layer: M=1×2 is ~3.4× slower than suffix_tops's per-layer because M=1×2 uses 2 separate
dispatches (no batch amortization of attention/dense + per-dispatch begin/end overhead). So M=1×2
is a *conservative* baseline; vs the actual batch verifier (suffix_tops, amortized) the M=2 is
even further behind. Either way: M=2 loses on cost.

## Status
Cost-gate: **FAIL** (M=2 routed-MoE ms NOT < batch-verifier-equivalent; it's 4.4× worse, cold).
Pending codex gate B (independent assessment of fixability + the verdict).
