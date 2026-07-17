# Lead 08 — M=2 fused routed-expert COST (cold all-layers sweep)

> **2026-07-17 measurement correction:** the reported 4.4x magnitude is **not decision-grade**.
> M=2 ran before an M1x2 comparator whose selected-expert cache then stayed warm across three
> passes; artifact 09 exposed the order artifact and artifact 10 replaced the comparator with a
> cache-controlled production batch path. The qualitative iteration-1 NO-GO remains: the kernel
> computes 18 instead of 12 token-expert streams and holds two tokens' live state, so it is the
> wrong design. Do not reuse 4.4x or the derived 2.9x bound as a cold-vs-cold estimate.

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

## Verdict (iteration-1 cost — the simultaneous-fusion mechanism)
**Iteration-1 NO-GO.** The M=2 fused kernel was **reported as 4.4x slower**, but the comparator
was not cold-equivalent (see correction above). The de-dup was overwhelmed in this design by
18-vs-12 stream work plus the doubled live state. **This rules out the
*simultaneous-compute-fusion* mechanism, not Lead 08's
thesis** (de-dup is real) — Lead 08 continues with iteration 2 (sequential fusion / union-aware
de-dup, which avoids the 2× register pressure); see `pending/lead_08_fused_verify_kernel.md`.

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

## Codex gate B corrections (2026-07-16, gpt-5.5 xhigh — report retained)
Codex independently confirmed **NO-GO stands**, with these corrections to my analysis above:
1. **My "same compute as M=1×2" claim was WRONG.** The M=2 kernel computes BOTH tokens' gate/up
   MAC for every union slot (the sa/sb checks gate only the final writes, not the inner MAC) →
   18 token-expert streams for the 9-union/12-pair case, not 12. So the artifact's "same compute,
   fewer loads" premise is false. Fixing this (gating the inner MAC) bounds M=2 at
   13.33 × 12/18 ≈ 8.9 ms → **still 2.9× slower than M=1×2**. The bug is real but does not rescue it.
2. The slowdown is better described as `1.5× wasted-singleton compute × ~3× slower per token-expert
   unit` (register/occupancy), not a flat 6× per expert. A real fix is a new kernel design, not bounded.
3. **M=1×2 is NOT literally "batch-verifier-equivalent"** (suffix_tops batches all layers in one
   command stream + has a distinct batch MoE path). But M=1×2 is a *lenient* bar — losing 4.4× to
   it means losing worse to the real batch verifier.
4. **Timing-order caveat**: M=2 always ran before M=1×2; a rigorous repeat should alternate. But a
   4.4× gap is too large to be an order artifact. (Also corrected: M=1×2's two calls share ONE
   begin/end block, so it does NOT pay 2× dispatch overhead as I claimed.)
5. Correctness indexing confirmed sound (union, mid selection-order, the `weights[slot]` fix).
   Fidelity localization fair; the "bit-exact by construction" comment is too strong — soften.
6. Latent (non-blocking): the down path always dispatches the Q2_K sum6 pipeline without enforcing
   `down_type==Q2_K`; the M=2 kernel hard-codes 384 experts instead of `args.ne02`.

**Net: the fused-kernel approach is occupancy-bound here; the de-dup is overwhelmed. NO-GO on cost
is sound and not a bounded bug-hunt away from viability.**
