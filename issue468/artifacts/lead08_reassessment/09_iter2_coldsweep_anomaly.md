# Lead 08 workstream — iteration-2 measurement-first probe: COLD-SWEEP ANOMALY (methodology blocker)

**Date:** 2026-07-16. **Status:** iteration-2 measurement-first probe DONE, but it surfaced an
**unexplained anomaly in the cold-sweep methodology** that must be resolved before any reliable
iteration-2 kernel GO/NO-GO.

## The probe (codex lead-ID's recommended no-kernel test)
Added an overlapped-vs-disjoint M=1×2 cold sweep to `metal_graph_test_m2_fidelity_unit` (ds4.c):
same K=2, same 12 physical pairs, but overlapped selections (n_unique=9) vs disjoint (n_unique=12).
Purpose: does the per-token dispatch cost track unique experts or physical pairs? (If equal → no
reuse → the grouped kernel has headroom.)

## The anomaly
```
OVERLAP_PROBE M1x2 overlapped(n_unique=9)=2.924 ms/layer vs disjoint(n_unique=12)=0.406 ms/layer
```
**Disjoint (12 distinct experts) is ~7× FASTER than overlapped (9)** — impossible for cold DRAM
loads (more experts must be ≥ cost). DS4_N_EXPERT=256, so the disjoint ids (6..11) are in-range
(not an out-of-range no-op). The disjoint sweep ran THIRD (after the M=2 + M=1×2-overlapped sweeps),
so this is a **sweep-order cache artifact**: later sweeps hit experts warmed by earlier ones.

## Likely cause (unconfirmed)
The M=1×2-overlapped sweep (2.9 ms, 3 passes) fits **pass-1-cold averaged with passes-2-3-cached**
(e.g. (8 + 0.4 + 0.4)/3 ≈ 2.9). The disjoint sweep (third) is mostly cached (experts 0..5 warmed by
the overlapped sweep) → 0.406 ms. The caching agent is most likely the **stream-expert-cache /
selected-slots path** inside `ds4_gpu_routed_moe_one_tensor` (gated on `use_iq2_selected_slots` /
`use_stream_expert_cache`, ds4_metal.m:23493), which `routed_moe_one` populates on pass 1. The M=2
orchestrator (`routed_moe_pair_tensor`) is a different path + doesn't use/populate it → stays cold.
**Not fully confirmed** — `use_iq2_selected_slots` may require a graph flag the standalone harness
lacks; the exact cause needs a focused pass.

## Impact
- **Iteration-1's M=2 4.4× slowdown is INFLATED by this confound** (M=2 fully cold vs M=1×2
  partially cached). The **qualitative** conclusion stands — codex independently confirmed the M=2's
  occupancy collapse (2× per-thread state, `yl_a/yl_b` + 4 accumulators) — but the **4.4× magnitude
  is not trustworthy** as measured.
- **Any iteration-2 kernel GO/NO-GO is unreliable** until the cold-sweep is fixed.

## Fix plan (for the next pass)
1. Pin the cause: run the overlap probe with the selected-slots / stream-cache paths disabled
   (`DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS=1`, `DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS=1`,
   and/or the streaming-prefetch disables) — does disjoint become cold (≈ overlapped)?
2. If yes → those paths were the cache; disable them for all cold measurements.
3. If no → isolate each sweep: single-pass (PASSES=1) AND each configuration as the FIRST/only sweep
   (env `DS4_M2_COST_MODE`), or separate process invocations (model reload evicts).
4. Re-validate iteration-1: M=2 vs M=1×2 cold-vs-cold (both first sweep, cache disabled). Expect the
   gap to shrink materially (the M=2 is occupancy-bound but not necessarily 4.4×).
5. Then build + measure iteration-2's sequential-tile kernel on the fixed methodology.

## Carry-over (unchanged)
The de-dup **thesis** is unaffected (the probe's pair-vs-unique showed routed cost tracks physical
pairs). The sequential-tile kernel design (codex-validated: per-token partials in threadgroup
memory, expert-major grid, reuse the existing down/sum6) is still the iteration-2 plan. Only the
**measurement methodology** needs fixing first.
