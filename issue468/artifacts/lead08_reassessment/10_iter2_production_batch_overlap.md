# Lead 08 iteration 2 — production batch overlap probe

Date: 2026-07-17. Status: **measurement blocker resolved; bounded prototype authorized.**
Data: `10_iter2_production_batch_overlap.csv`.

> **Follow-up completed:** artifact 11 built the one authorized prototype. It is bit-exact but
> 22.9% slower at the target shape, so the grouped branch is now **NO-GO / stopped**. The
> conditional decision below is the historical iteration-2 gate, not the current verdict.

## Why the prior probe was invalid

Artifact 09 compared `ds4_gpu_routed_moe_one_tensor` twice after earlier sweeps. That path enters
the persistent selected-expert cache, while the M=2 prototype does not. Its 2.924 vs 0.406 ms
overlap anomaly and the earlier 4.4x M=2/M1 magnitude were cache-order artifacts. They do not
measure the production verifier targeted by iteration 2.

The replacement probe calls `ds4_gpu_routed_moe_batch_tensor`, the routed path used by
`metal_graph_verify_suffix_tops`. It fixes K=4 and 24 physical `(token, expert)` rows while varying
the number of unique experts: 6, 12, 18, and 24. Each sample:

- clears the software resident expert cache outside the timed region;
- traverses all 43 routed layers once with layer-varying valid expert IDs;
- alternates ascending/descending case order across four rounds;
- records end-to-end routed-MoE wall time;
- separately profiles `gate_up` and `down` with `DS4_METAL_MOE_STAGE_PROFILE`.

The stage profiler synchronizes between stages, so its absolute milliseconds are inflated. The
within-profile ratios are the decision evidence. Harness env:
`DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_BATCH_OVERLAP_PROBE=1`.

## Result

| unique / 24 pairs | total ms/layer | gate_up profile | down profile |
|---:|---:|---:|---:|
| 6 | 5.451 | 3.046 | 0.673 |
| 12 | 7.277 | 3.208 | 0.617 |
| 18 | 10.334 | 4.642 | 0.695 |
| 24 | 13.489 | 4.787 | 0.815 |

Two facts replace the earlier physical-pair-only story:

1. **The production batch path is already union-aware at preparation/residency.** Total routed
   cost tracks unique experts strongly (four-point linear `R^2 = 0.986`), not only physical pairs.
   The old K-sweep attribution conflated growing K, physical compute, and unique preparation.
2. **Sparse doubleton overlap is still poorly reused inside gate+up.** The realistic partial-overlap
   shape (18 unique: six duplicated assignments plus 12 singletons) has 25% fewer unique experts
   than the disjoint case, but `gate_up` is still 4.642/4.787 = **97%** of disjoint. The existing
   address-table kernel therefore gets little GPU-stage benefit from the six doubletons. Full
   overlap benefits more, so the mechanism is nonlinear; a blanket "no reuse" claim is also wrong.

`down` is small in this harness (0.695 vs 4.642 profile ms at 18 unique) and already falls 15%
below disjoint. It is not the first prototype target.

## Iteration-2 decision (superseded by artifact 11)

**CONDITIONAL GO for exactly one bounded gate+up prototype.** The controlled production path
confirms headroom at the actual K=4 overlap shape, but does **not** prove the prior 6–8 ms saving.
That prize is now unproven because resource preparation already de-duplicates unique experts and
the remaining GPU-stage saving depends on the cost of spilling partial accumulators.

Prototype design and gates:

- expert-major grid `(row_group, unique_expert)`;
- keep one token's `yl[32]` and accumulators live at a time;
- spill per-token partial gate/up accumulators to threadgroup memory across `ib32` tiles;
- retain existing selection-ordered down/sum6;
- compare at 18 unique / 24 physical pairs against the production `addr` gate+up kernel;
- continue only if gate+up improves at least 15% without fidelity regression beyond the retained
  relaxed sub-ULP/zero-argmax-flip bar;
- final Lead 08 GO still requires an end-to-end exact K=4 verifier at `verify_ms(4) <= 50.5 ms`.

The naive "compute token A then B while reusing registers" description is incomplete: both tokens
need partial sums across all input tiles. The threadgroup spill is the mechanism that avoids the
iteration-1 two-token register footprint while preserving tile reuse. If its synchronization and
shared-memory traffic erase the 15% gate, stop the grouped-kernel branch rather than iterate.
