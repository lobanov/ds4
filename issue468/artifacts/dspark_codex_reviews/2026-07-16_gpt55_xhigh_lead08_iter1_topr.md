**Verdict:** Job 1 remains **NO-GO for ≥20%**, but with one important caveat: the `DS4_TOP_R` probe is not a perfectly clean semantic top-r probe for the first hash-routed layers.

`DS4_TOP_R` does propagate into batch verify:

- `ds4_spec_bench.c` reads env and calls `ds4_override_expert_used()` after engine open: [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1500)
- override mutates `g_ds4_shape.n_expert_used`, backing `DS4_N_EXPERT_USED`: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:304), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:26687)
- batch verify calls `metal_graph_encode_layer_batch()`: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21863)
- batch router select gets `DS4_N_EXPERT_USED`: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19393)
- routed MoE batch gets it as `n_expert`: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19633)
- Metal batch MoE uses `n_tokens * n_expert` physical rows/pairs: [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:25183)

The arithmetic is directionally sound: if the reducible routed work is roughly 40% of verify time, top-5’s 1/6 slot cut predicts about `0.40 * 1/6 = 6.7%`, matching the observed `55.8 -> 52.3 ms`. One caveat: for `n_expert < 6`, the path disables the direct `sum6` fast path and falls through to generic down + sum behavior, so this is not a pure “same kernel, fewer slots” measurement: [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:25562).

The bigger caveat: early hash-routed layers are likely mis-strided under the override. The model table is validated/loaded as width 6, but after override the GPU hash path uses `n_expert_used` as the row width. That means top-5/top-4 can read the hash table with the wrong stride, not merely truncate each row. Relevant code: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:3704), [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:22208). So the acceptance drop is a real fidelity red flag, but I would not call it clean evidence of “correct top-r routing hurts acceptance” without fixing that probe.

That caveat does **not** flip the NO-GO: the cost win is too small even before quality risk, and the measured top-5 throughput went down.

For Job 2: **sequential-tile grouped MoE is plausible, but the current design has a hidden state problem that must be made explicit.**

M=1 keeps one `yl[32]` plus two accumulator arrays live: [moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1059). M=2 keeps `yl_a`, `yl_b`, and four accumulator arrays: [moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1213). That explains the occupancy failure.

But if iteration-2 loads/dequants a small `ib32` tile once and applies it to multiple tokens, each token still needs accumulators across all `ib32` tiles. You have only three choices:

1. keep per-token accumulators in registers: repeats M=2 occupancy failure;
2. process one token across all `ib32`: loses de-dup because you reread/dequant for each token;
3. spill per-token partial accumulators to threadgroup memory: register-safe, but adds shmem traffic.

Option 3 is the viable design. It can keep register footprint near M=1, but it is not “no hidden state”; the hidden state moves to threadgroup memory. For K=4/5 this is probably affordable, but it must be measured.

Recommended shape:

- build a tiny per-layer unique map from `batch_router_selected`: `unique_ids`, `sub_count`, `(token, slot)` subscribers;
- dispatch gate/up as expert-major: `grid = (row_groups, unique_count, 1)`;
- keep `threadsPerThreadgroup = (32, FC_mul_mv_nsg=2, 1)`, matching M=1;
- `row_groups = ceil(expert_mid_dim / (N_R0_IQ2_XXS * NSG))`;
- write `mid[token, slot, row]` in the original selection slot layout;
- run the existing per-token selection-ordered down/sum path unchanged for exact sum order: [moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:2776).

On prize: `17 unique / 24 physical` at K=4 gives a 28.5% ideal de-dup ceiling. The artifact’s separator puts realistic de-dup at about **6-8 ms**. That is enough to beat the current batch verifier, but **not a guaranteed ≥20% clear** if the fixed-K baseline is ~59 ms and the gate is `≤50.5 ms`. It needs the high end of the estimate, low overhead, or composition with down de-dup / STS retune.

Best measurement-first probe before building: add a no-new-kernel controlled-selection batch MoE timing in the existing harness: same K and same physical pair count, but compare all-overlapped selected experts vs mostly-disjoint selected experts, with `DS4_METAL_MOE_STAGE_PROFILE` on. If current time is identical, cache reuse is not saving you today and the grouped kernel has real headroom. Also run the down-only cold sweep from the existing plan; down may be a smaller but cleaner composable win.

Main failure risks to avoid:

- compiler keeps `sum[token][row]` in registers via unrolling;
- partial accumulators spill to global instead of threadgroup memory;
- map construction requires CPU readback per layer, breaking the batch command stream;
- dequanting to shared as pre-multiplied floats changes FP order if bit-exactness is required;
- K=2/3 have little overlap, so gate the new path by duplicate factor or K.