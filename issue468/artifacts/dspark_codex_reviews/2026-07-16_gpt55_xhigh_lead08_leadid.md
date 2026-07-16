Read-only pass completed.

**Grounding**
Current `verify_suffix_tops` calls the batch layer path, which reaches `ds4_gpu_routed_moe_batch_tensor`; that path still dispatches routed work over physical `(token, expert)` pairs, not unique experts. The failed M=2 kernel is specifically the bad shape: it keeps two token vectors and four accumulators live in one thread, so it repeats the M5 Max occupancy failure.

**Priority Order**

1. **Union-aware grouped MoE, but sequential-tile, not simultaneous M=2**
   Mechanism: group token-expert assignments by unique expert; load/dequant an expert tile once, then process each subscribed token with M=1-sized register state.
   Saving: best match to the measured prize, about 6-8 ms likely, 9-10 ms optimistic at K=4; could clear the +20% gate if both gate/up and down share loads with low overhead.
   Fidelity: relaxed-ok; bit-exact only if per-token MAC and down sum order stay M=1-compatible.
   Surface: [ds4_gpu_routed_moe_batch_tensor](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:24978), current IQ2 pair kernels at [moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1022) and addr batch path at [moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1434).
   Probe: extend the existing K sweep/profile to dump per-expert token histograms from [metal_graph_batch_selected_profile_stats](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:12401); GO if K=4 remains around 17 unique / 24 pairs and routed gate/up+down is still >=25 ms real. Add a no-kernel cold all-layer “union prefetch then M=1x2” variant to `metal_graph_test_m2_fidelity_unit`.
   Occupancy: safe only if it keeps one token’s `yl[32]` and accumulators live at a time. Any design resembling [the failed M=2 state](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1213) is likely occupancy-bound.

2. **MoE-Spec-style expert budgeting / top-r slot compaction**
   Mechanism: cap or compact routed experts during verify, e.g. run top-5 or top-4 instead of top-6, or budget unique experts per layer using router mass.
   Saving: top-5 can plausibly save ~4-5 ms; top-4 can approach ~8-10 ms and might clear only if quality holds. Pure “unique budget with substitution” saves almost nothing in today’s DS4 path unless paired with de-dup, because current cost tracks physical slots.
   Fidelity: highest risk; requires ds4-eval no-regression. Relaxed bar is mandatory.
   Surface: router selection/weights at [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:22040), CPU equivalent at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:7560), routed batch call at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19633).
   Probe: no new Metal kernel: pack budgeted/top-r selected IDs and weights into scratch tensors, call existing `ds4_gpu_routed_moe_batch_tensor` with `n_expert=r`, run `DS4_DSPARK_VERIFY_DIST_PROBE` plus ds4-eval. GO only if top-r cost actually drops and argmax/eval holds.
   Occupancy: not occupancy-bound; it reduces physical work.

3. **Down/sum6-specific de-dup**
   Mechanism: share Q2_K down expert loads across verify tokens while preserving each token’s six-expert sum order.
   Saving: likely 2-3 ms alone, not enough for +20%, but useful if gate/up grouped work lands.
   Fidelity: bit-exact possible if down remains selection-ordered per token; reordering the six-slot reduction risks flips.
   Surface: [kernel_mul_mv_id_q2_K_sum6_f32](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:2776), addr version at [moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:2970).
   Probe: add a down-only cold sweep to `metal_graph_test_m2_fidelity_unit`: generate/reference mids once, then time sum6 with overlapped vs disjoint selected experts across all layers.
   Occupancy: moderate risk if multiple token `sumf` arrays are live; low risk if token processing is sequential after shared tile load.

4. **Cost-aware STS / temporal-overlap scheduler**
   Mechanism: use expert-overlap cost in `verify_n` choice, preferring K where expected accepted tokens per verify-ms is best.
   Saving: likely 1-4 ms effective average, unlikely to clear +20% alone; low-risk composition with grouped MoE.
   Fidelity: bit-exact/no quality risk, because it changes scheduling only.
   Surface: DSpark verify length logic near [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:30090), plus existing profile counters from `DS4_MTP_VERIFY_EXPERT_PROFILE`.
   Probe: replay existing timing logs with `(verify_n, accepted, avg_unique, conf_logits)` and compare current STS vs cost-aware thresholds. GO only for a measurable tokens/sec gain on held-out traces.
   Occupancy: no GPU occupancy risk.

5. **Targeted union prefetch / address-table reuse**
   Mechanism: explicitly prepare the union experts before current per-pair kernels, hoping repeated expert slots hit cache/resident resources.
   Saving: cheap to test, but likely 0-2 ms; generic readahead already measured as noise.
   Fidelity: exact.
   Surface: selected-address path gated around [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:8878) and batch addr call at [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:25570).
   Probe: in the M2 harness, alternate cold sweeps: baseline M=1x2, union-prefetched M=1x2, and disjoint-selection M=1x2. GO only if prefetch moves cold cost materially.
   Occupancy: none, but likely bandwidth/cache-capacity limited.

**Recommendation**
Run probes 2, 4, and 5 first because they need no new Metal kernel and can retire risky branches quickly. The main kernel iteration should be rank 1: expert-major load sharing with sequential per-token compute. Do not build another simultaneous M=2 accumulator kernel.

Sources checked: vLLM fused MoE modular kernel, PyTorch locality-aware MoE scheduling, Cohere Apr 2026 MoE speculative decoding post, and arXiv:2602.16052 MoE-Spec.