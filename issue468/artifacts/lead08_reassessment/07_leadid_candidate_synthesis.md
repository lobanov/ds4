# Lead 08 workstream — iteration-0 lead-ID synthesis (candidate verify-cost mechanisms)

> **2026-07-17 final update:** artifact 11 closes candidate 1 negative: the bounded
> threadgroup-spill prototype is bit-exact but 22.9% slower at the target shape. Top-r and prefetch
> were already negative; down-only de-dup and cost-aware STS were composition-scale and no longer
> have a winning base mechanism. Artifact 12 also closes the margin fallback. Artifact 13 finds
> <2% sensitivity to contiguous, same-set permuted, or slab-wide strided expert placement, closing
> address remapping/packing. Artifact 14 finds only noisy 1-3% effects from exact NSG 1/4/8
> production variants. Artifact 15 finds exact row tiles 2/8 slower total with no steady-state gain.
> Artifact 16 finds a large exact cache-replay upper bound only on the DSpark-incompatible SSD
> selected-address path; the compatible mapped path changes by 0.04%. Both geometry knobs and the
> current-path residency carrier are far below the stage gate. Artifact 17 corrects the original
> actual-selection locality profile; the corrected reuse remains real but does not change the
> carrier result. Rankings below are
> historical context, not active recommendations.

**Date:** 2026-07-16. **Provenance:** 2 online searches (Cohere Apr-2026 MoE+SD post; MoE-Spec
arXiv:2602.16052; + the standard fused/grouped-MoE pattern: vLLM Fused-MoE-Modular, PyTorch
Locality-Aware MoE, FusedXpert SC25) + 1 codex lead-ID pass (gpt-5.5 xhigh, report retained at
`issue468/artifacts/dspark_codex_reviews/2026-07-16_gpt55_xhigh_lead08_leadid.md`).

## Grounding (codex-confirmed)
`metal_graph_verify_suffix_tops` (ds4.c:21808) → `ds4_gpu_routed_moe_batch_tensor`
(ds4_metal.m:24978) dispatches routed work over **physical (token, expert) pairs**, not unique
experts → the de-dup gap is real and lives in the production batch path. Iteration 1's simultaneous
M=2 (2× per-thread state) repeated the M5-Max occupancy failure — **do not build another
simultaneous-accumulator kernel.**

## Candidates, priority-ranked (mechanism / saving / fidelity / surface / probe / occupancy)

**1. Union-aware grouped MoE — sequential-tile, NOT simultaneous.** Group token-expert assignments
by unique expert; load/dequant an expert **tile** once, then process each subscribed token with
**M=1-sized register state** (one token's `yl[32]`+accumulators live at a time).
- Saving: best match to the prize — ~6–8 ms likely, 9–10 ms optimistic @ K=4; **could clear +20%**.
- Fidelity: relaxed-ok; bit-exact only if per-token MAC + down sum order stay M=1-compatible.
- Surface: `ds4_gpu_routed_moe_batch_tensor` (ds4_metal.m:24978); IQ2 pair kernel (moe.metal:1022); addr batch path (moe.metal:1434).
- Probe: (a) dump per-expert token histograms from `metal_graph_batch_selected_profile_stats` (ds4.c:12401) — GO if K=4 stays ~17 unique / 24 pairs; (b) **add a no-kernel cold "union-prefetch then M=1×2" variant to `metal_graph_test_m2_fidelity_unit`** — if prefetch alone drops M=1×2 from 3.0 ms toward cached, ordering captures the de-dup cheaply (no kernel).
- Occupancy: SAFE only with single-token register state. Any M=2-like 2× state is occupancy-bound.

**2. MoE-Spec expert budgeting / top-r slot compaction.** Cap routed experts at verify (top-5/top-4
instead of top-6), or budget unique experts per layer by router mass.
- Saving: top-5 ~4–5 ms; top-4 ~8–10 ms (clears only if quality holds). Pure unique-budget saves little today (cost tracks physical slots) unless paired with de-dup.
- Fidelity: **highest risk** — needs ds4-eval no-regression. Relaxed bar mandatory.
- Surface: router select/weights (ds4_metal.m:22040; CPU ds4.c:7560); routed batch call (ds4.c:19633).
- Probe: **no new kernel** — pack top-r selected IDs/weights into scratch, call existing `ds4_gpu_routed_moe_batch_tensor` with `n_expert=r`, run `DS4_DSPARK_VERIFY_DIST_PROBE` + ds4-eval. GO if cost drops AND argmax/eval holds.
- Occupancy: none (reduces physical work).

**3. Down/sum6-specific de-dup.** Share Q2_K down expert loads across verify tokens, preserve each token's 6-expert sum order.
- Saving: ~2–3 ms alone (not enough alone) — composes with #1.
- Fidelity: bit-exact possible if down stays selection-ordered per token.
- Surface: `kernel_mul_mv_id_q2_K_sum6_f32` (moe.metal:2776); addr variant (moe.metal:2970).
- Probe: add a down-only cold sweep to the harness — generate mids once, time sum6 with overlapped vs disjoint selections, all layers.
- Occupancy: moderate if multiple `sumf[]` live; low if token processing is sequential after a shared tile load.

**4. Cost-aware STS / temporal-overlap scheduler.** Use expert-overlap cost in the verify_n choice (prefer K with best accepted-tokens/verify-ms).
- Saving: ~1–4 ms avg; unlikely +20% alone; low-risk composition with #1.
- Fidelity: none (scheduling only — bit-exact, no quality risk).
- Surface: DSpark verify-length logic (ds4.c:30090); `DS4_MTP_VERIFY_EXPERT_PROFILE` counters.
- Probe: replay timing logs (verify_n, accepted, avg_unique, conf_logits), compare current STS vs cost-aware thresholds. GO for a measurable tokens/sec gain on held-out traces.
- Occupancy: none.

**5. Targeted union prefetch / address-table reuse.** Explicitly prepare the union experts before the per-pair kernels, hoping repeated slots hit cache.
- Saving: ~0–2 ms likely (generic readahead was already noise, −0.04 ms).
- Fidelity: exact.
- Surface: selected-address path (ds4_metal.m:8878); batch addr call (ds4_metal.m:25570).
- Probe: in the harness, cold-sweep baseline M=1×2 vs union-prefetched M=1×2 vs disjoint-selection M=1×2. GO only if prefetch moves cold cost materially. (Overlaps with #1's probe (b).)
- Occupancy: none; bandwidth/cache-capacity limited.

## Recommended iteration order (codex)
- **First: the cheap no-kernel probes** — #1's probe(b) [union-prefetch-then-M=1×2 cold] + #2
  [top-r budgeting via existing batch kernel] + #5. These retire branches with zero kernel work;
  #1's probe(b) is decisive for whether the de-dup is capturable without a new kernel.
- **Then: the main kernel — #1** (expert-major load sharing, **sequential** per-token compute), and
  if it lands, compose with #3 (down de-dup) + #4 (cost-aware STS).
- **Never:** another simultaneous M=2-style 2×-accumulator kernel.

## Mapping to the 3 goal iterations (provisional; adjust per probe results)
- **iteration-1:** the no-kernel probe batch — #1-probe(b) union-prefetch cold + #2 top-r budgeting
  (existing kernel) + #5 prefetch. Retire/advance the cheap branches; decide if the de-dup is
  capturable without a new kernel.
- **iteration-2:** the main kernel — #1 sequential-tile grouped MoE (if iter-1 didn't already win
  via prefetch), composed toward ≥20%.
- **iteration-3:** #3 down de-dup and/or #4 cost-aware STS composition (or a refinement of #1),
  toward clearing the +20% gate.
