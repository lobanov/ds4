# Lead 08 workstream — iteration-1 (no-kernel probe batch): top-r budgeting

> **2026-07-17 final update:** top-r remains NO-GO. Artifact 10 conditionally advanced one bounded
> threadgroup-spill gate+up prototype; artifact 11 subsequently measured it bit-exact but slower and
> closed that branch. Artifact 12 closes the separate margin fallback.

**Date:** 2026-07-16. **Candidate:** #2 from the lead-ID (MoE-Spec-style verify-time expert budgeting).
**Mechanism:** env-gated `DS4_TOP_R` overrides `g_ds4_shape.n_expert_used` post-load → the router +
batch verify use top-r instead of top-6 (no new kernel; reuses `ds4_gpu_routed_moe_batch_tensor`).
**Probe:** M3 full stack (`VERIFY_BATCHED=1 DRAFT_METAL=1 ANCHOR_REUSE=1 DRAFT_METAL_STS=1
VERIFY_PREFIX_CHECKPOINT=1 TIMING=1`), 5-entry corpus, top-6 vs top-5 vs top-4.

## Results (5-entry corpus, mean over entries; warm target)
| r | verify_ms | tokens/sec | accepted_mean |
|---|-----------|------------|---------------|
| 6 (baseline) | 55.83 | 38.41 | 2.471 |
| 5 | 52.31 (−6.3%) | 35.06 (−8.7%) | 2.112 (−14%) |
| 4 | 53.74 (−3.7%, noisy) | 39.22 (+2.1%) | 2.412 |

(n=5 entries → noisy absolutes, but the shape is clear.)

## Verdict: NO-GO (for ≥20%)
- **Cost benefit is small:** routed experts are only ~40% of verify_ms; top-5 cuts that by ~1/6 →
  ~6% verify_ms max. Even noise-free, that is ~5× short of the ~18% verify_ms cut needed for ≥20%.
- **Fidelity cost (the killer):** top-r changes the target's routing → drafter acceptance drops
  (2.47→2.11 at top-5) → **t/s goes DOWN** (38.4→35.1) despite the lower verify_ms. The output
  changes (the MoE-Spec quality risk, here realized). Would need a ds4-eval no-regression that is
  unlikely to pass given the acceptance drop.
- **Net:** top-r budgeting alone is not viable for ≥20%. (It could compose as a minor knob, but not
  a standalone win.)

## Rest of the no-kernel batch (retired)
- **#5 targeted union prefetch → NO.** The readahead flag was already a no-op (−0.04 ms, the probe);
  the cold-vs-warm sweep shows experts cache only within a tight loop, not for the one-shot-per-cycle
  production verify (between a token's two dispatches the intervening work evicts the shared expert).
  A prefetch cannot keep the union hot → no cheap ordering win.
- **#4 cost-aware STS → deferred.** ~1–4 ms, no quality risk, but not a standalone ≥20% win; keep as
  a composition to layer onto the main kernel if it lands.

## Conclusion / hand-off
The no-kernel probes (#2, #4, #5) do not yield ≥20%. This artifact originally handed off to #1;
artifact 11 now records that kernel's performance NO-GO. Candidate #3 (down de-dup) and #4
(cost-aware STS) were only low-prize compositions if #1 landed, so they do not remain standalone
Lead 08 paths.

## Codex review (gpt-5.5 xhigh, report retained at dspark_codex_reviews/2026-07-16_gpt55_xhigh_lead08_iter1_topr.md)
- **NO-GO confirmed** — traced `DS4_TOP_R` → `g_ds4_shape.n_expert_used` → batch router select →
  `ds4_gpu_routed_moe_batch_tensor` (n_tokens×n_expert physical rows); the −6% matches 0.40×⅙.
- **Two probe caveats (don't flip NO-GO):** (i) for n_expert<6 the direct sum6 fast-path is
  disabled → generic down+sum (not a pure "same kernel, fewer slots"); (ii) early hash-routed
  layers may mis-stride under the override (hash table loaded as width 6, GPU hash path uses
  n_expert_used as row width) → the acceptance drop may be a probe artifact, not clean fidelity
  evidence. NO-GO stands regardless (cost win too small even before quality risk; top-5 t/s down).
- **Iteration-2 design correction:** a naive "sequential-tile" kernel is NOT magically
  occupancy-safe — each token needs accumulators across ALL ib32 tiles. Viable design = spill
  per-token partial accumulators to **threadgroup memory** (register footprint ≈ M=1; shmem
  traffic the cost). Expert-major grid `(row_groups, unique_count)`, threads `(32, NSG=2)`, write
  `mid[token,slot,row]` selection-ordered, reuse the existing per-token down/sum6 unchanged.
- **Measurement-first probe before building:** batch-MoE controlled-selection timing (same K, same
  physical-pair count; all-overlapped vs mostly-disjoint selections, `DS4_METAL_MOE_STAGE_PROFILE`) —
  if equal time, the batch MoE does NOT reuse overlap today → the grouped kernel has real headroom.
- Prize check: 17/24 unique @ K=4 → 28.5% ceiling → ~6–8 ms realistic — enough to beat the batch
  verifier but **not a guaranteed ≥20% clear** (baseline ~59 ms vs gate ≤50.5 ms); needs the high
  end + low overhead + composition (down de-dup / STS).
