# Lead 08 workstream — iteration-1 (no-kernel probe batch): top-r budgeting

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
The no-kernel probes (#2, #4, #5) do not yield ≥20%. **The path is the main kernel — #1
(union-aware grouped MoE, sequential-tile, single-token register state)** for iteration-2, with #3
(down de-dup) + #4 (cost-aware STS) as composition if it lands. The top-r + prefetch retirements
de-risk iteration-2 (no point chasing ordering/budgeting when the prize needs a load-sharing kernel).
