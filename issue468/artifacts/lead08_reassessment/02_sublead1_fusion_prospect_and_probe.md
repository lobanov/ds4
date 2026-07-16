# Lead 08 re-assessment — sub-lead 1: the verify-cost-decomposition + fusion-prospect probe

Date: 2026-07-15. Goal: `mrmkwnp6-6n9z9x`. Status: **LOCKED as the first sub-lead / the decisive falsification gate.**
Supersedes the "marginal" characterization in `01_reorient` §3 (that was wrong — see the corrected stacking math below).

## 0. Corrected framing — a verify saving is amplified ~1:1 into throughput

The verify is **89.5% of the cycle** (62.07 ms of 69.38 ms total, full corpus). So a verify saving X translates to a throughput gain of **X/(69.38−X)** over the current full stack, stacking on the existing +4.9%:

| verify saving X | over current full stack | vs **plain** (38.16) |
|---|---|---|
| 5 ms | +7.8% | +13.1% (43.2 t/s) |
| 8 ms | +13.0% | +18.6% (45.3 t/s) |
| **10 ms** | **+16.8%** | **+22.6% (46.8 t/s)** ← clears **+20%** |
| 12 ms | +20.9% | +26.8% (48.4 t/s) |
| 16 ms | +30.0% | +36.3% (52.0 t/s) |

**Target number for the whole lead: ~10 ms off the verify = +20% over plain.** This is NOT marginal — the prior "8–18% of the verify = marginal" conflated % of verify with % of cycle. Corrected.

## 1. The fusion prospect — two mechanisms (the second is bigger)

Traffic is **weight-dominated** (activation/intermediate round-trips ~0.5 MB/token/layer, negligible vs ~GB weights). So "reuse data in registers/cache" means the **dequanted expert weights**, not activations. Two distinct fusion mechanisms:

1. **Dequant-once-apply-to-all-tokens** (register reuse of weights across the K tokens) — the load-sharing. Saves redundant dequant (compute) + redundant fetch. **~5–8 ms if the path is partly dequant-bound.** Confirmed real: the batch expert kernel `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` (moe.metal:1257) dequants per-(token,expert) — two tokens selecting the same expert re-dequant independently.
2. **Coalesced union access → decode bandwidth** (the bigger lever). The verify expert stream runs at **~190 GB/s vs decode's ~410–450 GB/s (63%)** — likely the scattered per-(token,expert) access pattern (poor coalescing). A fused kernel processing each union expert once (weights resident in registers/L1 across the token applications) is coalesced → could reach decode bw. Expert stream ~28 ms @ 190 GB/s → ~12.5 ms @ decode bw → **~16 ms recoverable.**

**Expert-fusion prize: ~8–16 ms (mechanism 1 + 2).** At 10 ms → clears +20%; at 16 ms → +36% over plain. **The single-stage expert prototype could clear +20% on its own — IF the ~190 GB/s gap is recoverable, not fundamental.**

## 2. The crux — why is the verify at ~190 GB/s? (three live hypotheses)

| hypothesis | implication | prize |
|---|---|---|
| scattered access (coalescing) | fixable by a coalesced fused kernel | ~16 ms bandwidth (mechanism 2) |
| dequant-compute-bound (the "190 GB/s" is a misleading effective number) | the prize is dequant load-sharing | ~5–8 ms compute (mechanism 1) |
| fundamental (batch-attn scattered KV / inherent dequant) | smaller / needs the full build | TBD |

## 3. Sub-lead 1 — the 4 probe measurements (NO novel kernel code)

1. **Synthetic contiguous union-expert load microbench** (stream the union expert bytes coalesced, no compute): does it hit ~410–450 GB/s? **If yes → access pattern is the culprit → the ~16 ms prize is real → strong GO signal.** This is the single most important measurement.
2. **Compute(dequant)-vs-bandwidth split** of the expert matmul — GPU counters (ALU vs memory utilization), or real-weights vs cached/fake-weights timing. Sizes mechanism 1 vs 2 + tests the "190 GB/s is misleading" hypothesis.
3. **Expert overlap / union size at verify_n≈3.78** (instrumentation already logs selected GiB; derive union vs per-(token,expert)) — sets the load-sharing ceiling (the redundancy fraction).
4. **Attention + dense share of the verify + improvability** — bounds how much of the remaining ~34 ms the full build could touch (the batch attention's scattered KV is the expected hard part).

## 4. Decision rule (feeds task 3 / codex gate A)

- **GO (escalate to the single-stage expert prototype):** probe 1 hits decode bw (≥~380 GB/s) OR probe 2 shows meaningful dequant-compute that load-sharing removes → the ~8–16 ms prize is plausible → build it.
- **NO-GO (skip the prototype, record falsified-on-cost):** probe 1 caps well below decode bw (~<250 GB/s) AND probe 2 shows the path is bandwidth-bound with no recoverable redundancy → the ~190 GB/s is fundamental → the expert fusion can't beat the batch → cost viability falsified. (Exactness-viability then becomes the only remaining question, as a separate sub-lead.)

Carries forward: probe artifacts under `issue468/artifacts/lead08_reassessment/02_*`.
