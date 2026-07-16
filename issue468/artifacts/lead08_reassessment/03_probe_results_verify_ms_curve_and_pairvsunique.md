# Lead 08 re-assessment — sub-lead 1 probe results (measurements 1 + 2): the verify_ms(K) curve + the pair-vs-unique fork

Date: 2026-07-16. Goal: `mrmkwnp6-6n9z9x`. Method: fixed-K sweep (DS4_DSPARK_VERIFY_K=K, STS bypassed) on a 5-entry subset of c_spec_fixed.jsonl, full-stack env (batched + anchor-reuse + prefix-ckp + Metal drafter), `DS4_MTP_VERIFY_PROFILE=1` (accurate total — NO per-stage sync inflation) + `DS4_MTP_VERIFY_EXPERT_PROFILE=1`. Artifacts: `/tmp/lead08_sweep/k{2,3,4,5}.err`.

## The verify_ms(K) curve (layer_execute is ~93% of the verify)

| K | n | layer_execute (ms) | encode | verify≈ | union GiB | avg_unique | pairs (K×6) | physical GiB | de-dup ceiling |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 179 | 37.76 | 2.25 | 41.5 | 3.34 | 11.8 | 12 | 3.40 | **1.9%** |
| 3 | 147 | 48.26 | 2.62 | 52.4 | 4.23 | 14.9 | 18 | 5.10 | 17.2% |
| 4 | 134 | 54.95 | 2.67 | 59.1 | 4.86 | 17.1 | 24 | 6.80 | **28.5%** |
| 5 | 116 | 63.85 | 3.14 | 68.5 | 5.33 | 18.8 | 30 | 8.50 | **37.4%** |

- `selected` GiB = avg_unique × 0.284 GiB/expert → **confirms the logged GiB is the UNION bytes** (codex), not the physical per-pair traffic.
- The **de-dup ceiling grows with K** (1.9% → 37.4%): at K=2 the two tokens' experts barely overlap; at K=4–5 the overlap is high. So load-sharing is worth more at high K.
- K=4 verify≈59 ms matches the live full-stack 62 ms baseline (the small diff is subset/cycle-context). ✓

## THE DECISIVE FORK — pair-vs-unique correlation (measurement 2)

The marginal cost per K vs the marginal bytes:

| step | Δlayer_exec | Δunion GiB | Δphysical GiB | predicted if ∝union | predicted if ∝physical |
|---|---|---|---|---|---|
| K2→K3 | +10.50 ms | +0.89 | +1.70 | +4.4 ms | +8.4 ms |
| K3→K4 | +6.68 ms | +0.63 | +1.70 | +3.1 ms | +8.4 ms |
| K4→K5 | **+8.91 ms** | **+0.47** | **+1.70** | **+2.3 ms** | **+8.4 ms** ✓ |

**At K4→K5: the measured +8.91 ms matches the PHYSICAL prediction (+8.4 ms), NOT the union prediction (+2.3 ms).** The cost tracks **physical pairs**, not unique-union bytes. → **The redundant per-(token,expert) expert loads are DRAM cache-misses (not L2 hits)** → **de-dup (load-sharing) is a REAL DRAM-bandwidth saving, not just a dequant-compute saving.** This reverses codex's cache-hit worry.

Linear fit: `layer_execute ≈ 21 ms (intercept) + 4.92 ms/GiB × physical_GiB` → effective physical bandwidth **~208 GB/s** (vs decode ~410–450). The expert (physical-pairs) portion at K=4 ≈ 6.80 × 4.92 ≈ **33.5 ms**; non-expert (dense + attention + KV + fixed) ≈ 21 ms.

## The prize estimate (mechanism 1, de-dup — at K=4)

De-dup removes the redundant physical load (28.5% of the expert physical at K=4 = 1.94 GiB):
- saving ≈ 1.94 GiB × 4.92 ms/GiB ≈ **~9.5 ms off layer_execute** → verify 59 → ~49.5 ms.
- end-to-end: cycle 69.38 → ~59.9 ms → **+15.8% over the current full stack → ~+21.5% over plain (clears +20%).**

**Mechanism 1 (de-dup) ALONE projects to clear +20%**, IF the fused kernel achieves the physical→union reduction at the same ~208 GB/s. Mechanism 2 (coalesced access → decode bw ~410–450) is ADDITIONAL headroom (the ~208 GB/s is ~half of decode) — could push toward ~16 ms / +36%.

## Verdict signal: GO for the single-stage expert prototype

- The de-dup is real (DRAM cache-miss confirmed) + the prize (~9.5 ms at K=4) clears +20% on its own.
- The escalation bar is met: routed-expert is ~33.5 ms isolated (the linear-fit expert portion) AND the cost tracks physical pairs (de-dup real).

## Caveats (for codex gate A)
1. The bandwidth attribution per-step is inconsistent (162/254/191 GB/s — codex's valid point); the ~208 GB/s / 4.92 ms/GiB are averages. The real de-dup saving depends on the fused kernel's actual achieved bandwidth.
2. The intercept (21 ms non-expert) is a K→0 extrapolation; the attention/dense split within it is unmeasured (the single-layer stage profile, running separately, gives the proportions). This bounds the FULL-build prize, not the single-stage GO.
3. The prize assumes the fused kernel runs the union bytes at the same ~208 GB/s with no offsetting overhead (the union-expert dispatch + shared dequant). The prototype + cost-gate settle this.

## Still open (deferred unless gate A requests)
- **Measurement 1 completeness — the in-tree stage profile is UNRELIABLE.** The single-layer stage profile (layer 40, K=4) returned attn=96% / ffn=4% but **misses the routed experts entirely** — they're dispatched via the SSD-streaming selected-expert path, outside the in-layer `attn`/`ffn` boundaries (DS4_METAL_MOE_ONE_STAGE_PROFILE didn't fire on the batch path either). Confirms codex's skepticism of in-tree stage profiling. **Use the linear fit instead** (expert ~33.5 ms, non-expert ~21 ms at K=4) — that's the reliable decomposition. The attention/dense split within the 21 ms non-expert stays unmeasured (it bounds the FULL-build prize, not the single-stage GO).
- Measurement 3 (Instruments L2 hit rate): the pair-vs-unique correlation already settled cache-miss decisively; Instruments would confirm + give the coalescing headroom magnitude.
- Measurement 4 (readahead/overlap flag sweep): a cheap parallel avenue (could win ms without a fused kernel).
- Measurement 5 (exactness margin probe): for the exactness secondary.
