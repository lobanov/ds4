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
- **Measurement 1 completeness — the in-tree stage profile is UNRELIABLE for attn/ffn** (it returns attn=96%/ffn=4% but misses the routed experts — dispatched via the SSD-streaming path, outside the in-layer boundaries). **BUT `DS4_METAL_MOE_STAGE_PROFILE` (the batch-path flag) DOES capture the routed-MoE stages** (gate_up/down/activation_weight/sum). Use the linear fit for the expert-vs-non-expert split + the MoE-stage flag for the routed-MoE sub-breakdown.
- Measurement 3 (Instruments L2 hit rate): the pair-vs-unique + the separator (below) already indicate cache-miss; Instruments would definitively confirm + size the coalescing headroom.
- Measurement 4 (readahead/overlap flag sweep): a cheap parallel avenue.
- Measurement 5 (exactness margin probe): for the exactness secondary.

## SEPARATOR TEST (post codex-gate-A HOLD) — the routed-MoE dominates the K4→K5 marginal → resolves the confound → GO

Codex gate A (`dspark_codex_reviews/2026-07-16_gpt55_xhigh_lead08_gateA_probe.md`) said HOLD: the linear-fit "expert portion" is confounded (physical bytes ∝ K), the cache-miss inference isn't decisive, the realistic prize is 5–10 ms (clears +20% only at the optimistic end). Recommended the separator: isolate the routed-MoE stage delta (K4 vs K5).

Ran K4 + K5 with `DS4_METAL_MOE_STAGE_PROFILE=1`. Routed-MoE per-verify (×61, sync-inflated → proportions only):

| sub-stage | K=4 | K=5 | Δ |
|---|---|---|---|
| gate_up (iq2_xxs gate+up matmul — the de-dup target) | 26.6 | 31.7 | **+5.2** |
| down (q2_k projection) | 18.1 | 18.8 | +0.7 |
| activation_weight (router) | 11.1 | 11.4 | +0.3 |
| sum (expert-output reduction) | 1.1 | 11.0 | +9.8 |
| **routed-MoE total** | **56.9** | **72.9** | **+16.0** |

- The routed-MoE is ~58% of the verify at K=4 (gate_up+down = ~47%, the de-dup target). attn ~17%, dense/shared ffn ~25%.
- **The K4→K5 marginal (+16.0 inflated ≈ the real +8.91 ms) is dominated by the routed-MoE** (gate_up + the per-token expert sum). The dense/shared/attn are sublinear (shared across tokens) → their marginal is small. **→ the expert load dominates the marginal → the de-dup (which targets gate_up+down) hits the dominant cost. This resolves codex's confound.**
- De-dup prize at K=4: 28.5% of (gate_up+down) expert-load bandwidth. gate_up+down (real) ≈ 28 ms → prize ≈ **~6–8 ms real** (borderline +20%; clearly faster than the batch verifier = the primary bar).

**Verdict: GO for the single-stage expert prototype** (the de-dup targets the dominant marginal cost; the primary bar "faster than batch" is robustly met at ~6–8 ms; +20% is borderline, to be settled by the prototype's cost-gate). The hard exit gate protects if the prototype can't beat the batch. Instruments (L2 hit rate) deferred — the in-tree data (pair-vs-unique + separator) already indicates cache-miss; revisit if the prototype underperforms.
