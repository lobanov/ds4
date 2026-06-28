# CORRECTED DSpark speedup projection (doc 15 verify-cost error)

Date: 2026-06-28. Supersedes the projection in `15_acceptance_first_results.md`.

## The error in doc 15

`15_acceptance_first_results.md` projected speedup using **verify(L=5) ≈ 20ms**
("interpolating: 36ms at L=1, 12ms/token at L=8"). That interpolation was
arithmetic — the actual Phase-1 measured batch-verifier curve (`06_phase1_results.md` §2)
gives a 4x-larger number. Refreshing from the artifact caught this before the
Metal-port investment was made on a false premise.

## Real measured numbers

From `06_phase1_results.md` §2 (batch `metal_graph_verify_suffix_tops`, the exact
verifier DSpark Phase 5 uses), median ms at ctx 2k/4k/8k:

| L | batch (ms) |
|---|---|
| 1 | 31 / 36 / 36 |
| 2 | 42 / 50 / 50 |
| 3 | 51 / 58 / 57 |
| 4 | 70 / 69 / 69 |
| **5** | **76 / 75 / 75** |
| 8 | 97 / 95 / 96 |

**verify(L=5) = 75 ms** (context-flat — batch is independent of prefix length
because of the SWA window). Marginal cost ~4-8 ms per added token.

Plain decode (measured 2026-06-28, IQ2XXS, Metal, code prompt):
**25.7 ms/token** (38.96 t/s). Matches doc 06 (26ms@2k, 31ms@8k).

DSpark acceptance (doc 15): avg accepted prefix **2.79**, +1 bonus = **3.79
committed tokens per draft cycle**. This was measured at LOW KV fill (2-20
window entries); real decode has up to 128.

## Corrected speedup math

Per speculative cycle (draft then verify):
  cycle_ms = draft_ms + verify(L=5)
  committed_tokens = accepted_prefix + 1   (the +1 is the correction/bonus token)
  speedup = (25.7 × committed_tokens) / cycle_ms

| draft est | verify | prefix | cycle | ms/tok | vs 25.7 plain | speedup | gate (>20%) |
|---|---|---|---|---|---|---|---|
| 15 ms (optim) | 75 | 3.79 | 90 | 23.7 | 8% faster | 1.08x | **FAIL** |
| 30 ms (pessim)| 75 | 3.79 | 105 | 27.7 | slower | 0.93x | FAIL |
| 15 ms | 75 | 3.0 (pessim acpt) | 90 | 30.0 | slower | 0.86x | FAIL |
| **15 ms** | **75** | **5.0 (if KV fill lifts prefix to 4.0)** | **90** | **18.0** | **30% faster** | **1.43x** | **PASS** |
| 15 ms | 75 | 4.0 | 90 | 22.5 | 12% | 1.14x | FAIL |

**Reading:** at the *measured* acceptance (2.79 prefix), the project is **marginal
to negative** — best case ~8%, below the 20% gate. The gate is met ONLY if
average accepted prefix climbs to ~4.0+ at realistic KV fill.

## The swing factor: acceptance at realistic KV fill

Doc 15 explicitly caveated: "Acceptance is strong even with minimal KV cache
(2-20 entries). The real decode scenario has up to 128 window entries, which
should improve acceptance further." This caveat is now load-bearing — it is the
difference between a negative result and a passing one.

**The decisive, cheap measurement:** run the existing numpy oracle's acceptance
test with main_hidden captures at positions where the drafter's window KV is
substantially filled (e.g. positions 200-400, where 48-128 anchors are cached).
This is FAR cheaper than the Metal port (the oracle exists and runs in ~8.5s)
and determines whether the Metal-port investment is justified.

- If avg prefix climbs to ≥4.0 at realistic fill → Metal port justified, project
  on track for terminal outcome A.
- If avg prefix stays ~2.79 at realistic fill → terminal outcome B (documented
  negative result): acceptance too low for >20% even with a free drafter,
  because verify(L=5)=75ms dominates. No Metal port needed to reach the verdict.

## Implication for the plan

Do NOT proceed to the Metal drafter port (phase4-forward Metal) under the old
premise. The next step is the **acceptance-vs-KV-fill sweep** with the numpy
oracle — it is the gate that decides proceed-vs-negative before any large
investment. This re-orders the remaining work toward the cheapest decisive
measurement first.

## Note on draft-cost estimate

The draft cost (15-30ms estimate) is itself unverified. Scaling argument: the
drafter is 3 layers reusing `metal_graph_encode_layer_batch` (the same machinery
the 75ms/43-layer verify drives), so its 3 blocks ≈ 3/43 × 75 ≈ 5ms, plus fixed
overhead (shared lm_head over 5×vocab, embed, HC stages, sequential Markov head).
The fixed overhead likely dominates (~20-25ms). Measuring it precisely requires
the Metal port — but the verify cost (75ms) is the dominant, fixed term and does
NOT depend on the draft estimate. So the swing factor is acceptance, not draft
cost.
