# Lead 08 — Fused low-K batch-verify kernel (close the verify-vs-floor gap)

Date: 2026-07-07 (refreshed 2026-07-12). Status: **Phase A resolved 2026-07-11
→ Phase B active** (the fused-kernel work; this doc stays in `pending/` as the
active lead). Phase A result:
`issue468/summaries/mtp_verifier_engineering_and_phaseA.md`.
**Two-phase: a cheap profiling gate first (DONE, passed), kernel work only if
the gate passes (it did — proceed to Phase B).**
Related to lead 06 but a distinct thesis: the verifier is above its *own*
bandwidth floor, independent of the redundant anchor decode.

> **2026-07-12 refresh.** The milestone-2 model-on-Metal validation confirmed
> this lead is the **swing term**: recovering the ~19 ms verify headroom flips
> the model from 0.98× (below baseline) to ~1.26× (clears the +20% gate) at
> oracle acceptance with the runtime stack. The crossover measurement showed
> the *existing* batched primitive is only break-even, so a new fused kernel is
> required (not just `verify_suffix_tops`); the exactness measurement quantified
> the batch-vs-decode divergence (small, near-tie); and `decode2_exact` was
> ruled out as linear. These are folded into the rationale, Phase B item 6, and
> the success criteria below.

## Rationale

The bandwidth audit's verdict ("memory-bandwidth-bound") describes the binding
regime, not that the implementation achieves the bound. The dossier's own
measurements say it does not, by ~1.5–2×:

- **Wrong intercept.** Fitting the long-bench medians (K=3..6: 59.7/65.8/74.5/
  79.6 ms) gives verify(K) ≈ **40 ms fixed + 6.6 ms/K**. The slope is roughly
  the expert-union byte growth (physical). The fixed component is not: a verify
  pass's mandatory one-time traffic (one dense stream, ~16–20 ms of the 26 ms
  decode) should put the intercept *below* a decode. Decode itself sits at its
  achievable floor (~300 GB/s effective, tuned streaming/readahead); the batch
  verify path (self-described "production-shaped verifier *attempt*",
  `ds4.c:21117`) carries ~15–20 ms/cycle that is not bytes.
- **The floor is demonstrably reachable in-tree.** verify(K=2) measured at
  **27.3 ms** (code_4k bench, ≈1.05× decode — at the floor) and at **49.5 ms**
  (8k bench, decode2-exact path streaming weights twice by design). Same K, ~2×
  apart by code path: implementation, not physics, sets the current cost.
- M=2–6 is the untuned valley between the optimized decode path (M=1) and the
  compute-bound prefill path (363 t/s at large M).

What a fused kernel buys: dequant-once-apply-to-all-K (weight block stays in
registers/threadgroup memory across the M loop), collapsed per-layer
encode/launch/sync (61 layers × per-stage encodes), on-GPU argmax compare (no
per-position readback stalls), expert readahead overlapped with dense compute
as the decode path already does. What it cannot buy: the dense stream itself
and the expert-union bytes (the ~6.6 ms/K slope).

**Prize at current acceptance** (E[a|4]+1=3.175, S(4)=0.288, draft 10, decode 26),
fixed K=4:

| verify(4) | anchor-reuse accounting | shipped accounting |
|---:|---:|---:|
| 65.8 (today) | −0.9% | −18.9% |
| 55 | +14% | −4% |
| 45 | +32% | +7% |
| ~35 (byte floor) | +57% | +16% |

Three implications: (a) a floor-level verifier **revises the "acceptance-limited"
conclusion** — at verify(4) ≈ 40–45 ms the +20% primary gate clears at *current*
acceptance; (b) it is partially an *alternative* to anchor reuse, not only a
complement — even under shipped accounting a floor-level verify approaches the
gate, which matters if lead 01 falsifies reuse. A cheaper/flatter verify curve
also changes lead 02's scheduling economics (longer blocks cheaper, pruning less
valuable) — re-run that simulation against any new curve. (c) **The existing
batched primitive is insufficient on its own** (confirmed 2026-07-12): the
crossover measurement found `verify_suffix_tops` (66 ms at K=4) is only
break-even vs the short-circuiting sequential verify (~57 ms at model acceptance
E[a|4]≈2.2), because the sequential path stops at the first mismatch. So the
fused kernel's prize is hitting **~45 ms — below the sequential ~57 ms**, not
merely below the current 66 ms. That is why Phase B (a new kernel) is needed
rather than just reusing `verify_suffix_tops`.

## Content of work

**Phase A — profiling gate (1–2 days, no kernel code, start anytime):** DONE.

1. Split the verify cycle per stage with `DS4_METAL_GRAPH_TOKEN_PROFILE=1` /
   `DS4_METAL_LAYER_STAGE_PROFILE` (encode vs execute vs readback), K=2..6,
   same protocol as `run_mtp_verifier_bench_long.py`.
2. Capture achieved GB/s during verify vs decode (Instruments / GPU counters).
3. Log per-cycle expert-union sizes (shared instrumentation with lead 05) and
   compute the true byte floor: dense-once + union-experts + KV at the measured
   effective bandwidth.
4. Deliverable: a measured headroom number — `verify_ms(K) − floor_ms(K)` — and
   its decomposition (encode overhead / bandwidth inefficiency / readback).
   *(2026-07-12 note: Phase A delivered the headroom number — 19–22 ms/cycle at
   K=3..5 — but the encode-vs-bandwidth-vs-readback decomposition is still
   partially open; useful to finish for targeting Phase B, but the 19 ms number
   alone sets the target.)*

**Phase B — kernel work (weeks, Metal-specific):**

5. Fused dequant+GEMM micro-batch kernels (M=2..6) for the dense Q8_0 and
   IQ2_XXS expert paths, M-inner-loop over resident weight tiles; single
   command buffer per layer group; on-GPU suffix argmax compare.
6. **Exactness strategy — the actual hard part.** The fused kernel must be
   **both sublinear AND greedy-exact** — neither existing primitive qualifies:
   `verify_suffix_tops` is sublinear but flips greedy tokens, and
   `metal_graph_verify_decode2_exact` is exact but **linear** (runs two full
   decodes, ~2× decode, no amortization — correctness-only, not a verifier).
   Either (preferred) one kernel family serving both decode (M=1) and verify
   (M=K), making spec output equal target-only output *by construction*; or a
   margin-guarded fallback (re-verify near-ties with the exact path). The
   **2026-07-12 exactness measurement**
   (`issue468/artifacts/rejection_acceptance/verify_dist_probe_exactness.jsonl`)
   quantified the batch-vs-decode divergence: median TV 0.0035, **argmax flip
   rate 0.64%** (1/156 positions) — so the non-exactness is *small and
   concentrated on near-ties*, which both **validates the margin-guarded
   fallback as cheap** (it would trigger on only ~0.64% of positions) and lets
   the top1−top2 margin threshold (Q4-ceiling estimate ≲ ~0.5 logits) be **set
   empirically** from the measured flip distribution rather than assumed. Note:
   the single-family option changes baseline decode numerics; the exactness
   gate (spec == target-only, same build) still holds, but re-baseline the t/s
   denominator.
7. Measure with the retained bench protocol; feed the new verify curve back
   into the lead 02 simulation and the speedup model.

## Success criteria

- **Phase A gate (MET):** measured headroom at K=3..5 ≥ ~15 ms/cycle (verify
  demonstrably ≥1.4× above its byte floor). Came in at 19–22 ms/cycle → proceed.
- **Phase B target:** verify(4) ≤ ~45 ms on the 8k corpus (from 65.8), K=2
  batch-exact ≤ ~30 ms (from 49.5), with exact greedy output preserved on the
  full exactness corpus (hard requirement).
- **Net effect:** fixed-K=4 ≥ +25% with anchor reuse (or ≥ +5% under shipped
  accounting if lead 01 falsified reuse) at current acceptance — i.e., the
  kernel moves the primary gate from "needs a better drafter" to "needs no
  acceptance improvement." It still requires the **concrete runtime stack**
  (anchor reuse + GPU drafter, both engineering, not speculative) but **not** a
  better drafter / higher acceptance. The 2026-07-12 swing-term re-derivation
  independently confirms: verify(4) 66→47 ms flips the model 0.98×→~1.26×.
- **Abort condition:** exactness cannot be preserved by either strategy without
  reintroducing per-position exact re-verification on >~5% of positions (eating
  the gain) — record as the numerics falsification of the fused approach.

## Worklog

### 2026-07-12 — doc refreshed for Phase B launch

Tidied this lead doc for Phase B: corrected the stale "resolved & archived"
header to "Phase A resolved → Phase B active"; folded in the milestone-2 session
evidence — the crossover finding (existing batched primitive is only break-even,
so a fused kernel must hit ~45 ms, below the sequential ~57 ms, not just below
the current 66 ms), the exactness measurement (median TV 0.0035, flip 0.64% →
margin-guarded fallback is cheap and the margin threshold can be set
empirically), and the `decode2_exact`-is-linear ruling (the fused kernel must be
both sublinear AND exact). Tightened the net-effect criterion to make the
dependency stack explicit (needs anchor reuse + GPU drafter, not a better
drafter). Core thesis and prize estimate unchanged and independently
re-confirmed by the swing-term re-derivation.

### 2026-07-11 — Phase A profiling gate PASSED; recommendation: proceed to Phase B

Implemented retained `DS4_MTP_VERIFY_PROFILE` instrumentation and measured the shipped
verifier on the long-context corpus for K=3..5. The resulting artifact
(`artifacts/mtp_phaseA_profile/summary.json`) shows verifier wall time dominated by layer
execution, not host readback: for K=4, verify medians were ~65.9/66.5/67.1 ms while the
initial layer-execute medians alone were ~62.6/63.0/64.0 ms on
`code_8k`/`synthesis_8k`/`grounded_8k`.

Selected routed-expert bytes at K=4 were only ~4.8–5.4 GiB against a full-routed
72.56 GiB layer set, and the measured `verify_ms(K) - floor_ms(K)` headroom remained about
19–22 ms/cycle at K=3..5. That clears the lead's `~15 ms` proceed gate. Recommendation:
**proceed to Phase B fused low-K kernel work** if verifier acceleration remains a live
research path. Canonical summary:
`issue468/summaries/mtp_verifier_engineering_and_phaseA.md`.
