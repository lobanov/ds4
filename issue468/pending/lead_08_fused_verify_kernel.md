# Lead 08 — Fused low-K batch-verify kernel (close the verify-vs-floor gap)

Date: 2026-07-07. Status: pending. **Two-phase: a cheap profiling gate first
(independent, can start immediately), kernel work only if the gate passes.**
Related to lead 06 but a distinct thesis: the verifier is above its *own*
bandwidth floor, independent of the redundant anchor decode.

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

Two implications: (a) a floor-level verifier **revises the "acceptance-limited"
conclusion** — at verify(4) ≈ 40–45 ms the +20% primary gate clears at *current*
acceptance; (b) it is partially an *alternative* to anchor reuse, not only a
complement — even under shipped accounting a floor-level verify approaches the
gate, which matters if lead 01 falsifies reuse. A cheaper/flatter verify curve
also changes lead 02's scheduling economics (longer blocks cheaper, pruning less
valuable) — re-run that simulation against any new curve.

## Content of work

**Phase A — profiling gate (1–2 days, no kernel code, start anytime):**

1. Split the verify cycle per stage with `DS4_METAL_GRAPH_TOKEN_PROFILE=1` /
   `DS4_METAL_LAYER_STAGE_PROFILE` (encode vs execute vs readback), K=2..6,
   same protocol as `run_mtp_verifier_bench_long.py`.
2. Capture achieved GB/s during verify vs decode (Instruments / GPU counters).
3. Log per-cycle expert-union sizes (shared instrumentation with lead 05) and
   compute the true byte floor: dense-once + union-experts + KV at the measured
   effective bandwidth.
4. Deliverable: a measured headroom number — `verify_ms(K) − floor_ms(K)` — and
   its decomposition (encode overhead / bandwidth inefficiency / readback).

**Phase B — kernel work (weeks, Metal-specific, gated on Phase A):**

5. Fused dequant+GEMM micro-batch kernels (M=2..6) for the dense Q8_0 and
   IQ2_XXS expert paths, M-inner-loop over resident weight tiles; single
   command buffer per layer group; on-GPU suffix argmax compare.
6. **Exactness strategy — the actual hard part.** Either (preferred) one kernel
   family serving both decode (M=1) and verify (M=K), making spec output equal
   target-only output *by construction*; or a margin-guarded fallback
   (re-verify near-ties with the exact path — Q4-ceiling diagnostics showed
   argmax flips need top1−top2 margins ≲ ~0.5 logits). Note: the single-family
   option changes baseline decode numerics; the exactness gate (spec ==
   target-only, same build) still holds, but re-baseline the t/s denominator.
7. Measure with the retained bench protocol; feed the new verify curve back
   into the lead 02 simulation and the speedup model.

## Success criteria

- **Phase A gate:** measured headroom at K=3..5 ≥ ~15 ms/cycle (i.e., verify
  demonstrably ≥1.4× above its byte floor at the achieved decode bandwidth).
  Below that: record the verifier as at-floor, close the lead for ~zero cost —
  itself a valuable hardening of the bandwidth audit.
- **Phase B target:** verify(4) ≤ ~45 ms on the 8k corpus (from 65.8), K=2
  batch-exact ≤ ~30 ms (from 49.5), with exact greedy output preserved on the
  full exactness corpus (hard requirement).
- **Net effect:** fixed-K=4 ≥ +25% with anchor reuse (or ≥ +5% under shipped
  accounting if lead 01 falsified reuse) at current acceptance — i.e., the
  kernel alone moves the primary gate from "needs a better drafter" to "needs
  nothing else."
- **Abort condition:** exactness cannot be preserved by either strategy without
  reintroducing per-position exact re-verification on >~5% of positions (eating
  the gain) — record as the numerics falsification of the fused approach.
