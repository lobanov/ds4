# Lead 02 — Confidence-scheduled verification (local, single-request)

Date: 2026-07-07. Status: pending. Depends on: lead 01 for the anchor-reuse
accounting (the shipped-verifier variant of this lead is still informative if
lead 01 falsifies, but only reaches ~break-even).

## Rationale

Scheduled verification is the third component named in `GOAL.md` and the only one
never explored. The DSpark confidence head exists in the vendored GGUF
(`mtp.2.confidence_head.proj.weight`) and `dspark_oracle/forward.py` already
computes its scores ("for completeness") — no experiment has ever used them.

Two measured facts make per-cycle scheduling valuable *locally*, independent of
the paper's batch-capacity argument:

- **First-draft misses are the largest single leakage**: 17–50% of cycles pay
  draft + verify(K) and accept zero drafts (`summaries/mtp_verifier_bench_results.md`).
  Under anchor-reuse accounting that is ~76 ms for one token vs 26 ms to decode it.
- **Verify is not flat in K** (~7–9 ms/position: 43.6/59.7/65.8/74.5 ms for
  K=2..5), so pruning doomed suffix positions has a real price attached. And
  under anchor reuse, rejection is the *cheap* outcome (correction token for
  free) while full acceptance forces a 26 ms decode — so the optimal length given
  expected survival `a` is ℓ = a+1 (harvest the correction).

**Oracle ceiling from retained numbers** (temp-0 prefix histogram
P(a=0..5) = .1875/.1625/.225/.1375/.125/.1625; verify_ms from the long bench;
draft=10, decode=26): an oracle scheduler yields 21.0 ms/token = **+23.8%** vs
baseline — the only configuration modeled anywhere in this dossier that clears
the +20% primary gate with the current drafter. Best fixed-K is −0.9%. Under
shipped-verifier accounting the oracle reaches only ~break-even (vs −18.9%
fixed), which is why lead 01 gates the headline. The realized number lands
somewhere in [−0.9%, +23.8%] depending entirely on the head's per-cycle
discrimination — the one unknown, and it is measurable offline.

Known risk: the head was trained/calibrated against the FP teacher; on the
IQ2XXS target its calibration is likely off (fixable — STS is a 1-D grid search)
and its discrimination may degrade (the killer; measured in step 1). Fallback if
the head is weak: the drafter's top1−top2 margin as a confidence proxy (Stage 0
showed misses concentrate at small margins).

## Content of work

All offline against retained artifacts; no ds4 engineering:

1. **Measure the head.** Extract per-position confidence scores across the 30
   bundles; label with actual accept/reject; report AUC and ECE per position;
   recalibrate with sequential temperature scaling on a held-out split.
2. **Per-cycle simulation.** Extend `model_spec_speedup.py` from aggregate S(K)
   to per-cycle replay: each bundle cycle scheduled by a threshold policy on the
   recalibrated cumulative confidence (skip-verify gate + adaptive ℓ, including
   the ℓ = â+1 correction-harvest rule), under both verifier accountings.
3. Report the realized-policy speedup with per-prompt spread, alongside the
   fixed-K and oracle bounds. Retain artifacts + summary in dossier format.

Estimated effort: ~2–4 days.

## Success criteria

- **Realized modeled speedup ≥ ~+10%** (anchor-reuse accounting): scheduling
  becomes a top engineering priority alongside lead 05, and the primary gate is
  live again on the current in-RAM setup.
- **+5–10%**: keep as a stacking lever (with fine-tune / lead 05), secondary-gate
  material.
- **< ~+5%**, or head AUC on the Q2 target materially below the paper's 0.81–0.90
  and the margin-proxy fallback also weak: drop the lead and record it as a
  closed direction.
