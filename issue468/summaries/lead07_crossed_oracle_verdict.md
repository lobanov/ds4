# Lead 07 — Crossed FP/IQ2 oracle: VERDICT

Date: 2026-07-16. Status: **PIVOT — closed negative.** (codex gate pending)

Experiment 1 (`issue468/pending/lead_07_upstream_quality_ceiling.md`) is complete. The
crossed 2×2 FP/IQ2 oracle, run on a **teacher-forced common trajectory** (the IQ2XXS-vs-native
greedy trajectories diverge — ~6% token agreement — so the doc's "common-prefix" alignment
does not exist; it was created by teacher-forcing the IQ2 model onto the FP trajectory via
a new `teacher_force` ds4-spec-bench submode, commit bb9c01f).

## Setup

- 59 prompts (lead3_corpus, codealpaca/dolly/jsonex × 0080–0099; 1 skipped — too few anchors
  for the block), 5554 anchors, common FP trajectory.
- D_f32 torch drafter. Anchors always = Y_fp (the FP-trajectory context tokens).
- H_fp/Y_fp from the retained Lead 04 Modal captures (NOT re-captured); H_iq2_tf/Y_iq2_tf
  from the teacher-force re-capture (the IQ2 model driven through Y_fp).
- Validated: ds4 prompt length == FP prompt_tokens for 0/60 differing (clean alignment);
  Y_iq2_tf==Y_fp at 92.5% on the common trajectory (per-step label-drift is 7.5%; the ~94%
  full-trajectory divergence is compounding, not large per-step drift); fidelity gate PASS
  (H_iq2_tf reproduces sane p1).

## The 2×2 (common FP trajectory)

| cell | p=1 | E[a\|4] |
|---|---|---|
| ceiling   A(D_f32, H_fp,     Y_fp )    | **0.849** | **2.786** |
| baseline  A(D_f32, H_iq2_tf, Y_iq2_tf) | 0.842 | 2.660 |
| hidden    A(D_f32, H_fp,     Y_iq2_tf) | 0.835 | 2.666 |
| label     A(D_f32, H_iq2_tf, Y_fp )    | 0.833 | 2.660 |

Attribution (prompt-clustered bootstrap, n=59):
- **lift (ceiling − baseline) p1 = +0.007, CI[-0.002, +0.016]** — includes 0. Negligible.
- hidden-side main effect p1 = +0.005; label-side main effect p1 = +0.002 (both ~0).
- interaction p1 = +0.023; **at E[a|4] the interaction dominates**: ceiling 2.786 is an
  outlier — crossing *either* H or Y drops E[a|4] to ~2.66.
- IQ2-native reference (deployable IQ2 trajectory) p1 = 0.792.

## Why PIVOT (neither) — the recoverable hidden-side effect is ~0

1. **On a common trajectory, FP and IQ2 hiddens are equivalent for the drafter.** The p1 lift
   (ceiling−baseline) is +0.007 (CI includes 0). The hidden-side *main* effect is +0.005. There
   is no hidden-precision gap to recover.

2. **The recoverable hidden-side effect (the deployable case) is ~0 in both metrics.** The
   deployable path uses the IQ2 target's labels (Y_iq2), not the FP target's. The cell that
   isolates "FP hidden with deployable IQ2 labels" is the hidden cell:
   hidden − baseline = 0.835 − 0.842 = **−0.007 p1**; E[a|4] 2.666 − 2.660 = **+0.006**.
   Making the IQ2 hidden FP-like, while judging against IQ2 labels, yields nothing.

3. **The FP ceiling's block advantage requires the FP target's labels (undeployable).** The
   ceiling E[a|4] (2.786) is high only when *both* H_fp and Y_fp are FP (the self-consistent
   diagonal). It is an **interaction / self-consistency** effect, not a hidden-side main
   effect — and Y_fp (the native-FP argmax) cannot be produced by the IQ2XXS target. So the
   +0.126 E[a|4] ceiling−baseline gap is not reachable on the local path.

4. **The native-vs-IQ2 p1 gap is a trajectory/context effect, not hidden precision.** On the
   common trajectory ceiling (0.849) ≈ baseline (0.842). The gap to the deployable IQ2-native
   (0.792) is baseline − IQ2-native = **+0.050** — i.e. the IQ2 model, run freely, generates a
   *harder-to-draft* trajectory than the FP model. That is a target-trajectory/distribution
   property, not a drafter hidden-input deficiency, and it is not recoverable by hidden
   adaptation.

## Decision

**PIVOT — the native-vs-IQ2 acceptance gain is NOT a recoverable hidden-side effect.** Per
Lead 07's rule ("neither cell much better than IQ2 baseline → stop"), Experiment 2 (the
body-side adapter recovery test) is **not warranted**: there is no hidden-side lever to
recover (the recoverable hidden-side effect is ~0; the ceiling requires the undeployable FP
target labels; the remaining gap is target-trajectory difficulty). **Lead 07 closes negative.**

## What this resolves / leaves open

- **Resolved:** the "native-hidden ceiling" from Lead 04 is NOT an IQ2XXS-recoverable
  hidden-precision effect. The drafter is not hidden-input-limited relative to FP on a common
  context. The +5–15% native ceiling is a target-trajectory + FP-self-consistency artifact,
  not a deployable lever.
- **Still open (other leads):** the *runtime* +20% remains Lead 08 (fused verify kernel —
  verify dominates the cycle). The *target-trajectory difficulty* (IQ2 generates harder-to-draft
  continuations than FP) is a real ~5pp p1 effect but is a target/quantization property, not a
  drafter-fixable hidden-side lever.

## Artifacts

- `issue468/artifacts/lead07_crossed_oracle/`: run_crossed_oracle.py, crossed_oracle_result.json
  (cells, attribution, per-source, per-prompt rows), fidelity_gate.py, prep/verify scripts,
  recapture_result.jsonl. Raw captures (fp_captures_raw/, tf_dump_full/) gitignored
  (reproducible from Modal + ds4-spec-bench teacher_force).
- ds4-spec-bench `teacher_force` submode: commit bb9c01f.
