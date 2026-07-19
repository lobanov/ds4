# Lead 11 — Target-confidence draft bypass + STS recalibration (a two-stage scheduler)

Date: 2026-07-19. Status: **proposed (not yet started).**

## Purpose / hypothesis

With the verify-cost lever (Lead 08) currently intractable and the acceptance lever (Lead 10)
trajectory-capped, the remaining structural lever is **scheduling** — deciding *earlier* (pre-draft)
which cycles are worth drafting, using the **target's own confidence** rather than the drafter's.

**Hypothesis (two claims):**

1. **Speed up by bypassing the draft on low-target-confidence cycles.** The target's margin at
   the anchor is available **pre-draft** (it's computed by the previous cycle's verify) and it
   predicts the drafter's first-draft rejection. Bypassing the draft+verify on those cycles
   (plain decode) saves the ~7.6 ms draft cost that the STS is *structurally incapable of*
   recovering — the STS keys on the drafter's `confidence_head`, which is a product of the draft
   forward, so it can only trim `verify_n`, never undo the draft.

2. **Higher acceptance of the drafted prefix.** Bypassing the low-confidence cycles **filters**
   the drafting population to higher-confidence cycles, so the drafter's acceptance on the
   cycles that *do* draft rises (a selection effect). Reaping this requires **recalibrating the
   STS** (its `verify_n` schedule was tuned on the full population and is stale for the filtered
   one) and likely **retraining the confidence model** (its calibration shifts on the filtered
   input distribution).

If the hypothesis holds → **implement the two-stage scheduler in the engine.**

## Why this lead exists (the gap)

- **Lead 08 (fused verify kernel):** intractable for now (attempted in another branch). The
  direct +20% path is blocked.
- **Lead 10 (drafter re-distillation):** trajectory-capped (Lead 07) **and** the gate probe
  (below) shows the drafter's failures are at *hard* (target-uncertain) positions, not fixable
  drafter bugs → negative prior.
- **The STS (M3 lever 5):** post-draft → can only trim `verify_n`; its re-tune was "marginal,
  within noise."
- **The open lever:** a **pre-draft** gate on the *target's* margin (ground-truth difficulty,
  free from the prior verify), composed with a **recalibrated STS** on the filtered cycles. This
  is the one acceptance/scheduling route that doesn't depend on Lead 08 (intractable) or Lead 10
  (trajectory-capped).

## The mechanism (a two-stage scheduler)

1. **Pre-draft gate** — at the start of cycle N+1, read the target's margin at the anchor
   (retained from cycle N's verify). `margin < θ_bypass` → **bypass** (commit the anchor / one
   plain decode; no draft, no verify).
2. **Post-draft STS** — on the non-bypassed cycles, set `verify_n` from the drafter's
   `confidence_head` per a **recalibrated** schedule `θ_sts` (re-tuned for the filtered,
   higher-confidence population).

The two signals are **complementary, not redundant**: the target's ground-truth difficulty acts
early (decides bypass-vs-draft), the drafter's confidence acts late (decides verify depth).
`θ_bypass` and `θ_sts` interact (a more aggressive bypass leaves an even-more-confident drafted
population → a different optimal `verify_n`), so they must be **jointly** tuned.

## What's already established (the gate probe + the IQ2 logprobs capture, 2026-07-19)

- **The target's margin strongly predicts the drafter's first-draft rejection.** On the FP
  captures (`target_confidence_gate_probe.py`, n=5790 anchors): successmedian margin 12.0 vs
  fail-median 1.75; **corr(entropy, success) = −0.44**; a `margin < 0.75` skip rule catches
  TPR=0.25 of rejections at FPR=0.03 (net positive). The signal is real and strong.
- **The IQ2 deployment target's margin distribution matches the FP target's** (mean 8.73 /
  median 8.33 vs FP 9.0 / 8.6) — captured via the new single-jsonl `--dump-logprobs-jsonl` mode
  on the lead3 corpus (`iq2_logprobs.jsonl`, 5811 records). So the FP-side predictiveness should
  carry to the deployment target.
- **Key alignment finding:** the existing lead3 IQ2 H captures do **not** align with this logprobs
  capture (0–3% greedy agreement — different prompt handling), so the gate eval (and Lead 10's
  soft labels) need a **unified H+logprobs capture from one run**. The logprobs half is built +
  working; the consolidated H half is the next implementation step.

## Scope / experiments

- **Exp 0 — measurement (the gate's value).** (a) A **unified IQ2 capture** (H + logprobs in one
  run, aligned — add a single-file appended H dump so the capture stays "no many small files").
  (b) A **cycle-economics simulation**: per cycle (target margin, drafter first-draft success,
  per-position verify cost from the M3 timing model), sweep `(θ_bypass, θ_sts)` → expected t/s.
  Sizes the gain + finds the joint optimum, with no ds4 changes.
- **Exp 1 — the bypass (claim 1).** Implement the pre-draft target-margin gate in ds4 → measure
  the speedup from skipping the draft on bypassed cycles (vs the false-skip cost).
- **Exp 2 — the STS recalibration (claim 2).** Recalibrate `θ_sts` for the filtered population
  + retrain the confidence model (the `conf_proj`/temp, or the confidence head) on the filtered
  distribution → measure the acceptance + throughput gain on the drafted cycles.
- **Exp 3 — the engine implementation (conditional).** If Exp 1+2 clear the bar, land the full
  two-stage scheduler in ds4 (the pre-draft gate + the recalibrated STS), re-bench the M3 stack,
  the 20-Q gate, + the 92Q.

## Decision rule (locked before Exp 1)

- **PROCEED:** the cycle-economics simulation (Exp 0) predicts a combined throughput gain
  **≥ +3%** at the joint `(θ_bypass, θ_sts)` optimum, AND Exp 1+2 realize it (CI excl 0 vs the
  M3 full-stack baseline). → implement in the engine (Exp 3).
- **MARGINAL:** +1–3%, or the gain is almost all from the bypass (claim 1) with no selection-effect
  uplift (claim 2). → land the bypass alone if it's net positive; report honestly; don't claim
  claim 2 without the recalibration proving it.
- **STOP:** ≤ +1%, or the false-skips outweigh the saved drafts, or the recalibration doesn't
  move acceptance. → the scheduling axis is exhausted alongside Lead 08 (intractable) + Lead 10
  (capped); the +20% is unreachable locally.

The bar is deliberately modest (+3%, not +20%): the verify still dominates, so a scheduling gain
is bounded — but with Lead 08 blocked, this is one of the few viable near-term levers, and even
a few % is worth landing.

## Non-goals

- The verify-cost reduction itself (Lead 08 — intractable for now).
- Drafter body re-training for acceptance (Lead 10 — the gate probe shows the failures are at
  hard positions, not drafter bugs).
- A new drafter architecture; expert tuning; the +20% target as a Lead-11 headline (it needs
  Lead 08).
- Changing the committed-output byte-exactness contract (the bypass emits the anchor / a plain
  decode — both exact; the STS recalibration must preserve the score-neutral verify behavior).

## Deliverables

1. The unified IQ2 capture (H + logprobs, aligned, single-file-each) for the lead3 corpus.
2. The cycle-economics simulation (Exp 0) + the joint `(θ_bypass, θ_sts)` optimum + the predicted gain.
3. The pre-draft bypass implementation (Exp 1) + the recalibrated STS / retrained confidence model (Exp 2).
4. The measured throughput gain (vs the M3 full-stack baseline) + the PROCEED/MARGINAL/STOP verdict.
5. If PROCEED: the engine implementation (Exp 3) + the re-bench (20-Q gate, 92Q, full corpus).

## Exit conditions

- **PROCEED:** +3% combined (CI excl 0) → engine implementation + re-bench.
- **STOP:** ≤ +1% or harm → the scheduling axis is exhausted; the +20% is unreachable without Lead 08.

## One-line verdict (proposal)

Lead 11 tests whether a **pre-draft target-margin bypass gate** (using the verifier's own
confidence, available before the draft) composed with a **recalibrated STS** can lift throughput
~3–5% by (1) skipping hopeless drafts and (2) raising the drafted-prefix acceptance — the one
scheduling lever left with Lead 08 blocked and Lead 10 capped. The gate signal is already
established (margin predicts rejection, corr −0.44); the decisive next step is the unified
IQ2 capture + the cycle-economics simulation.

## Worklog

### 2026-07-19 — gate probe + IQ2 logprobs capture (Exp 0 setup, pre-execution)

- **Gate probe on the FP captures** (`target_confidence_gate_probe.py`, n=5790): the target's
  margin/entropy strongly predicts the drafter's first-draft rejection — successmedian margin
  12.0 vs fail-median 1.75, **corr(entropy, success) = −0.44**; `margin<0.75` skip → TPR 0.25 /
  FPR 0.03 (net positive). → the gate signal is real. Also: the failures are at *hard*
  (target-uncertain) positions, not drafter bugs → reinforces Lead 10's negative prior.
- **New `--dump-logprobs-jsonl` ds4-spec-bench mode** (single appended JSONL, one record per
  anchor `{id,pos,sel,top:[[id,logprob]×128]}` — no many small files). Captured the IQ2 target's
  distribution on the lead3 corpus → `iq2_logprobs.jsonl` (5811 records). IQ2 margin distribution
  (mean 8.73 / median 8.33) matches the FP target's → the gate signal should carry to the
  deployment target.
- **Alignment finding:** the existing lead3 IQ2 H captures do NOT align with this logprobs
  capture (0–3% greedy agreement — different prompt handling). → Exp 0 needs a **unified
  H+logprobs capture** (one run, aligned). Open item: add the consolidated single-file H dump
  (mirror the logprobs jsonl pattern) so the capture stays single-file-per-modality.

### 2026-07-19 — orientation + decision rule LOCKED (research-lead skill, steps 1–2)

Oriented (skill principle 1 — verified the assets, didn't trust memory):
- **Lead 11** (this doc) — the two-stage scheduler (pre-draft target-margin bypass + recalibrated STS).
- **Established findings:** gate probe (`target_confidence_gate_probe.py`, FP, n=5790) — target
  margin predicts drafter rejection, **corr(entropy,success)=−0.44**, fail-median margin 1.75 vs
  success 12.0, `margin<0.75` skip → TPR 0.25 / FPR 0.03 (net+). IQ2 logprobs capture
  (`iq2_logprobs.jsonl`, 5811 records) — IQ2 margin (mean 8.73 / median 8.33) **matches FP** →
  the gate signal carries to the deployment target.
- **The M3 baseline (the comparison):** full stack **40.04 t/s**; cycle ≈ decode(0.1ms, folded)
  + draft(7.6ms) + verify(~59–62ms, STS-adapted verify_n~3.8) ≈ 66ms; verify_ms **sublinear**
  (~29@1, ~53@3, ~67@5; ~16ms/token); standalone anchor decode ~28ms; plain ds4 **38.16 t/s**
  (~26.2ms/token).
- **The lead3-misalignment finding stands:** the existing IQ2 H does NOT align with the logprobs
  capture → Exp 0a needs the unified H+logprobs capture (one run).

**Decision rule LOCKED (principle 2):** the gate's value is the **cycle-economics expected t/s**
vs the M3 baseline (40.04 t/s).
- **PROCEED** at **≥ +3% (≥ 41.24 t/s)** at the joint `(θ_bypass, θ_sts)` optimum.
- **MARGINAL** +1–3%.
- **STOP** ≤ +1% or harm (the false-skips outweigh the saved drafts).
The bar is deliberately modest — the verify still dominates; Lead 08 is intractable, Lead 10
is capped. Do NOT headline a Lead-11 gain as the +20% path.

**Estimator LOCKED (principle 3 — the cycle-economics expected-t/s):**
`throughput(θ_bypass, θ_sts) = Σ_cycles tokens(cycle) / Σ_cycles cost(cycle)`, where per cycle:
- if `target_margin < θ_bypass` → **BYPASS**: tokens=1 (the anchor), cost = `plain_decode_ms` ≈ 26.2 ms
  (the standalone anchor forward; from plain ds4's 38.16 t/s).
- else → **DRAFT**: tokens = 1 + `accepted` (the drafter's block acceptance), cost = `draft_ms` (7.6) +
  `verify_ms(verify_n)`, where `verify_n = sts_schedule(drafter_confidence, θ_sts)`.
- Cost parameters (from M3): `draft_ms=7.6`; `verify_ms(verify_n)` = the sublinear M3 curve
  (~29@1, ~53@3, ~67@5); `plain_decode_ms≈26.2`.
- Inputs (from the unified capture + the offline drafter run): per-anchor `(target_margin,
  drafter block-acceptance)`.
- Claim split (the hypothesis): claim 1 (draft-skip) = the bypass saving on reject-early cycles;
  claim 2 (selection effect) = the acceptance uplift on the filtered (drafted) cycles.

**Skill principles confirmed:** orient (✓ this step); lock-the-rule (✓ above); estimator-is-the-result
(✓ the cycle-economics t/s); fidelity-gate (Exp 0a H+logprobs alignment; Exp 1 M3-baseline-reproduces);
two codex gates (Exp 0 + Exp 3); honest scope (✓ the +3% bar); per-source (the full corpus);
single-process memory (one ds4 run at a time).

**Next:** Exp 0a — add the consolidated single-file H dump to ds4-spec-bench + the unified capture
(open implementation question: is `dspark_main_hidden` accessible from the bench for a single-file dump?).
