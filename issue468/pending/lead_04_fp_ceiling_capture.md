# Lead 04 — Served-precision acceptance ceiling (drafter vs native FP4/FP8 target)

Date: 2026-07-08. Status: pending. **Conditional purchase — trigger on lead 07's
PoC landing flat or ambiguous** (if the PoC clears +3–5 pp p=1 on its own,
realized gains trump mechanism and this drops to nice-to-have). Prep work
(capture script + validation plan) is local and free; start anytime. Requires
GPU rental — the target cannot run on local hardware.

## Rationale (short — see prior work for the full argument)

Measure the vendored DSpark drafter's acceptance against the **native release
checkpoint** of DeepSeek-V4-Flash (284B total / 13B active; ~158 GB on disk,
MoE experts natively FP4, rest FP8 — there is no BF16; this *is* the
distillation-time precision, so frame as "served-precision ceiling," not "full
precision"). This is codex's decisive mechanism test, deferred in
`summaries/quant_mismatch_recommendation.md` as "needs a model not on disk":
it resolves what Stage 0/Stage 1 could not — whether the local acceptance
deficit (p=1 0.8125) is IQ2XXS-attributable distribution shift (recoverable →
lead 07 has a defined target) or native drafter quality (fine-tune must beat
the drafter's own training quality — a much weaker bet). It is also the only
defensible interpreter for a flat lead 07 PoC (bad fine-tune vs capacity
limit). Best available prior on the gap size: DFlash pos-1 local 0.68 vs val
0.74 (~6 pp quant penalty) — a similar-order DSpark gap lands exactly where the
fine-tune needs it. No effect on verify-side leads 01/02/08, which stay ahead
in priority.

## Content of work

**Phase A — capture script + validation plan (local, free, ~2–4 days; start
anytime):**

1. Write the FP-side capture driver against the vendored HF `inference/model.py`
   (or vLLM/SGLang if V4-Flash support is confirmed — check first; HF+hooks is
   the fallback, slow but sufficient at capture volumes): greedy decode with
   per-position top-128 logits and layer-40/41/42 hidden-state dumps in the
   exact tap representation the local oracle consumes (mHC HC state [4,4096],
   hc-outer — mirror `run_exactness_small_bundles.py` / `run_dflash_capture.py`
   bundle format so `dspark_oracle/measure_acceptance_bundle.py` runs unchanged).
2. Write the **fidelity validation harness** (the d2t lesson: validate
   algebraically before trusting any number). Checks, all scripted in advance:
   - FP hidden states at layers 40/41/42 correlate strongly with the retained
     Q2 captures on the same prompt prefixes (close but NOT identical — both
     "identical" and "uncorrelated" are red flags);
   - FP greedy tokens agree with the retained Q2 greedy spine at a high rate
     (expected ~90%+; record the disagreement rate — it is deliverable D1);
   - local drafter run on FP-captured hiddens at positions where FP and Q2
     greedy agree reproduces acceptance ≈ the retained local values;
   - embed/lm_head round-trip: drafter logits argmax over FP hiddens lands
     in-vocab and matches an independently computed projection on 2–3 spot
     positions.
3. Choose the corpus: the 10 retained exactness prompts (mandatory, for paired
   comparison) + the widened lead 03 corpus subset if available (target 30–100
   prompts). Same protocol: temp=0, same measure steps, 14+ generated tokens.

**Phase B — rental + capture (1–2 days wall clock, GPU cost low hundreds of $):**

4. Rig: 2×H100/H200 or single 192 GB-class GPU. If lead 06's Stage 2 training
   already rents GPUs, **bundle this capture into the same session** (marginal
   cost ≈ 0; the captured corpus doubles as fine-tune eval data against
   served-precision labels).
5. Run the validation harness first; abort and fix on any red flag before
   burning capture time.
6. Capture, per prompt, in one session:
   - (a) **FP-native trajectory**: FP greedy rollout + hiddens + top-128 — the
     true ceiling protocol, mirroring the local bundles;
   - (b) **teacher-forced Q2 spine**: one pass over the retained Q2 greedy
     spine — yields the FP-vs-Q2 flip rate and the crossed-oracle cells;
   - (c) **teacher-forced drafter blocks**: append the retained drafter rollout
     blocks as extra positions — yields FP-label acceptance of the exact drafts
     already measured locally (paired, per-position).
7. Ship bundles home; all measurement runs locally on the existing oracle.

**Phase C — analysis (local, ~1–2 days):**

8. Headline: FP-ceiling p=1 and E[a|5block] vs the retained IQ2XXS values,
   paired per prompt/step, with bootstrap CIs (template:
   `run_stage1_q4tap_compare.py`).
9. Crossed-oracle decomposition at real precision (the four cells Stage 1's
   Q4-tap proxy could not produce): {Q2, FP} hidden × {Q2, FP} labels →
   input-shift vs label-shift split.
10. Per-position rank diagnostic on the FP side (Stage 0 protocol) — does the
    shallow-miss shape persist against the native target?
11. Summary note + artifacts in dossier format; update
    `quant_mismatch_recommendation.md` with the resolved mechanism.

## Success criteria

Fidelity gates (must pass before any conclusion is admissible):

- Validation harness green: hidden-state correlation in the expected band,
  FP-vs-Q2 greedy agreement ≥ ~85%, drafter-on-FP-hiddens sanity check
  reproduces local acceptance at agreeing positions within noise.
- Paired coverage: every retained exactness cell has its FP counterpart; the
  Q2-side numbers re-derived from the new run match the retained artifacts
  exactly (same fidelity discipline as Stage 0's checks).

Decision outcomes (either is a win — this experiment cannot fail to inform):

- **Gap confirmed** (FP p=1 − Q2 p=1 ≥ ~+4 pp, CI excluding zero; E[a|5block]
  gap consistent in sign): quant-attributable headroom is proven. Lead 07
  scale-up is motivated with the measured gap as its recovery target; a flat
  PoC reads as "fine-tune executed poorly — iterate," not "stop."
- **Gap absent** (≤ ~+1–2 pp, CI tight enough to exclude +4 pp): the deficit is
  native drafter quality. IQ2XXS-specific fine-tuning is deprioritized to
  generic distillation with materially lower expected gain; a flat lead 06 PoC
  then reads as a defensible **stop** for the drafter-quality axis.
- **Intermediate / wide CI**: extend the corpus (Phase B is resumable and
  cheap per prompt) until the ±2 pp band resolves; do not conclude from an
  underpowered gap (the Stage 1 lesson).

Secondary deliverables (recorded regardless of outcome):

- **D1 — target flip rate**: FP-vs-IQ2XXS greedy argmax disagreement rate on
  the spine corpus — the never-measured baseline bounding the entire
  quant-mismatch thread.
- **D2 — crossed-oracle split**: input-shift vs label-shift shares of the
  local deficit, closing Stage 1's open localization question.
- **D3 — corpus difficulty calibration**: FP-native per-position acceptance vs
  the paper's reported DSpark curves, separating "our corpus is hard" from
  "our target is degraded" in all prior and future acceptance comparisons.
