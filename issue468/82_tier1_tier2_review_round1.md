# Tier 1–2 lead review, round 1 — after Opp-a retry and verify simplification

Date: 2026-07-02.

Purpose: independent review of prior DSpark research/productionization artifacts after:

- Opp-a retry was attempted and measured (`issue468/80`)
- verifier optimization attempt 1 was attempted and measured (`issue468/81`)

This review is limited to **Tier 1–2, code-only, in-scope** leads under the active goal.

## 1. Inputs reviewed

Primary source of truth:
- `issue468/32_research_handoff_note.md`

Key historical notes re-read for this round:
- `issue468/36_p1p4_results_and_perf_blocker.md`
- `issue468/37_block_size_sweep_and_opp1_projection.md`
- `issue468/38_oppa_correction_decode_infeasible.md`
- `issue468/39_perf_gate_lever_consolidation.md`
- `issue468/41_codex_loop_iterations_and_convergence.md`
- `issue468/43_pr482_device_test_report.md`
- `issue468/70_pi_agent_review_full.md`
- `issue468/70_pi_agent_review_leads.md`
- `issue468/78_iteration5_verify_n_sweep.md`
- `issue468/79_iteration5b_timing_breakdown.md`
- `issue468/80_oppa_merge_correction_anchor_investigation.md`
- `issue468/81_verifier_n_simplification_measurement.md`

## 2. What is now firmly established

### 2.1 Opp-a is no longer hypothetical

The old structural objection in `issue468/38` applied to eliminating the correction forward entirely. That is no longer the only formulation.

`issue468/80` demonstrated a viable **merged correction-anchor** path:
- correction decode skipped on partial accept
- correction token carried into next cycle as pending anchor
- next cycle decodes it without re-emitting it

Measured first temp=1 gain:
- `34.69 -> 36.71 t/s`
- accepted drafts/cycle `2.786 -> 3.000`
- logical committed/cycle `4.571 -> 4.812`

Conclusion:
- Opp-a retry is a **real Tier-1 win**, but by itself does not clear baseline.

### 2.2 Simple verifier simplification is not enough

`issue468/81` measured runtime verify-length truncation.

Best result:
- `verify_n=4`: `32.09 t/s`
- verify cost drops to `60.58ms`
- but acceptance/commits fall enough that end-to-end still loses to plain baseline

Conclusion:
- **fewer verified positions** is not the right verifier fix.
- A winning verifier lead must improve the **per-position cost curve**, not just lower `N`.

### 2.3 The gate remains near-miss, not collapse

From `issue468/79` and the new Opp-a measurement:
- verify remains the dominant phase (~70ms)
- Opp-a-style correction elimination clearly matters
- the remaining gap after Opp-a is small enough that a meaningful verifier reduction still has a plausible path to the gate

So the post-Opp-a situation is:
- not solved
- but also not converged to “no possible code-only improvement” yet

## 3. Tier 1–2 lead inventory, re-ranked

### Lead A — custom microbatch verifier (REMAINS VIABLE; highest priority)

Why it remains viable:
- `issue468/32` explicitly identifies the generic batch prefill verifier as the dominant cost center and names a custom microbatch verifier as the main remaining path.
- `issue468/06` showed the exact-fused N=2 verifier was slow because it was not real fusion; that does **not** rule out a true DSpark-specific microbatch path.
- `issue468/81` rules out `verify_n` truncation as a sufficient simplification, which raises the priority of a **kernel/path optimization** rather than a policy optimization.

Why it is still in scope:
- code-only
- verifier-path optimization
- does not alter drafter model or target model
- preserves the current B2 contract if implemented correctly

Independent read on likely win size:
- current best Opp-a path on sampled prompt: `36.71 t/s`
- benchmark prompt plain baseline: `35.84 t/s`
- benchmark prompt best verify_n simplification: `32.09 t/s`
- a ~10ms verifier reduction on the live merged-correction path is plausibly enough to cross on at least some benchmark prompts; a ~19ms reduction is the original break-even target from `issue468/79`

Status:
- **viable**
- next concrete lead to investigate/implement

### Lead B — verifier readback / output-head trimming (LOW confidence; likely insufficient)

Why considered:
- current DSpark B2 path reads full `row_logits` for all verified positions
- this is heavier than the original phase-1 batch timing harness, which was mostly “tops-only” timing

Why it is probably not enough:
- prior phase-1 findings (`issue468/06`) already showed output-head / readback overhead was small relative to layer compute in the exact path
- current live verify remains ~70ms even after other cleanup; the likely dominant cost is still the batch layer pass itself

Status:
- possible sublead inside Lead A
- not strong enough to stand alone as the next main lead

### Lead C — further correction/anchor-cycle fusion beyond merged pending-anchor (LOW confidence / likely out of scope)

Why reconsidered:
- Opp-a retry succeeded in merged form, so further control-flow fusion is tempting

Why downgraded:
- the old redundant-forward claims around anchor/correction were already heavily audited in `issue468/37` and `issue468/38`
- the current merged pending-anchor path likely captures the safe in-scope portion already
- further savings would drift toward algorithm/control-flow redesign

Status:
- **not the next lead**

### Lead D — custom verifier policy without kernel changes (EXHAUSTED)

Includes:
- verify-length truncation
- block-size simplification
- similar runtime policy-only changes

Evidence:
- block-size wash (`issue468/37`)
- `verify_n` simplification negative (`issue468/81`)

Status:
- **exhausted**

## 4. Review conclusion

After Opp-a and verifier simplification:

- the only clearly viable remaining Tier 1–2 lead is a **real custom microbatch verifier** (or a closely related verifier-kernel/path optimization)
- policy-only verifier simplifications are now measured negatives
- further anchor/correction fusion beyond the merged pending-anchor path is low-confidence and likely risks crossing into redesign territory

## 5. Next action from this review

Proceed to verifier lead 2:

1. investigate the current generic verifier for removable work beyond `verify_n`
2. if the investigation still points to the same bottleneck, implement a targeted custom microbatch verifier experiment
3. measure at temp=1 and report:
   - live vs oracle accepted drafts/cycle delta
   - live vs oracle committed tokens/cycle delta
   - ms/token decode speed

## 6. Round-1 verdict

This is **review round 1**, and it does **not** conclude convergence.

There is still one credible in-scope lead left:
- custom microbatch verifier / verifier-kernel optimization

So the blocker rule is **not** yet satisfied for “two consecutive reviews with no viable leads.”
