# Tier 1–2 lead review, round 2 — verifier-profile conclusion

Date: 2026-07-02.

Purpose: second independent Tier 1–2 review after the live verifier microprofile in `issue468/83`.

## Inputs reviewed

- `issue468/80_oppa_merge_correction_anchor_investigation.md`
- `issue468/81_verifier_n_simplification_measurement.md`
- `issue468/82_tier1_tier2_review_round1.md`
- `issue468/83_verifier_profile_round2.md`
- independent codex review (gpt-5.5 xhigh) of the remaining in-scope lead after the verifier microprofile

## What changed since round 1

Round 1 (`issue468/82`) kept exactly one in-scope lead alive:
- custom microbatch verifier / verifier-kernel optimization

Since then, `issue468/83` profiled the live verifier and found:

| component | ms |
|---|---:|
| upload | 0.027 |
| **layers** | **96.218** |
| output head | 0.056 |
| top-1 kernel | 2.077 |
| read tops | 0.000 |
| read logits | 0.037 |
| **verify total** | **98.415** |

This materially changes the viability of the remaining lead.

## Review conclusion

### 1. Is there still a bounded in-scope Tier 1–2 lead left?

**No.**

The only remaining path would need to reduce the verifier's **layer batch compute itself** by roughly `20–22 ms/cycle` while preserving the current B2 contract.

That is no longer a bounded Tier 1–2 optimization. It is deep Metal verifier-kernel work.

### 2. What this means for the remaining verifier lead

The microprofile shows:
- readback / top-k / output-head trimming is negligible
- verifier simplification by lowering `N` already failed (`issue468/81`)
- the remaining cost is overwhelmingly the generic batch layer pass

So the old surviving lead from round 1 is no longer viable in the intended Tier 1–2 sense.

### 3. Consecutive-review status

This round **does count as a no-viable-leads review**.

But it is **not** the second consecutive no-viable-leads review, because round 1 (`issue468/82`) explicitly kept one lead alive.

So after this note the state is:
- round 1: one viable lead remained
- round 2: no bounded viable Tier 1–2 lead remains

A further review is not actually needed to re-prove the same point unless a new in-scope lead is surfaced.

## Important correction to round-1/round-2 framing

Two flaws need to be recorded explicitly:

1. **Prompt mixing**
   - the `36.71 t/s` Opp-a result in `issue468/80` came from a short sampled Python prompt
   - the `35.84 t/s` baseline in `issue468/81` came from the benchmark chat prompt
   - those are not directly comparable, so prior “near miss” language overstated how close the gate was

2. **Verifier profile should be used for attribution, not absolute timing**
   - `issue468/83` measured a higher absolute verifier wall time than the earlier non-profiled runs
   - the profile is still decisive for attribution (layers dominate overwhelmingly)
   - but the exact profiled milliseconds should not be substituted directly for the unprofiled baseline timing model without caution

## Practical outcome

After Opp-a retry, verifier simplification, and the verifier microprofile:

- there is no bounded Tier 1–2 code-only lead left under the current goal scope
- any remaining path to the perf gate would require:
  - deep new Metal verifier-kernel work, or
  - an out-of-scope algorithm/model change

That makes the current state suitable for a perf-blocker report **once the goal’s explicit blocker sequencing is satisfied and reported carefully**, with the correction that this is the **first** no-viable-leads review, not the second consecutive one.
