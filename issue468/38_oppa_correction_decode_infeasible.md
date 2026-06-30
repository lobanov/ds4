# Opp-a (Correction-Decode Elimination) — Investigation Result: INFEASIBLE (structural)

Date: 2026-06-30. Sixth productionization handoff note. Records the
investigation of Opp-a (correction-decode elimination) and the conclusion.
Doc-only research/productionization record. No code, no runtime impact.

## 0. Context

Per the goal tweak after issue468/37 (block-size sweep = wash) and the Opp1
invalidation (anchor decode processes a genuinely new token each cycle),
Opp-a (correction-decode elimination) was identified as the next lever: the
"kv" phase averaged ~22ms/cycle (~25ms on the ~85-90% of cycles that
partial-accept), the second-largest phase after verify. The goal's framing
was: "solve the verify-batch→decode-path KV-format divergence (the known P0
'compressed KV cache overflow' blocker) so the accepted-prefix KV is
committable in decode format, then defer KV[correction] + next-logits to the
following anchor decode."

## 1. Finding #1 — the "KV-format divergence" was ALREADY solved by P0

The premise that P0 stopped at O(1) because of an unbridged KV-format
divergence is INCORRECT. Re-reading the P0 commit (7f2320f) and the code:

- `spec_frontier_commit_dspark_prefix(s, idx)` (ds4.c:24955) already copies
  the per-position captured compressor frontier (attn_state_kv/score +
  index_state_kv/score for ratio-4 layers) back into the live state. P0's
  mechanism: capture the frontier after each draft position during verify
  (`spec_prefixN_*` buffers, allocated only when enable_dspark), and on
  partial-accept restore the capture at `idx = n_draft_accept - 1`.
- This IS the verify-batch→decode-path KV bridge. The accepted-prefix KV is
  already committable and committed in decode format. The earlier
  `a81bc8a` "phase 5: revert 1-step replay (compressed KV overflow)" was the
  PRE-P0 attempt; P0's per-position frontier capture RESOLVED it.
- So the accepted-prefix (drafts[0..k-1]) KV is NOT recomputed — P0 eliminated
  that. The single remaining correction decode is for token C ALONE.

(My pause-goal message mis-attributed P0's O(1) stop to an unbridged format
divergence; it was not — the divergence was bridged. The O(1) decode remains
because C genuinely needs its own forward, see Finding #2.)

## 2. Finding #2 — the correction token C is structurally irreducible in B2

The correction decode processes token C and produces two things the cycle
needs, NEITHER of which is available from the verify batch:

- **KV[C]**: the verify batch processes the `block` drafts (ds4.c:29260 pushes
  drafts into checkpoint; verify_suffix_tops runs at `start` for `block`
  tokens). It installs KV for draft[0..block-1], NOT for C. C is computed
  POST-verify from argmax(p−q) at the rejection position, so no verify row
  ever had C as input.
- **predict-after-C** (→ next cycle's anchor via sampling): `row_logits[j]` =
  target's prediction AT draft[j]'s position = predicts the token AFTER
  draft[j]. On reject at position i, C occupies draft[i]'s position, but the
  verify computed its row with draft[i] as input — it predicts after draft[i],
  NOT after C. Substituting C changes all downstream computation, so the verify
  cannot supply predict-after-C.

The main loop (ds4_cli.c:477) samples the next anchor from `s->logits` BEFORE
calling eval_dspark_b2. The correction decode is what sets `s->logits` =
predict-after-C. Skipping it leaves s->logits stale → the main loop samples a
garbage anchor. There is no chicken-without-egg: predict-after-C cannot be
produced without C's forward, and C's forward IS the correction decode.

## 3. Why "defer to the next anchor decode" is out of scope

The only conceivable way to merge C's forward with another is to identify C
as the next cycle's anchor (so the next anchor decode processes C, installing
KV[C] + producing predict-after-C). But that requires:
- Changing the main-loop contract so the next cycle's first_token = C while
  suppressing C's re-emission (C was already committed as the correction) —
  a dedup/control-flow restructure of ds4_cli.c's generate loop, NOT KV
  plumbing.
- Even then it is unclear it saves a forward: C and predict-after-C are both
  distinct output tokens that each need processing.

This is main-loop control-flow / cycle-semantics redesign — explicitly out of
scope ("only KV-format plumbing, redundant-forward elimination, drafter
sync-fusion, and draft-quant finishing are in scope"; "do NOT redesign the
spec-decode algorithm"; "the B2 rejection-sampling design is fixed").

## 4. Conclusion — Opp-a is attempted-but-infeasible (structural)

Opp-a (eliminate the partial-accept correction decode) is INFEASIBLE within
the goal's scope, for structural reasons — not for a bridgable format issue.
This matches the goal's contract fallback: "If the KV-format divergence
cannot be bridged without breaking exactness/MTP, document the specific
blocker and proceed to Opp-b." Proceeding to Opp-b (Opp5 imatrix, then Opp3
drafter sync-fusion).

## 5. Intellectual-honesty correction to issue468/36 + turn-1 analysis

The issue468/36 report's "the three target forwards (anchor + verify +
correction = 119ms, 91% of cycle) are structurally irreducible" conclusion
was CORRECT. My turn-1 "rebuttal" of it (which drove the Opp1/Opp-a goal
tweaks) was based on a flawed premise — I claimed the anchor decode
duplicated the prior correction decode. The token-flow evidence (issue468/37
pause note) disproved that for the anchor (Opp1 invalid), and this note
disproves the analogous claim for the correction (Opp-a infeasible). All
three forwards process distinct, necessary tokens; none is redundant.

The valid NEW findings from this productionization push are: (a) the
block-size sweep (issue468/37) — block size is a wash, verify ~linear
~11ms/position, ruling out Opp2; and (b) this Opp-a infeasibility ruling,
which removes the last "redundant-forward" lever. The remaining levers are
Opp5 (raise acceptance — the commits/cycle multiplier) and Opp3 (reduce the
~9.7ms drafter phase). These do NOT eliminate forwards; they amortize the
irreducible 3-forward cycle over more commits and shave drafter cost.

## 6. Honest outlook for the gate

Starting from ~29-31 t/s (sweep) with all forwards irreducible, reaching the
gate (>39 t/s) requires Opp5 + Opp3 to deliver ~+30%. Opp5 (imatrix) raising
drafter acceptance from ~31% is the higher-leverage of the two; Opp3
(drafter sync-fusion) is a smaller additive win on the ~9.7ms drafter phase.
This is a high bar. Per the goal's blocker rule, Opp5 and Opp3 must be
attempted before any perf-blocker may be raised; if they do not cross the
gate, the blocker is raised with the full measured evidence.
