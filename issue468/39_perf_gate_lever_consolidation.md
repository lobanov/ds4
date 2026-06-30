# DSpark Perf-Gate Lever Consolidation — Structural Infeasibility Evidence

Date: 2026-06-30. Seventh productionization handoff note. Consolidates the
complete lever-by-lever investigation toward the perf gate (DSpark > baseline
39 t/s) and the evidence-backed conclusion. Doc-only.

## 0. The gate

DSpark end-to-end gen t/s must STRICTLY EXCEED plain-decode baseline (39.0 t/s
on this M5 Max, flat across 8k-32k). Current DSpark: 29-31 t/s (0.74-0.81×).
Required improvement from current: ~+30%.

## 1. Lever-by-lever outcome (all in-scope levers investigated)

| lever | status | evidence |
|---|---|---|
| **Opp1** anchor-decode-elimination | INFEASIBLE | token-flow evidence (issue468/37): anchor decodes a genuinely NEW token each cycle (first_token 2581,2019,12268,... distinct; cycle-1 full-accept yet cycle-2 anchor = predicted-next, not last draft). Installs KV[anchor] + produces drafter main_hidden. Not redundant. |
| **Opp2** block-size (d=1..5) | WASH | issue468/37 sweep: ms/tok flat ~32 across all d. Verify scales ~linearly (~11ms/pos), commits/cycle drops proportionally → cancel. No d crosses the gate (best d=4 = 0.81×). |
| **Opp-a** correction-decode-elimination | INFEASIBLE | issue468/38. P0 (7f2320f) ALREADY bridged the verify→decode KV-format divergence (spec_frontier_commit_dspark_prefix). Correction token C is structurally irreducible: verify processes drafts not C (C computed post-verify), so no KV[C] installed and verify rows predict-after-draft[j] not after-C. Deferring = main-loop control-flow redesign (out of scope). |
| **Opp5** imatrix (generic path) | FAILED | issue468/34 §2.7: 3 real candidates at 8k scored +0.11% / −0.52% / −1.21% acceptance. The 65536-anchor (most-data) candidate REGRESSED. Doc's own conclusion: "objective mismatch" — generic imatrix does not align with the acceptance metric. |
| **Opp3** drafter sync-fusion | CEILING ≤+2 t/s | Drafter phase = 9.7ms (d=5), shrinks with d. Even eliminating the ENTIRE phase (impossible — real compute) saves 9.7ms/130ms-cycle ÷ 4 commits = 2.4ms/token → ~34 t/s. Realistically +1 t/s. Cannot cross gate. Implementation is moderate-effort / high-correctness-risk (per-layer state machine + reused scratch tensors + explicit sync for a reason). |
| **Opp5** imatrix (acceptance-targeted) | UNATTEMPTED — large hypothesis | issue468/34 §4: "finishing" = implement a NEW teacher-forced, position-weighted, acceptance-boundary collection mode. Explicitly "a measurement hypothesis, NOT a guaranteed fix" (§4.1). Contrary evidence: generic path regressed at scale. Multi-day effort. |

## 2. Why the gate is structurally hard here

The issue468/36 report's "three target forwards (anchor + verify + correction =
119ms, 91% of cycle) are structurally irreducible" conclusion was CORRECT (my
turn-1 rebuttal, which drove the Opp1/Opp-a tweaks, was flawed — see the
intellectual-honesty correction in issue468/38 §5). All three forwards process
distinct, necessary tokens; none is redundant; none is eliminable without
redesigning the B2 algorithm or the main-loop control flow (both out of scope).

On this hardware (M5 Max), the baseline is additionally fast (39 t/s) AND flat
(doesn't slow with context), so each accepted token saves only ~25.6ms — too
little to amortize the irreducible 3-forward cycle at ~31% acceptance
(~3.5 commits/cycle). The same code WOULD win on slower baselines per the
issue468/32 projection; it loses specifically because this baseline is fast and
flat.

## 3. The two remaining options (both need user direction)

**Option A — authorize Opp5-targeted (acceptance-targeted imatrix):** the only
remaining lever with any theoretical path to the gate. BUT: multi-day research
effort (new collection mode: teacher-forcing, position-weighting, hard-case
emphasis, bundle-driven), explicitly hypothesis-level confidence, and the only
existing imatrix evidence (generic path) regressed at scale. Realistic prospect:
uncertain; could plausibly fail the +5% acceptance gate just as the generic
path did, and even +5% acceptance may be insufficient to cross the 39 t/s perf
gate from a 29-31 t/s base.

**Option B — accept the structural perf-blocker:** with Opp1/Opp2/Opp-a ruled
out and Opp5-generic measured-failed, and Opp3 ceiling-bounded at ≤+2 t/s, the
gate is not achievable with bounded/high-confidence in-scope work. The
implementation is correct, self-contained, B2-exact, MTP-regression-free, and
production-ready; it simply cannot beat this fast-flat baseline. This matches
the issue468/36 conclusion (now independently re-confirmed via the lever
sweep).

## 4. What IS delivered (production-ready regardless of the gate)

- Clean self-contained port; make 0 warnings; --dspark opt-in; MTP untouched.
- P0 KV-replay elimination (per-position frontier capture + O(1) correction);
  the prior O(k+1) replay is gone.
- P1 GPU Markov head (+6.5%, bit-exact).
- Full-accept s->logits dedup fix (P4).
- Block-size compile-time override (abcf8cf) + measured sweep (issue468/37).
- B2 exactness preserved; output verifier-guaranteed correct.

## 5. Recommendation

The evidence strongly favors Option B (structural perf-blocker) as the honest
outcome: every bounded/high-confidence lever has been measured and rules out or
fails. Option A (Opp5-targeted) is the only path with theoretical gate-
crossing potential, but it is a multi-day hypothesis with contrary evidence —
exactly the kind of expensive, uncertain effort that warrants explicit user
authorization rather than autonomous commitment, and whose likely outcome
(failing the +5% acceptance gate as the generic path did) would not cross the
perf gate anyway.

Awaiting user direction: (A) authorize the Opp5-targeted research effort,
(B) accept the structural perf-blocker, or (C) other.
