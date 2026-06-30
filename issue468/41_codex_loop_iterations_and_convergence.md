# Codex Review Loop — Iterations 1-3 + Temp-Aware B2 Experiment + Convergence

Date: 2026-06-30. Ninth productionization handoff note. Records the codex
(gpt-5.5, high-effort) review loop, the temp-aware B2 experiment (iter-3 lever),
and the convergence conclusion. Doc-only research/productionization record.

## 0. Loop setup

Per the goal tweak: iteratively run codex review (gpt-5.5, high-effort, imatrix
excluded, drafter model frozen), implement the next-highest-leverage code-only
finding, repeat until convergence. Each finding independently verified before
implementing.

## 1. Iteration 1 — findings + outcomes

Codex surfaced 4 findings. Each independently verified against the code + an
empirical run:
- **Bug #1 (HIGH): drafter KV window stuck at n_real=1.** FIXED (75b6e77) + cap
  OOB fix (b334681). PERF-NEGATIVE: 31.50→28.58 t/s (growing window slightly
  worse than stale window). Kept (restores intent).
- **Bug #2 (codex HIGH → downgraded): B2 not greedy-exact at temp=0.** Real but
  pre-existing (B2 uses stochastic min(1,p/q) + argmax(p-q) regardless of temp).
  Folded into the iter-3 temp-aware experiment.
- **Bug #3 (medium): B2 RNG fixed static.** FIXED (7ee304d). Now seeded from --seed.
- **Bug #4 (low): capture silent stale.** FIXED (7ee304d). Now hard-fails.
- Perf ceilings: readback <0.3ms (noise), CPU B2 loop <0.2 t/s (noise), custom
  verifier +2-6 t/s upper (high-risk), drafter sync-fusion +~1 t/s (ceiling ≤2).

## 2. Iteration 2 — CONVERGED verdict + n_real cap bug

Codex verdict: **CONVERGED** for code-only perf. Surfaced a HIGH-severity bug
in my Bug #1 fix: the n_real cap at DS4_N_SWA allowed an OOB (window 134 >
raw_cap 133 at cycle 129+). FIXED (b334681): capped at SWA-1, verified n_real
plateaus at 127 over 216 cycles. Custom verifier re-assessed as not-gate-crossing
(the ~18ms "fixed overhead" is mostly unavoidable DS4_N_LAYER compute, not
removable plumbing).

## 3. Iteration 3 — ONE-MORE-LEVER: temp=0-aware B2

Codex proposed ONE-MORE-LEVER: make B2 respect --temp (greedy-match acceptance +
argmax(p) correction at temp≤0). Estimated +5-8 t/s (~35-38, not confidently
>39). Go/no-go: ≥36 t/s at d=5.

### Experiment (implemented then REVERTED)

Implemented (locally): added `temperature` param to eval_dspark_b2; for temp≤0,
accept iff draft==argmax(p) (using row_tops + sample_argmax(s->logits)), correction
= argmax(p); temp>0 unchanged (stochastic, for ds4-eval at temp=1.0).

### Result: FAILED go/no-go

Measured n=256, ctx=8192, temp=0:
- gen t/s: **25.78** (go/no-go was ≥36; stochastic baseline ~28-31). WORSE.
- commits/cycle: **3.32** (stochastic was 3.66). WORSE.
- greedy-exactness: **NOT achieved** (`f3dee7..` ≠ plain greedy `1d4393..`).

### Root cause of failure

The batch verify's argmax (row_tops from metal_graph_verify_suffix_tops) diverges
from sequential-decode's argmax — the known batch-vs-decode divergence the code
comments warn about ("small row-wise differences in HC/MoE/output kernels are
enough to flip future greedy tokens"). So greedy-match checks drafts against a
slightly-WRONG target:
1. Stricter than stochastic → lower acceptance (3.32 vs 3.66 commits/cycle).
2. Doesn't match plain greedy anyway (batch argmax ≠ sequential argmax).

Codex's "+5-8 t/s" estimate assumed the probe's 2.74/5 greedy-match rate, but
that was measured against the ORACLE (sequential decode), not the batch verify.
The live batch-verify greedy-match rate is lower. The lever is perf-negative.

### Decision: REVERTED

Reverted entirely (signature, accept loop, both callers). Working tree back to
b334681 (stochastic B2 + Bug #1 + cap fix + Bug #3 + Bug #4). The stochastic B2
is correct for ds4-eval (temp=1.0) and produces higher t/s than greedy-match at
temp=0. Neither respects temp=0 greedy-exactness (a pre-existing B2 design
property, not a regression).

## 4. Convergence conclusion

The codex-review loop is **CONVERGED**:
- Iter 2: CONVERGED (custom verifier not gate-crossing).
- Iter 3: ONE-MORE-LEVER (temp-aware) → tried, FAILED go/no-go, reverted.

No remaining VIABLE code-only lever crosses the gate. Exhaustive lever inventory:
- Block size d=1..5: WASH (ms/tok flat ~32).
- Opp1 (anchor-decode-elim): INVALID (token-flow evidence).
- Opp-a (correction-decode-elim): INFEASIBLE (P0 already bridged KV; C irreducible).
- Bug #1 (drafter KV window): fixed, PERF-NEUTRAL/NEGATIVE.
- Temp-aware B2: tried, FAILED (25.78 < 36 go/no-go).
- Custom verifier: not gate-crossing (high-risk, +1.5-4 t/s upper).
- Drafter sync-fusion: below-noise (+0.1-0.4 t/s).
- Fused Markov: below-noise (+0.3-0.7 t/s).
- Readback early-exit: below-noise (<0.3ms).
- CPU B2 loop fusion: below-noise (<0.2 t/s).
- Imatrix/Opp5: OUT OF SCOPE (drafter model frozen).

## 5. Gate status

DSpark ~28-31 t/s (best measured ~31.88; sweep best d=4 31.76). Baseline 39.0 t/s.
Gate (DSpark > 39): **NOT MET** (0.73-0.82×). All code-only levers on a frozen
drafter are exhausted. The gap (~8-11 t/s, ~25-35%) cannot be closed without
either a better drafter (frozen) or a non-code lever (imatrix, out of scope).

Per the goal's blocker rule, all 5 preconditions are now satisfied:
1. Baseline-regression check clean ✓ (39.08 vs 39.01).
2. P1-P4 attempted ✓.
3. Block-size sweep measured ✓ (wash).
4. Bug #1 implemented + measured ✓ (perf-neutral/negative).
5. Codex-review loop converged ✓.
