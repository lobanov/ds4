# Post-falsification next avenues

Date: 2026-07-01

## Context

The latest 4-context acceptance-targeted collector variants now include:

- default weighted 19-token bundle merge: `+0.2854%`
- front-loaded position weights: `-0.0297%`
- baseline-hardness anchor weights: `+0.3620%`
- teacher-forced block inputs: `-0.1700%`

`49_qdump_oracle_envelope_falsification.md` then established a much stronger
upper bound on this family:

- even an impossible per-anchor oracle chooser across baseline plus all four
  current candidate q-dump families reaches only `+1.0051%` mean accepted on
  `8192/16384/24576/32768`

That is far below the assignment gate of:

- `>= +5%` average acceptance across `8k..64k`

So the useful question is no longer "which small collector weight tweak should
be tried next?" It is "which larger hypothesis changes still have a defensible
chance of moving the assignment?"

## Ranked avenues still worth exploring

### 1. Tensor / layer sensitivity mapping

This is the highest-value next avenue.

Reason:

- the current evidence says the main limitation may be **where** the fixed
  `Q4_K` error budget lands, not just how imatrix examples are reweighted
- acceptance could be dominated by a small subset of `mtp.*` tensors, layers,
  or expert classes
- if one slice is disproportionately sensitive, generic all-tensor imatrix
  averaging can easily miss it

Practical experiments:

- compare candidate quality when selected `mtp.*` tensors are left at higher
  precision in controlled ablations
- run per-layer or per-tensor-class sensitivity sweeps
- use those results to decide whether another imatrix pass is even worth doing

Why this is still credible:

- it is a larger quantization-allocation hypothesis, not a minor variant of the
  already-falsified collector family

Constraint:

- if the assignment interpretation forbids any non-`Q4_K` tensor exceptions in
  the evaluation artifact, this becomes a diagnostic path rather than a ship
  candidate

### 2. Failure decomposition by context / anchor / step

This is worth doing as a diagnostic, not as a main product bet.

Reason:

- the current averages may hide sharply different regimes by context length,
  anchor hardness, or drafted position
- if gains are concentrated in a narrow regime, that would clarify what the
  real bottleneck is

Useful questions:

- are regressions concentrated at one context band?
- do early rejection steps dominate the loss?
- are some anchor classes already near baseline while others are consistently
  bad?

Why this is lower value than sensitivity mapping:

- the oracle result already implies that the current q-dump family has limited
  total headroom
- decomposition may explain the failure well without creating a path to `+5%`

### 3. A genuinely different calibration objective

This is only worth exploring if it produces a **new** q-dump family, not
another merge-weight variant on the same collected states.

Reason:

- the current variants all reshuffle closely related evidence
- the oracle cap means that better post-hoc reuse of the same family is not
  enough

Examples of "different enough":

- collect only reject-boundary or near-disagreement states
- optimize around drafter-target mismatch events rather than generic activation
  preservation
- isolate quantization-sensitive failures from inherently hard baseline misses

Why this still has a chance:

- it could generate states that are outside the current oracle envelope

Why the bar is high:

- teacher-forced block inputs already tested the strongest obvious state-match
  hypothesis and did not help

### 4. Long-context-only validation at `40k..64k`

This is a weak but still defensible check.

Reason:

- the current oracle falsification covers only `8k..32k`
- the assignment gate averages through `64k`

Why this is not a strong mainline bet:

- to rescue the assignment, the long-context tail would need to be
  disproportionately positive
- the existing mid-context results are too small to make that likely

Best use:

- run only if the cost is low and the goal is to close the remaining empirical
  gap before declaring the current family exhausted

## Avenues that are no longer worth meaningful investment

The following now look effectively exhausted on the current path:

- more draft-position weight schedules
- more anchor-hardness reweighting
- more teacher-forced vs noise-seeded variants within the same collector family
- more post-hoc q-dump selection over the already collected sweep root

Reason:

- the best real measured result is `+0.3620%`
- the impossible oracle envelope is only `+1.0051%`
- both are far below the `>= +5%` assignment gate

## Recommendation

The default recommendation is:

1. stop spending serious time on current-family collector tuning
2. pivot to a larger hypothesis, with tensor / layer sensitivity mapping first
3. only keep low-cost diagnostics that improve confidence in the no-ship
   conclusion

Bluntly:

- if the assignment must remain pure `Q4_K + imatrix` on the unchanged runtime
  path, the current evidence says the gate is unlikely to be reachable through
  further local tuning
- the next useful work should either expose a larger quantization-allocation
  mechanism or strengthen the case to stop
