## Is HOLD The Right Tier? (vs GO / STOP)

Yes, **HOLD is the right tier**, but for weaker reasons than stated.

FP p=1 recomputes exactly: `6/7, 6/7, 4/7, 4/7, 5/7 = 25/35 = 0.7143`.

The reported CI `[-0.251, +0.130]` is correctly computed for the table as written: per-prompt deltas vs Q2’s 8-row rates, df=4. But it is not a clean same-anchor comparison. FP uses 7 rows/prompt; Q2 uses 8, and the FP/Q2 token spines differ. Restricting Q2 to 7 rows gives mean delta `-0.0286`, CI about `[-0.223, +0.166]`.

Not **STOP**: underpowered, CI allows material positive lift, representation still unproven.

Not **GO**: aggregate is not positive, only 2/5 prompts show `+10.7 pp`, no D1/aligned-spine proof, no CI excluding zero.

## Is The "Native Drafter Quality" Interpretation Sound?

Partly, but overstated.

Inventory supports the premise: drafter was distilled against native served precision. So a valid native-hidden pilot failing to beat Q2 would weaken the IQ2XXS-hidden-degradation hypothesis.

But this pilot does not prove “deficit = native drafter quality.” C1 remains unresolved: `mhc_post` capture is plausible, but not algebraically proven equivalent to the drafter’s real native input. If that representation is wrong, the test is invalid rather than negative.

## Is "IQ2XXS Does Not Degrade Hiddens" An Overclaim?

Yes. Supported wording is: **this pilot finds no robust evidence that native hiddens improve drafter p=1 enough to justify Phase B**.

Unsupported wording: **IQ2XXS does not materially degrade hiddens**. That is causal, underpowered, and representation-dependent.

## Any Remaining Issue That Makes The Go/No-Go Premature?

Not premature for **HOLD**. Missing proof and weak stats argue against spending on full capture.

Premature for stronger claims: “pipeline validated,” “native drafter quality,” and “IQ2XXS exonerated.”

Prefix-caching fix looks directionally credible: `enable_prefix_caching=False`, `max_num_seqs=1`, reset before prompt, no strict positional p1 collapse after fix. But retained artifacts do not fully prove absence of residual contamination; they only make the original bug unlikely.

## Final Gate Verdict

**APPROVE the HOLD.**

But amend rationale: HOLD because the pilot is underpowered, not same-anchor clean, and representation-unverified. Reject the causal overclaims until C1/aligned-spine/D1 evidence exists.