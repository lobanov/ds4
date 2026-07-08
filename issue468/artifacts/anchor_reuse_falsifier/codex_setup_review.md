## Verdict Per Claim

1. **Sound, with post-token qualification.** `main_hidden_posN` is built from `dump_hc_ffn_post-* _posN`, so it is post-token-at-position `N`, not pre-token. `target_tokens[k]` maps to absolute `pos0+k`; no off-by-one found. Evidence: [build_main_hidden_from_captures.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/build_main_hidden_from_captures.py:62), ds4 logs token then evals it at current `checkpoint.len`.

2. **Sound for this oracle.** In reuse mode, `main_hidden` only changes `win_kv[s][step]`; residual stream is `embed_w[draft_ids]`, with `draft_ids[0]=anchor`. Evidence: [measure_acceptance_bundle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/measure_acceptance_bundle.py:186).

3. **Questionable.** “Consistent-lag” is one plausible DSpark-window policy, not proven real. A folded verifier would process the correction as first token of the next verify batch, so prior-cycle slots could be backfilled with true hiddens; this variant was not run. Minor prose bug: `pos0` is generated token 0, not “last prompt token”.

4. **Questionable.** Fidelity gate proves the baseline reproduces retained oracle drafts, but retained summaries were produced by the same harness. It validates numerics/plumbing, not reuse semantics or verify-produced hidden equivalence.

5. **Likely-wrong as a conclusion.** Mechanical `SURVIVES` follows P3 because delta is `+0.025`; “optimistic edge legitimate / lead-05 unblocked” overclaims. Prefix CI is broad; real verifier cost, rollback, and verify-batch hiddens are untested.

## Premise Sensitivity

- **P1:** Under the premise, the drafter tolerates a valid-prefix stale hidden. If P1 fails, reuse falls back toward shipped-verifier accounting: K=4 about `-18.9%`, not `-0.9%`.
- **P2:** Under fixed DSpark/IQ2XXS, result is local. If either changes, no transfer.
- **P3:** Under user rule, verdict is `SURVIVES`. Under a statistical/practical rule, I’d call it **marginal/indeterminate**.

## Does This Test Real Reuse Cost?

**No. It tests one necessary representational condition, not end-to-end anchor reuse.**

What it does test: drafter acceptance when current anchor token enters as embedding and current target hidden is replaced by `H[pos-1]` from a clean greedy decode prefix.

What it does not test: hiddens produced by `metal_graph_verify_suffix_tops`, folded `K+1` verify behavior, target KV commit/rollback, residual cycle overhead, or whether the DSpark KV window should be lagged forever versus backfilled after verify. ds4 verifier code explicitly uses batch paths and prefix capture machinery; the falsifier never samples those hiddens.

## New Experiments To Try

1. **Actual verify-produced-hidden capture.** Force a rejection, dump `hc_ffn_post` rows from the verifier batch, compare to decode `main_hidden`, then feed those hiddens to drafter. Signal: directly validates P1. Effort: medium.

2. **Backfilled-window offline variant.** Current slot uses `H[pos-1]`; prior slots are updated to true `H[pos]` after the hypothetical verify. Signal: whether consistent-lag choice drives result. Effort: low plus one model load.

3. **Clustered stats + larger cells.** Recompute prompt-cluster bootstrap/sign tests over temp 0 and maybe retained temp 0.5/1.0. Signal: whether `+0.025` is stable or noise. Effort: low/medium.

4. **End-to-end folded verifier prototype.** Measure exactness and cycle cost against lead-05 success criteria. Signal: whether the `26 ms` decode is actually removed without new overhead. Effort: high.

## Leading Hypothesis

The “reuse beats baseline” result is mostly small-sample/noise plus a benign KV perturbation, not evidence the optimistic speedup edge is real. Aggregate match rate barely changes: `51.0% -> 51.25%`; prompt prefix deltas are mixed.

**Decisive test:** capture and use actual hiddens from a folded verifier pass, with the exact DSpark window update policy the implementation would use, then measure both acceptance and cycle timing.