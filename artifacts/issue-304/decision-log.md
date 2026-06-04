# Issue 304 Decision Log

## 2026-06-03: Correct the earlier Phase 0 save-path conclusion

Decision:

- Do not treat the earlier `distributed KV shards use different layouts` result as evidence that mixed Metal/CUDA immediate merged-`DSV4` staging is broken.
- Treat that earlier result as an invalid baseline caused by mismatched worker/coordinator session context.

Evidence:

- The earlier failing run used:
  - coordinator `ctx=16384`
  - worker default `ctx=32768`
  - and now reports the exact mismatch:
    - `distributed KV shards use different layouts: field=ctx local=16384 remote=32768 remote_layers=22:42`
- The route code accepts this for inference because worker admission only requires `worker_ctx >= coordinator_ctx`.
- Re-running the same Metal/CUDA topology with worker `--ctx 16384` produced:
  - successful route formation
  - successful prefill on `README.md`
  - successful immediate `ds4_session_stage_payload()`
  - payload bytes: `221,006,660`
  - parsed merged `DSV4` header with `ctx_size=16384`

Why this changes the notes:

- The original decision overgeneralized from an invalid reproduction.
- The precise constraint is narrower:
  - distributed generation tolerates worker `ctx >= coordinator ctx`
  - merged shard save requires exact layout equality, so `ctx` must match across participants
- The Phase 2 `DSV4`-first path remains viable if the runbook pins matching `--ctx` on all distributed participants.

## 2026-06-03: Implement a dedicated Phase 2 handoff harness before changing core transport

Decision:

- Validate Phase 2 with a dedicated harness built on existing APIs before attempting any distributed transport or payload-format changes.
- Keep the first Phase 2 proof at the `DSV4` handoff boundary:
  - distributed prefill,
  - merged payload stage,
  - fresh local full-session load,
  - handoff logits comparison,
  - greedy continuation comparison,
  - and forced-token trace comparison.

Evidence:

- Phase 0 already proved that immediate merged `DSV4` staging works on the mixed Metal/CUDA route when `ctx` matches.
- Phase 1 already proved that whole-payload local resume is a valid correctness boundary and that `CUDA -> Metal` drift appears after restore during subsequent decode evolution.
- The existing test and helper surface already exposed the needed primitives:
  - `tests/issue304_phase0_dgx` for distributed route, prefill, and stage
  - `tests/issue304_phase1_matrix` for logit comparison and forced-token trace logic
- A new local helper, `tests/issue304_phase2_handoff`, now builds successfully with:
  - `make tests/issue304_phase2_handoff`

Why this changes the work:

- It lowers Phase 2 ambiguity without committing to chunked KV return or in-memory payload APIs.
- It makes the next failure mode attributable:
  - if staging fails, it is still a distributed gather/layout problem
  - if load fails, it is a local payload compatibility problem
  - if handoff logits match but forced trace drifts, it is post-load decode evolution, not a bad handoff point

Follow-up:

- Run the helper on the DGX/Mac topology and record the resulting metrics before deciding whether the remaining work is same-backend handoff validation, acceptable backend-variance classification, or a specific payload-resume defect.

## 2026-06-03: Phase 2 confirms exact handoff but not exact resumed decode evolution

Decision:

- Treat Phase 2 as complete.
- Do not spend more Phase 2 effort on distributed KV gather or merged payload staging.
- Carry the remaining observed drift forward as a mixed-backend resumed-decode evolution classification problem.

Evidence:

- `tests/issue304_phase2_handoff` on the authoritative DGX/Mac topology produced:
  - successful distributed prefill
  - successful merged `DSV4` stage
  - successful fresh local Metal load
  - exact handoff logits (`top1`, `top5`, `top20`, `rms=0`, `max_abs=0`)
  - exact 16-token greedy continuation match
  - but forced-token trace divergence at first bad step `1`
  - forced token before bad step: `5`
  - `rms=0.0802887231`
  - `max_abs=0.507860184`
- Throughput also matched the intended motivation:
  - distributed decode `16.28 tok/s`
  - local decode after handoff `30.32 tok/s`

Why this changes the next steps:

- The current implementation direction is validated at the feature level:
  - distributed prefill can hand off to materially faster local decode through the existing merged payload path
- The remaining correctness issue is narrower than “handoff is broken”:
  - the checkpoint is exact
  - the sampled greedy path stayed stable
  - the drift only appears in full-logit forced comparison after resumed eval begins

Follow-up:

- Phase 3 should focus on classifying backend-specific post-load decode drift, not necessarily eliminating it. Cross-engine drift is acceptable if distributed prefill -> local decode matches fully local inference on the same decode backend and passes the official-vector/local-golden correctness gates.

## 2026-06-04: Re-scope Phase 3 around handoff parity, not bit-exact cross-engine logits

Decision:

- Phase 3 success does not require eliminating `CUDA -> Metal` forced-logit drift.
- Treat cross-engine drift as acceptable engine variance if it is isolated from the handoff path.
- Require distributed prefill -> merged `DSV4` -> local decode to match fully local inference on the same decode engine against:
  - `tests/test-vectors/official.vec`
  - `./ds4_test --logprob-vectors`
  - `./ds4_test --local-golden-vectors`

Evidence:

- Phase 2 handoff logits matched exactly and 16-token greedy continuation matched exactly.
- The observed drift appeared only in full-logit forced comparison after resumed eval began.
- Existing test-vector infrastructure already treats the official API as a top-logprob reference rather than a bit-exact full-logit oracle.
- Local golden vectors already exist to catch substantial local backend drift while tolerating implementation-level numeric differences.

Why this changes the next steps:

- The next test should compare distributed-prefill-to-local decode against fully local decode on the same backend before assigning blame to CUDA/Metal engine differences.
- If the same-backend comparison and official/local gates pass, Phase 3 can classify the forced-logit drift as acceptable backend variance and move on to residency/workflow planning.
