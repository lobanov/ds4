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

## 2026-06-04: Phase 3 rejects backend-variance-only classification

Decision:

- Do not classify the remaining resumed-decode drift as acceptable backend variance.
- Treat the current problem as a handoff-specific resumed-decode mismatch until proven otherwise.
- Keep Phase 4+ productization work blocked behind a same-backend parity fix or a tighter localization of the mismatch.

Evidence:

- Local and DGX GGUF hashes matched exactly before the runs:
  - `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- The new `tests/issue304_phase3_vectors` helper compared fresh local Metal decode against distributed-prefill -> merged `DSV4` -> local Metal decode.
- Official vector cases passed the existing selected-token/top-logprob gate on every run, but same-backend parity still drifted:
  - `short_code_completion` worst@5: `top20=17/20`, `top64=62/64`, `rms=0.356403291`
  - `short_italian_fact` worst@5: `top20=19/20`, `top64=62/64`, `rms=0.21716401`
  - `long_code_audit` worst@5: `top5=4/5`, `top20=15/20`, `top64=55/64`, `rms=0.876981199`, `max_abs=4.63468361`
- The local-golden frontier showed the clearest rejection:
  - the resumed handoff still passed the coarse local-golden fixture on all five runs
  - but same-backend continuation parity for `long_story_4096` failed worst@5 with `top5=3/5`, `top20=12/20`, `top64=46/64`, `rms=2.1708498`, `max_abs=12.3051796`

Why this changes the next steps:

- Cross-engine `CUDA -> Metal` drift is no longer the main question.
- The more important defect is that a distributed-prefill checkpoint resumed on the same local decode backend does not evolve like a fresh local checkpoint.
- The next work should inspect post-load decode evolution on Metal after merged distributed prefill, rather than moving on to residency or UX work.

Operational note:

- Phase 3 also confirmed that merged `DSV4` staging requires `prefill_cap` alignment in addition to `ctx` alignment.
- The DGX worker had to be launched with:
  - `DS4_METAL_PREFILL_CHUNK=2048` for the strict official-vector environment
  - `DS4_METAL_PREFILL_CHUNK=4096` for the local-golden environment

## 2026-06-04: Phase 3.5 broadens the drift classification across six routes

Decision:

- Keep treating resumed payload routes as suspect, but stop using the local-golden fixture as a local-Metal oracle.
- Treat official-vector acceptance as necessary but not sufficient.
- Split the remaining investigation into three independent buckets:
  - backend-specific long-prompt generation variance,
  - local-golden fixture drift,
  - payload-resume-specific variance beyond those two baselines.

Evidence:

- Phase 3.5 measured all six routes at worst@5 with the DGX worker and DGX-side helper running at `--power 50`.
- `short_code_completion` official acceptance was route-specific:
  - fail on route `1` pure local CUDA
  - fail on route `3` distributed generation `CUDA -> Metal`
  - fail on route `5` resumed `CUDA -> Metal`
  - pass on routes `2`, `4`, and `6`
- Longer official cases all passed the official gate, but resumed payload routes still had weaker parity:
  - `short_reasoning_plain` route `5`: `top64=61/64`, `rms=0.382280707`
  - `short_reasoning_plain` route `6`: `top20=18/20`, `top64=59/64`, `rms=0.504765928`
  - `long_code_audit` route `5`: `top5=4/5`, `top20=16/20`, `top64=50/64`, `rms=0.972514927`
  - `long_code_audit` route `6`: `top5=4/5`, `top20=15/20`, `top64=53/64`, `rms=0.903749108`
- Pure CUDA already shows longer-prompt drift without any payload resume:
  - `long_code_audit` route `1`: `top5=4/5`, `top20=16/20`, `top64=52/64`, `rms=0.893455684`, `max_abs=4.59287167`
- The stored local-golden fixture is not purely local-Metal-canonical:
  - `long_story_4096` route `2` pure local Metal missed it with `top5=3/5`, `top20=16/20`, `top64=46/64`, `top20_max_abs=5.2168293`
  - routes `1`, `3`, `4`, `5`, and `6` still passed the coarse fixture

Why this changes the next steps:

- Phase 3 was correct to block productization on “same-backend parity” alone, but Phase 3.5 shows that the remaining variance story is not one-dimensional.
- A resumed-route miss is only attributable to the handoff path after it is compared against:
  - the matching direct-generation backend,
  - the opposite direct-generation backend,
  - and the behavior of the stored local-golden fixture itself.
