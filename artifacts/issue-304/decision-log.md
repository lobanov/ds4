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

## 2026-06-04: Pivot next work toward practical handoff plumbing, defer deeper variance forensics

Decision:

- Do not spend the next implementation cycle on deeper route/backend variance investigation.
- Move next to practical handoff/product plumbing so the feature can be exercised and later benchmarked in more realistic workflows.
- Keep the existing Phase 3 and Phase 3.5 variance findings as active caveats, not resolved issues.

Evidence:

- Phase 3 and Phase 3.5 already produced enough evidence to show that route/backend variance exists and is not reducible to one simple explanation.
- The current workflow is still too research-specific to measure resumed routes convincingly on practical use cases.
- Some of the observed variance may shrink or shift as ongoing CUDA inference work lands, so further forensics now risks chasing a moving target.
- The more actionable next unknown is whether the handoff path can be turned into a usable operator-facing workflow without excessive complexity or unacceptable overhead.

Why this changes the next steps:

- Realistic resumed-route benchmarking depends on practical plumbing existing first.
- Official-vector and local-golden checks remain useful guardrails, but they are not enough by themselves to decide product value.
- The next useful deliverable is a runnable handoff workflow that can answer practical engineering questions:
  - how awkward the operator flow is,
  - where the real latency and memory costs land,
  - and whether resumed routes are good enough under actual benchmark workloads.

Deferred re-entry triggers:

- Re-open deeper variance investigation when one of these happens:
  - practical handoff plumbing lands and realistic benchmarks show meaningful regressions,
  - resumed routes underperform direct generation in user-visible ways,
  - CUDA inference changes materially and the route matrix should be re-run,
  - or product design starts depending on resumed routes being treated as equivalent.

Operational planning note:

- The next implementation phase should treat practical handoff plumbing and coordinator residency as one fused problem, not two separate phases.
- In this codebase, a benchmarkable workflow depends directly on how weights stay resident, how `--layers` maps to loaded spans, and whether the coordinator can transition from distributed prefill to local decode without harness-only engine choreography.

## 2026-06-05: Phase 4 targets final-worker residency, not topology flexibility

Decision:

- Keep topology flexibility out of Phase 4.
- Preserve the current coordinator-first route model:
  - coordinator prefills earlier layers,
  - worker(s) prefill later layers,
  - the final worker either owns output/logit generation or sends last-layer hidden state back to the coordinator.
- Focus Phase 4 on the model where the final worker owns output generation and keeps full-model residency, while participating in distributed prefill only for its later-layer route slice.
- Treat the final worker as the local decode owner after distributed prefill.
- Defer the alternative where the final worker sends last-layer output back to the coordinator unless a specific benchmark or deployment need makes it worth the extra transfer.

Evidence:

- The current distributed topology requires the coordinator to execute the early prefill slice before worker execution.
- Making the coordinator the full-resident decode node fights that topology and may force wasteful output/head or KV movement.
- The final worker already owns later layers and can own `N:output`; making it full-resident for local decode is a narrower change than decoupling the control-plane coordinator from execution topology.
- The current validated handoff boundary remains merged `DSV4`, but the practical direction should reduce toward sending the coordinator's earlier-layer KV state to the final worker rather than making the coordinator absorb all worker-owned state.

Why this changes the next steps:

- Phase 4 should decouple final-worker route ownership from final-worker model residency, not primarily coordinator route ownership from coordinator residency.
- The final worker should be allowed to load all layers and output head, while still advertising/executing only its assigned later-layer route range during distributed prefill.
- The handoff workflow should aim for:
  - distributed prefill over the existing route,
  - transfer or merge of the coordinator-owned early-layer KV into the final worker's full local decode session,
  - local-only decode on the final worker,
  - coordinator control/sampling integration only as needed by the chosen Phase 5 API shape.
- This keeps Phase 8 topology decoupling deferred.

Validation requirement:

- Once Phase 4/5 implementation work completes, validate both backend directions:
  - `CUDA -> Metal`
  - `Metal -> CUDA`
- Both directions should record correctness, payload/shard metadata, memory, timing, and failure cases in the existing issue artifacts before the workflow is treated as complete.

## 2026-06-05: Phase 4 workflow is viable; lifecycle stability remains in scope

Decision:

- Treat the core Phase 4 workflow question as answered: the full-resident final-worker handoff design works in both backend directions.
- Keep worker lifecycle stabilization inside Phase 4 scope before calling the workflow complete.
- Keep the startup memory guard enabled by default.

Evidence:

- `Metal -> CUDA` produced a passing final-worker handoff run with matching generated tokens.
- `CUDA -> Metal` also produced a passing run with matching generated tokens.
- DGX startup now rejects an unsafe second CUDA residency request before model mapping, preventing the earlier memory-exhaustion freeze class.
- Reused long-lived Metal workers produced inconsistent `CUDA -> Metal` outcomes until the Metal worker process was restarted.

Implications:

- The remaining question is no longer whether final-worker handoff can work.
- The remaining question is whether a reused worker process can survive coordinator reconnects and repeated handoff runs without stale state leaking into a later session.
- If the lifecycle bug is not fixed in Phase 4, the runbook must explicitly require fresh worker startup for authoritative `CUDA -> Metal` validation.

## 2026-06-05: Lifecycle hardening helped, but did not close repeated `CUDA -> Metal` instability

Decision:

- Keep the lifecycle fixes landed so far.
- Do not mark Phase 4 lifecycle stabilization complete yet.
- Continue treating repeated `CUDA -> Metal` validation as an open technical risk, not just an operator-runbook problem.

Evidence:

- Added hardening:
  - unique coordinator session ids,
  - epoch-gated stale worker data connections,
  - GPU synchronize before session free,
  - Metal runtime scratch reset when worker sessions are cleared.
- Those changes removed the simplest stale-session explanation, but repeated `CUDA -> Metal` validation still failed in two ways:
  - a fresh-worker run later diverged at generated step `12`,
  - the immediate next run on the same worker again diverged from the first generated token.

Implications:

- The remaining instability is no longer well described as session-id reuse alone.
- The next investigation should focus on Metal runtime state reuse across worker reconnects and on any remaining distributed/local decode variance that only appears after prior worker activity.

## 2026-06-05: Close Phase 4 with a fresh-Metal-worker validation rule

Decision:

- Mark Phase 4 complete.
- Treat the reused-Metal-worker `CUDA -> Metal` rerun mismatch as a Phase 3.5-class variance caveat, not a Phase 4 workflow blocker.
- Keep the startup memory guard and lifecycle hardening changes.
- Use a fresh Metal worker process for strict `CUDA -> Metal` parity validation in later phases unless and until reuse determinism becomes a specific goal.

Evidence:

- `Metal -> CUDA` passed three back-to-back sessions on one reused CUDA worker with identical generated tokens.
- `CUDA -> Metal` also continued to reproduce on a reused Metal worker, but the new diagnostic harness showed two important properties:
  - the divergent token streams were coherent text, not gibberish,
  - the first mismatch in both sampled reruns was a near-top1 logit flip with full `top5`/`top10` overlap.
- Example shapes:
  - mismatch at step `0`: handoff `#`, reference `This`
  - mismatch at step `11`: handoff ` (\``, reference ` (`

Implications:

- Phase 4 answered its actual question: the final-worker residency and in-process handoff workflow is viable and benchmarkable.
- The remaining reused-Metal-worker drift should not block Phase 5+ work unless those phases require strict repeated token parity on a long-lived Metal worker.
- Research documentation should now describe Phase 4 as complete with an operational validation caveat, not as an open residency/workflow question.
