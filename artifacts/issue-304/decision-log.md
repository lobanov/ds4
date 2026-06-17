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

## 2026-06-05: Phase 5 should ship the worker-owned CLI path first, without KV pipelining

Decision:

- Treat the current worker-owned `--local-decode` CLI path as the Phase 5
  implementation baseline.
- Keep KV pipelining out of Phase 5.
- Keep the existing `LOCAL_GENERATE` protocol surface for now; do not add a
  new catch-up or decode message family just to finish the first user-visible
  path.

Evidence:

- `--local-decode` now:
  - parses in the shared distributed CLI layer,
  - requires worker role plus `N:output`,
  - implies full worker residency,
  - and is advertised in HELLO/route metadata.
- The coordinator now pushes its own KV shard directly from the live session
  with stream-based `DSVL` helpers; the temp-file detour is removed from the
  Phase 5 handoff path.
- Reusable sessions now have coordinator catch-up from returned worker token
  ids, while one-shot CLI can ignore catch-up and exit after printing.
- Plain CLI measurements on 2026-06-05 showed:
  - `Metal -> CUDA`, full `README.md`: prefill `603.15 tok/s`, KV handoff
    `0.345 s` for `105,725,160` bytes, generation `13.08 tok/s`
  - `CUDA -> Metal`, fresh Metal worker, full `README.md`: prefill
    `589.93 tok/s`, KV handoff `0.350 s` for `107,097,320` bytes,
    generation `30.33 tok/s`
  - pure local Mac baseline, full `README.md`: generation `34.78 tok/s`

Why this changes the next steps:

- The current bottleneck is not KV handoff throughput. At roughly `100 MiB`
  the handoff stayed near `0.35 s`, which is small relative to the
  `23-25 s` distributed prefill runs.
- The next product-level gap is not transport but surface coverage:
  the worker-owned path was initially wired into greedy one-shot coordinator
  runs, not yet into the normal sampled distributed eval path.
- That makes the most defensible next task explicit:
  extend the user-facing decode surfaces only after preserving the current
  simple worker protocol and measured handoff cost.

## 2026-06-05: Sampled one-shot local-decode is enabled, and the remaining multi-turn variance starts after follow-up sync

Decision:

- Treat sampled one-shot coordinator handoff as implemented for Phase 5.
- Reclassify the main reusable-session issue from "catch-up is broken" to
  "follow-up sync on reused handoff state still diverges from a fresh
  full-transcript session".

Evidence:

- Plain CLI sampled one-shot on `Metal -> CUDA` now succeeds with:
  - `--temp 0.7 --top-p 0.95 --min-p 0.05`
  - route formation,
  - KV handoff log emission,
  - and sampled worker-owned output text.
- A coordinator bookkeeping bug was fixed: after worker-owned decode, the
  coordinator now keeps the worker's final logits even when no full logits
  trace buffer is requested.
- The DGX-backed multi-turn diagnostic now shows two distinct boundaries:
  - immediately after turn-1 worker decode plus coordinator catch-up:
    - reused top1 `1116`, fresh top1 `1116`
    - `top5=5/5`, `top10=10/10`, `top20=20/20`
    - `rms=0.27032664`
  - after syncing the next user follow-up on top of that reused state:
    - reused top1 `8474`, fresh top1 `267`
    - `top5=4/5`, `top10=6/10`, `top20=16/20`
    - `rms=0.99836105`

Why this changes the next steps:

- The broad "coordinator catch-up is wrong" diagnosis was too coarse.
- The remaining Phase 5 reusable-session problem is narrower and more
  useful: something in the next distributed sync/reused-state path is
  shifting the frontier before the second handoff.
- That means further investigation should focus on reused-session sync over
  an already-caught-up state rather than on the basic worker-owned
  generation or token replay mechanics.

## 2026-06-05: Normal sampled eval/server delegation is now enabled through local decode

Decision:

- Treat the normal distributed `ds4_session_eval(token)` surface as in
  Phase 5 scope and now implemented through worker-owned local decode.
- Keep the remaining Phase 5 blocker focused on reusable-session variance.

Evidence:

- After sync, the coordinator now lazily activates local decode by pushing
  its KV shard and seeding the worker with the current logits.
- Subsequent `ds4_session_eval(token)` calls use the existing
  `LOCAL_GENERATE` RPC with a forced-token one-step request, then catch up
  the coordinator's local slice and install the returned next logits.
- Validation passed on:
  - `Metal -> CUDA` CLI `--dump-logprobs`
  - `CUDA -> Metal` CLI `--dump-logprobs`
  - `Metal -> CUDA` `ds4_server`
  - `CUDA -> Metal` `ds4_server`
- The symmetric sampled multi-turn result on `CUDA -> Metal` still shows
  follow-up-sync frontier drift, but it is milder than the `Metal -> CUDA`
  case because the second-turn sampled tokens still matched exactly for the
  tested seed.

Why this changes the next steps:

- Phase 5 no longer has a major surface-coverage gap between one-shot and
  ordinary sampled decode.
- The remaining work is now primarily about reusable-session parity under
  follow-up sync, not about missing API wiring.

## 2026-06-06: `ds4-eval` must use distributed handoff for `--nothink` local-decode matrices

Decision:

- Treat `ds4-eval` local-decode coordinator runs as invalid unless they use
  the same worker-owned handoff workflow already exercised by the plain
  `ds4` CLI.
- Restrict the first evaluator change to `--nothink` cases, where the
  evaluator does not need token-by-token think-budget control.

Evidence:

- The first Phase 5 `ds4-eval` matrix runs used the ordinary
  `ds4_session_sample()` + `ds4_session_eval()` loop even when the route
  advertised worker local decode.
- That loop only exercised the already-existing lazy activation plus
  one-token forced `LOCAL_GENERATE` eval path, not the worker-owned
  one-shot handoff path used by:
  - plain `ds4 --role coordinator ... --local-decode` benchmarks,
  - the runbook validation commands,
  - and the previously recorded `30.10` to `30.33 tok/s`
    `CUDA -> Metal` local-decode results.
- After patching `ds4-eval` to use the distributed handoff API for eligible
  `--nothink` routes and rebuilding on DGX, a fresh `Q1`
  `CUDA -> Metal --local-decode` smoke passed through the intended path with:
  - `local_decode_expected: yes`
  - summary `local_decode_active_any_case: yes`
  - `generated_tokens: 539`
  - `decode_sec: 54.426`
  - `generated_tps: 9.903`

Why this changes the next steps:

- The evaluator-path bug is fixed: `ds4-eval` is no longer measuring the
  wrong local-decode workflow for eligible `--nothink` coordinator cases.
- The remaining throughput gap versus plain CLI (`~9.9 tok/s` in the smoke
  versus `~30 tok/s` in the CLI benchmark) therefore moves to the protocol
  level, not the evaluator loop.
- The next performance fix should target the worker local-generate RPC
  payload shape, especially the current always-on per-token logits-trace
  allocation/copy/return path, before drawing backend-level conclusions
  from evaluator throughput.

## 2026-06-06: Long `ds4-eval --nothink` local-decode runs were failing on the 60s distributed socket timeout

Decision:

- Treat the remaining `ds4-eval` `CUDA -> Metal --local-decode --nothink`
  blocker as a distributed socket-timeout issue, not as a silent
  coordinator crash or a broken handoff state.
- Raise the default distributed socket timeout from `60` to `600` seconds
  while preserving `DS4_DIST_SOCKET_TIMEOUT_SEC` as an override.

Evidence:

- An explicit DGX coordinator run with the intended evaluator path:
  - `./ds4-eval --ctx 16384 --plain --temp 0 --seed 1 --nothink`
  - `--tokens 4096 --questions 6 --role coordinator --layers 0:21`
  - `--listen 10.77.0.2 1241`
  - local Metal worker on the Mac:
    `./ds4 --ctx 16384 --role worker --layers 22:output --local-decode --coordinator 10.77.0.2 1241`
  failed immediately on case 1 with:
  - coordinator trace error:
    `failed to read frame header: Resource temporarily unavailable`
  - worker stderr:
    `distributed worker: protocol error: failed to read frame header: Resource temporarily unavailable`
- The failure reproduced at about the 60-second mark, which matches the
  existing default set in `dist_set_socket_low_latency()`.
- Re-running the exact same explicit commands with
  `DS4_DIST_SOCKET_TIMEOUT_SEC=600` on both coordinator and worker cleared
  the blocker:
  - case 1 passed in `113.1 s`
  - case 2 passed in `112.1 s`
  - case 3 passed in `112.0 s`
  - case 4 passed in `113.8 s`
  - case 5 passed in `113.3 s`
  - case 6 passed in `111.9 s`
  - full result: `6/6 passed`, runtime `00h:11m`
- After changing the default timeout to `600`, the same explicit no-env
  command line also passed a fresh one-question verification:
  - `1/1 passed`
  - case 1 runtime `113.8 s`

Why this changes the next steps:

- Phase 5 is no longer blocked on the evaluator dying in non-thinking
  distributed-prefill/local-decode mode.
- The remaining work returns to performance and reusable-session parity:
  - the evaluator path is now stable enough for longer local-decode runs,
  - but one-shot local-decode decode time is still large enough that socket
    policy must account for it,
  - and the local-generate RPC payload still carries avoidable overhead.

## 2026-06-07: Close Phase 5 on the fresh-worker worker-owned path

Decision:

- Close Phase 5 on the implemented worker-owned `--local-decode` workflow.
- Treat fresh-worker one-shot and evaluator evidence as the authoritative
  acceptance surface.
- Carry forward reusable-session follow-up drift as a post-Phase-5 caveat
  unless it turns into a concrete stale-KV, token-hash, or session-integrity
  failure.

Evidence:

- Plain CLI worker-owned local decode already passed in both backend
  directions with measured KV handoff timing.
- `ds4-eval --nothink` now uses the intended worker-owned handoff path and the
  timeout fix lets long answers complete.
- The real DGX/Mac distributed regression now also passes:
  - local worker:
    `./ds4 --ctx 2048 --role worker --layers 22:output --coordinator dgx-direct 1234 --local-decode`
  - DGX coordinator test:
    `DS4_RUN_DISTRIBUTED_TEST=1 DS4_TEST_DISTRIBUTED_LISTEN_HOST=0.0.0.0 DS4_TEST_DISTRIBUTED_LISTEN_PORT=1234 DS4_TEST_DISTRIBUTED_ROUTE_WAIT_MS=60000 ./ds4_test --local-decode-push`
  - observed pass line:
    `local_decode_active=yes first_handoff=2 reuse_eval=1 second_handoff=1`
- The authoritative `92`-question evaluator runs on `2026-06-06` scored:
  - local Metal: `67/92`
  - local CUDA: `69/92`
  - `CUDA -> Metal --local-decode`: `65/92`
- `make test` now passes locally with the default environment, and the new
  distributed regression remains opt-in as intended.
- That gap is small enough to treat as evaluation variance for Phase 5
  closeout rather than as evidence that the worker-owned handoff path is still
  routing incorrectly.

Why this changes the next steps:

- Phase 5 is now closed.
- Follow-on work moves out of closeout and into later phases:
  - reusable-session follow-up parity,
  - incremental/local-decode streaming UX,
  - and `LOCAL_GENERATE` protocol overhead reduction.

## 2026-06-08: Start Phase 5.5 from a fixed four-cell reusable-session matrix

Decision:

- Bound Phase 5.5 to a fixed reusable-session surface instead of chasing
  bit-exact parity everywhere.
- Keep the authoritative fresh-worker worker-owned local-decode workflow as
  the primary ship gate.
- Use this four-cell matrix as the reusable-session accuracy surface:
  - `Metal -> CUDA` greedy multi-turn
  - `Metal -> CUDA` sampled normal-eval multi-turn
  - `CUDA -> Metal` greedy multi-turn
  - `CUDA -> Metal` sampled normal-eval multi-turn
- Do not close Phase 5.5 until:
  - the fresh-worker Phase 5 surface stays green,
  - every remaining miss on that matrix is attributed as either
    backend-baseline variance, fixture drift, or handoff/reuse-specific
    variance,
  - and no integrity class is involved
    (`token hash`, `payload layout`, `stale KV`, `session mix-up`).
- Do not require bit-exact cross-backend logits or perfect agreement with
  the current `local-golden.vec` fixture as a Phase 5.5 exit condition.

Evidence:

- A clean `2026-06-08` rerun on rebuilt binaries produced:
  - `Metal -> CUDA` greedy: fail
    - follow-up frontier `8474` vs `267`
    - `top5=4/5`, `top10=6/10`, `top20=16/20`
    - `rms=0.99836105`
  - `Metal -> CUDA` sampled: fail
    - follow-up frontier `8474` vs `8474`
    - `top5=4/5`, `top10=8/10`, `top20=18/20`
    - `rms=0.75249523`
    - turn-two sampled tokens diverged
  - `CUDA -> Metal` greedy: fail
    - follow-up frontier `8474` vs `267`
    - `top5=4/5`, `top10=7/10`, `top20=15/20`
    - `rms=0.89044636`
  - `CUDA -> Metal` sampled: pass with bounded variance
    - follow-up frontier `8474` vs `8474`
    - `top5=4/5`, `top10=8/10`, `top20=15/20`
    - `rms=0.81043321`
    - turn-two sampled tokens still matched exactly
- The DGX coordinator reruns also exposed a separate operational caveat:
  the CUDA startup memory guard sometimes rejected the `0:21` coordinator
  slice even while `nvidia-smi` only showed desktop graphics processes.
  That required `DS4_DISABLE_STARTUP_MEMORY_GUARD=1` for reruns, but it is
  not itself an accuracy result.

Why this changes the next steps:

- Phase 5.5 should focus on the reused-state follow-up path, not on the
  already-closed fresh-worker handoff workflow.
- The current matrix is narrow enough to make progress measurable and broad
  enough to keep the work from collapsing into one lucky route or seed.

## 2026-06-08: Greedy Phase 5.5 mismatches are currently prefill-vs-decode, not reuse-vs-fresh-decode

Decision:

- Reclassify the current greedy Phase 5.5 mismatch from
  "reused session drift" to
  "fresh full-transcript prefill differs from decode replay over the
  assistant-generated turn".
- Keep the greedy cells open for Phase 5.5, but do not treat them as
  evidence of stale reused state or broken catch-up integrity.

Evidence:

- An opt-in replay diagnostic was added to
  `tests/issue304_phase5_multiturn`.
- On `Metal -> CUDA`, greedy:
  - reused vs fresh full-transcript prefill still failed at the follow-up
    seed frontier (`8474` vs `267`, `rms=0.90186453`)
  - reused vs fresh decode replay matched exactly:
    - post-turn1 `top5=5/5`, `top10=10/10`, `top20=20/20`, `rms=0`
    - post-followup seed frontier `top5=5/5`, `top10=10/10`,
      `top20=20/20`, `rms=0`
- On `CUDA -> Metal`, greedy:
  - reused vs fresh full-transcript prefill still failed at the follow-up
    seed frontier (`8474` vs `267`, `rms=0.90496105`)
  - reused vs fresh decode replay also matched exactly:
    - post-turn1 `top5=5/5`, `top10=10/10`, `top20=20/20`, `rms=0`
    - post-followup seed frontier `top5=5/5`, `top10=10/10`,
      `top20=20/20`, `rms=0`

Why this changes the next steps:

- The current four-cell matrix is still the right Phase 5.5 surface, but the
  leading greedy explanation is now narrower and more actionable.
- The next investigation should compare full-transcript prefill of
  assistant-generated tokens against token-by-token decode replay of the
  same tokens, rather than continuing to frame the issue primarily as
  reused-session corruption.

## 2026-06-08: Close Phase 5.5 as bounded prefill-vs-decode variance

Decision:

- Close Phase 5.5.
- Accept the remaining reusable-session differences as bounded
  prefill-vs-decode variance rather than an open handoff or reused-state
  correctness defect.

Evidence:

- The authoritative fresh-worker Phase 5 surfaces remain green:
  - plain `ds4` worker-owned local decode
  - normal token-by-token local-decode activation
  - `ds4-eval --nothink`
  - and `ds4_test --local-decode-push`
- Distributed replay diagnostics showed the reused session is self-consistent
  with a fresh decode-replay reference:
  - `Metal -> CUDA`, greedy: post-turn1 `rms=0`, post-followup seed `rms=0`
  - `Metal -> CUDA`, sampled: post-turn1 `rms=0`, post-followup seed `rms=0`
  - `CUDA -> Metal`, greedy: post-turn1 `rms=0`, post-followup seed `rms=0`
- New local-only same-backend proof removed distributed handoff from the
  equation entirely:
  - pure local Metal, greedy:
    - follow-up seed frontier `267` vs `8474`
    - `top5=4/5`, `top10=6/10`, `top20=16/20`
    - `rms=0.94414026`
    - turn-two greedy tokens diverged
  - pure local CUDA, greedy:
    - follow-up seed frontier `267` vs `8474`
    - `top5=4/5`, `top10=5/10`, `top20=14/20`
    - `rms=0.91843081`
    - turn-two greedy tokens diverged
  - pure local Metal, sampled:
    - `top5=4/5`, `top10=8/10`, `top20=17/20`
    - `rms=0.68071556`
    - turn-two sampled tokens still matched exactly
  - pure local CUDA, sampled:
    - `top5=4/5`, `top10=8/10`, `top20=15/20`
    - `rms=0.84279537`
    - turn-two sampled tokens still matched exactly
- No Phase 5.5 rerun localized the remaining differences to:
  - token-hash mismatch
  - payload layout mismatch
  - stale KV/session contamination
  - or stale-logits bookkeeping

Why this closes the phase:

- The remaining matrix differences are now attributed.
- They reproduce on the same backend without distributed handoff.
- They align with the expected consequence of different numerical
  trajectories between full-transcript prefill and token-by-token decode
  replay.
- The next work belongs to benchmarking and product-level tolerance
  decisions, not to continued correctness chasing under Issue 304.

## 2026-06-08: Re-scope Phase 6 to profiling before any KV-pipeline work

Decision:

- Re-scope Phase 6 from "implement pipelined KV return" to
  "profile bottlenecks across context lengths and session shapes".
- Keep `ds4-eval` out of that phase; evaluator-specific performance work
  is tracked separately.

Evidence:

- Existing Phase 5 CLI timings already show KV handoff is a small tail on
  the practical long-prompt workflow:
  - about `0.35 s` for roughly `100 MiB` of returned KV,
  - versus about `23-25 s` distributed prefill on the full
    `README.md` frontier.
- Earlier artifact tables also show merged stage/load overhead remains
  small relative to prefill on the representative long-frontier cases.
- Phase 5.5 closed with the explicit conclusion that the next work should
  be benchmarking and product-level tolerance decisions, not more
  correctness chasing or protocol expansion.

Why this changes the next steps:

- The remaining unknown is no longer "can pipelining work?" but "what
  actually dominates user-visible latency on short, medium, long, and
  follow-up workloads?"
- The right next artifact is an authoritative benchmark matrix for
  single-turn and multi-turn worker-owned local-decode sessions.
- Pipelined KV return should only be reopened later if the profiling pass
  shows handoff is a material contributor rather than a minor tail.

## 2026-06-08: Phase 6 profiling confirms direct-link bottlenecks in both directions

Decision:

- Close Phase 6 on the direct-link deployment target.
- Keep KV pipelining deferred for now.
- Treat distributed prefill and follow-up sync or decode cost as the next
  higher-value optimization targets if performance work resumes.

Evidence:

- Fresh June 8, 2026 `CUDA -> Metal` one-shot measurements on the direct-link
  `10.77.0.1 <-> 10.77.0.2` path show:
  - short:
    - sync `1.189 s`
    - handoff `0.022 s`
    - decode `0.209 s`
  - medium (`README` 4 KiB slice):
    - sync `3.513 s`
    - handoff `0.097 s`
    - decode `0.211 s`
  - long (`long_code_audit`):
    - sync `9.469 s`
    - handoff `0.118 s`
    - decode `0.217 s`
  - very long (`README.md`):
    - sync `36.979 s`
    - handoff `0.239 s`
    - decode `0.261 s`
- Fresh June 8, 2026 `Metal -> CUDA` one-shot measurements on the same
  direct-link path show:
  - short:
    - sync `0.851 s`
    - handoff `0.022 s`
    - decode `0.526 s`
  - medium (`README` 4 KiB slice):
    - sync `2.955 s`
    - handoff `0.049 s`
    - decode `0.561 s`
  - long (`long_code_audit`):
    - sync `9.099 s`
    - handoff `0.230 s`
    - decode `0.638 s`
  - very long (`README.md`):
    - sync `38.579 s`
    - handoff `0.338 s`
    - decode `0.601 s`
- Reused-session follow-up on the same matrix stays below the KV-pipelining
  materiality bar in both directions:
  - `CUDA -> Metal` remains mostly sync-bound
  - `Metal -> CUDA` is more decode-heavy, not handoff-dominated
- The longest direct-link handoffs moved about `106 MiB` in about:
  - `0.239 s` to Metal, about `425 MiB/s`
  - `0.338 s` to CUDA, about `301 MiB/s`

Why this changes the next steps:

- The direct-link deployment target still behaves like the original Phase 5
  intuition in both route directions.
- Distributed prefill remains the dominant one-shot cost by a large margin.
- Route-direction differences are real, but they are mostly decode-backend
  differences rather than transport differences.
- Follow-up turns remain too small, and too mixed between sync and decode, for
  repeated KV return to be the clearest next bottleneck to attack.

Default follow-on:

- Keep the published research artifacts centered on the direct-link matrix as
  the authoritative Phase 6 result.
- Keep KV pipelining deferred for the current worker-owned local-decode
  workflow.
- If more optimization work is needed, target distributed prefill or the
  follow-up sync path first, and only reopen KV pipelining if future data
  crosses the materiality rule in `PLAN.md`.

## 2026-06-12: Close Phase 7-8 on the shipped worker-owned local-decode workflow

Decision:

- Close the combined Phase 7-8 hardening and documentation pass.
- Treat Issue 304 as closed on the current worker-owned local-decode workflow.
- Keep topology decoupling and any later KV-pipeline work explicitly deferred.

Evidence:

- Local `./ds4_test --dist-cli-parse` passed after the CLI validation test was
  tightened to assert exact `--local-decode` rejection text.
- DGX `./ds4_test --dist-cli-parse` also passed after syncing the updated test
  source.
- Fresh June 12, 2026 DGX/Mac positive handoff smoke passed on the rebuilt
  code:
  - route `local 0:21 -> 10.77.0.1:59183 Q2 22:output`
  - `local_decode_active=yes`
  - first handoff `2`
  - reuse eval `1`
  - second handoff `1`
  - KV handoff `0.020 s`
  - decode windows `0.057 s` and `0.026 s`
- Fresh June 12, 2026 DGX/Mac runtime negative-path smoke also passed:
  - worker owned `22:output` but did not advertise `--local-decode`
  - route `local 0:21 -> 10.77.0.1:58761 Q2 22:output`
  - exact rejection:
    `distributed handoff requires worker local-decode capability`
- Worker-side diagnostics for snapshot save, snapshot load, and local-generate
  mismatches are now field-specific instead of a generic worker-state mismatch.
- The DGX runbook caveat is now explicit: remote `ds4_test` commands should set
  `DS4_TEST_MODEL=...` because `ds4flash.gguf` there may be a broken symlink to
  a macOS path.

Why this closes the phase:

- The shipped workflow now has both:
  - a passing positive distributed handoff smoke,
  - and a passing runtime capability reject for a real multi-host negative path.
- The preserved fail-closed boundaries are documented in one place and match
  the tested behavior.
- The remaining work is no longer about closing correctness or integrity gaps in
  the current workflow.

Deferred follow-on:

- Topology decoupling remains the next architectural follow-on.
- KV pipelining stays deferred unless future profiling on the intended topology
  shows the existing handoff cost has become materially limiting.
