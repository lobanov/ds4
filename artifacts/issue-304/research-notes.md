# Issue 304 Research Notes

This file tracks phase-wise findings for the staged investigation in `PLAN.md`.

## Phase status

| Phase | Status | Summary |
| --- | --- | --- |
| Phase 0 | Complete | The DGX/Mac baseline ran end-to-end, and the earlier merged-`DSV4` stage failure was narrowed to a worker/coordinator `ctx` mismatch rather than a backend or route-format incompatibility. |
| Phase 1 | Complete | The payload resume matrix is now filled: `Metal -> Metal`, `CUDA -> CUDA`, and `Metal -> CUDA` passed, while `CUDA -> Metal` preserved restore-point logits but diverged during subsequent greedy decode. |
| Phase 2 | Complete | Distributed prefill -> merged `DSV4` -> fresh local Metal load now validates end-to-end on the DGX/Mac route. Handoff logits and 16-token greedy continuation matched, but forced-token logits still diverged at the first post-load eval step. |
| Phase 3 | Complete | Same-backend Phase 3 parity was measured and rejected. The official-vector gate kept passing, but distributed-prefill-to-local decode did not match a fresh fully local Metal baseline closely enough, and the long local-golden continuation failed outright on same-backend parity. |
| Phase 3.5 | Complete | The six-route worst@5 matrix is now measured. Official top-logprob variance is route-dependent on short prompts, resumed payload routes remain the weakest parity cells, and the stored local-golden fixture is not anchored purely to the local Metal route. |
| Phase 4 | Complete | Final-worker full-resident handoff is now proven end to end. Both backend directions work; `Metal -> CUDA` is stable across repeated reused-worker sessions, and `CUDA -> Metal` reused-worker drift is currently classified as a Phase 3.5-style near-top1 variance caveat rather than a workflow blocker. |
| Later phases | Deferred | Pipelined KV return, topology decoupling, and broader optimization work stay deferred until the practical handoff workflow exists and can be benchmarked realistically. |

## Phase 0: Establish distributed baseline

### What was completed

- Added a dedicated local baseline tool: `tests/issue304_phase0_local`.
- Added a DGX/Mac coordinator helper: `tests/issue304_phase0_dgx`.
- Added a build/run target in `Makefile` for the tool.
- Defined the local smaller reproducible topology and the DGX issue topology in the runbook.
- Created `perf-breakdown.md` so Phase 0 timing results have a fixed home before DGX runs start.
- Ran the authoritative DGX worker + Mac coordinator issue-topology baseline on `README.md`.

### Findings

1. The current CLI/API surface is enough for a narrow two-host baseline harness.
   - Route readiness, distributed prefill, distributed decode, and staged payload save are all reachable through existing APIs.
   - A small helper binary was enough to exercise the route and record the relevant timings.

2. Same-host loopback distributed baseline is still blocked by current process policy.
   - The local tool launches a worker process and then opens a coordinator engine.
   - The worker exits before route formation because `ds4_engine_open()` refuses a second `ds4` process on the same host.
   - Observed worker-side failure: `another ds4 process is already running ... refusing to start`.

3. The DGX/Mac route formed cleanly and produced stable distributed timings on the issue topology.
   - Route: `local 0:21 -> 10.77.0.2:<worker-data-port> Q2 22:output`.
   - Output owner: worker.
   - Prompt: `README.md`, 14,318 tokens after chat formatting.
   - Prefill: about `38 s`, about `376 tok/s`.
   - Decode was measured before the `ctx` mismatch was diagnosed, and that earlier distributed decode number should be treated as route-performance context only, not as part of the corrected save-path validation run.

4. The earlier immediate post-prefill merged `DSV4` stage failure was caused by mismatched `ctx`, not by Metal/CUDA shard incompatibility.
   - The failing rerun used coordinator `ctx=16384` and worker default `ctx=32768`.
   - The save path now reports the first mismatching header field explicitly:
     - `distributed KV shards use different layouts: field=ctx local=16384 remote=32768 remote_layers=22:42`
   - This is consistent with current route admission logic, which accepts `worker_ctx >= coordinator_ctx` for execution.
   - It is also consistent with the merged save path, which requires exact equality of shard layout metadata.

5. Immediate post-prefill merged `DSV4` staging works on the mixed Metal/CUDA route when `ctx` matches.
   - Re-running the worker with `--ctx 16384` made immediate `ds4_session_stage_payload()` succeed.
   - The staged payload size was `221,006,660` bytes.
   - Parsed `DSV4` header:
     - `ctx_size=16384`
     - `prefill_cap=4096`
     - `raw_cap=4352`
     - `raw_window=128`
     - `comp_cap=4098`
     - `saved_tokens=14318`
     - `n_layer=43`
     - `head_dim=512`
     - `indexer_head_dim=128`
     - `vocab=129280`
     - `raw_live=128`

### Implication

The original broad conclusion was too strong. The mixed Metal/CUDA route does support immediate merged `DSV4` staging, but only when the worker session context matches the coordinator session context. The real actionable constraint is: Phase 2 reproductions must set the same `--ctx` on every distributed participant.

### Remaining caution

- The DGX worker repo was not updated during this pass. The local helper/build and the DGX worker binary were behaviorally compatible for the tested Phase 0 path, but the exact authoritative evidence here is:
  - route formation across those revisions worked,
  - immediate merged save worked once `ctx` matched,
  - not that every later Phase 2/3 path is already proven safe across revision skew.

## Phase 1: Prove whole-payload resume locally

### What was tested

- Added `./ds4_test --local-payload-resume`.
- Verified local `DSV4` save -> load -> resume on Metal.
- Verified 8-token greedy continuation equivalence after restore.
- Ran boundary frontiers at `1`, `3`, `4`, `5`, `127`, `128`, `129`.
- Added representative single-layer `DSVL` save/load/save byte-roundtrip smoke cases.

### Findings

1. Local same-backend `DSV4` resume is a valid correctness boundary.
   - Metal -> Metal resume matched exactly on every tested frontier.
   - Logit drift was `0` for all tested cases.
   - Greedy continuation matched exactly for all tested cases.

2. Raw-window boundary behavior is not the immediate blocker.
   - Resume remained exact at `raw_window - 1`, `raw_window`, and `raw_window + 1`.
   - That reduces the chance that later handoff failures will come from raw-ring reconstruction or cache-frontier serialization on the local path.

3. `DSVL` shard helpers appear healthy enough to use as building blocks.
   - Representative layer cases passed save -> load -> save byte identity checks.
   - Sampled layers:
     - `0` first layer
     - `21` middle compressed layer
     - `2` representative ratio-4/indexer layer
     - `42` final/output-adjacent layer

4. The current payload metadata is internally coherent on this backend.
   - Probe run observed:
     - `ctx_size=16384`
     - `prefill_cap=4096`
     - `raw_cap=4352`
     - `raw_window=128`
     - `comp_cap=4098`
     - `n_layer=43`
     - `head_dim=512`
    - `indexer_head_dim=128`
    - `vocab=129280`

5. Cross-backend whole-payload portability is asymmetric in the current evidence.
   - `CUDA -> CUDA`: pass.
   - `Metal -> CUDA`: pass.
   - `CUDA -> Metal`: fail.
   - The `CUDA -> Metal` failure is not a restore-point logit mismatch:
     - top1 matched
     - top5 overlap was `5/5`
     - top20 overlap was `20/20`
     - `rms=0`
     - `max_abs=0`
   - The first observed divergence happened during continued greedy decode at step `2`:
     - saved token `5655`
     - restored token `305`

6. Forced-token comparison narrows the `CUDA -> Metal` failure to post-load decode evolution, not token-selection branching.
   - The same CUDA-produced payload was loaded on CUDA and Metal.
   - Both sessions were then advanced with the exact same forced token sequence from the CUDA reference run.
   - The first logits divergence appeared immediately after the first forced eval step:
     - bad step: `1`
     - forced token before bad step: `3737`
     - top1 still matched (`14`)
     - top20 overlap dropped to `19/20`
     - `rms=0.304555088`
     - `max_abs=1.55702209`
   - This rules out the narrow explanation that the earlier divergence was only caused by each backend choosing a different token after load.

### What Phase 1 rules out

- “Local `DSV4` resume is fundamentally broken” is no longer a leading hypothesis.
- “Boundary frontiers immediately corrupt local resume state” is not supported by the current Metal results.
- “Representative `DSVL` helpers are obviously unusable” is not supported by the current local shard smoke.

### What Phase 1 does not prove

- Why `CUDA -> Metal` diverges immediately after the first identical forced post-load eval
- Distributed shard gather correctness
- Coordinator-side local decode after distributed prefill
- Whether a practical implementation can avoid a second full engine or model reload

### Implication for the goal

The goal is distributed prefill with local generation after handoff. Phase 1 reduces the problem to the real remaining unknowns:

- distributed gather/load correctness,
- classification of `CUDA -> Metal` post-restore decode divergence against same-backend and official/local baselines,
- and engine residency / local decode feasibility.

That means Phase 2 can use merged `DSV4` handoff as the first end-to-end correctness experiment with much lower ambiguity.

## Phase 2: Prove distributed prefill to local decode via existing payload path

### Current status

Complete. The dedicated handoff harness ran successfully on the DGX/Mac route after its local-decode step was changed to release the distributed engine before opening the full local one.

### Current working hypothesis

The intended correctness check was:

1. run distributed prefill,
2. stage the coordinator session as merged `DSV4`,
3. create a fresh full local non-distributed session,
4. load the merged payload,
5. compare local decode against continued distributed decode.

### Updated prerequisite

- Keep worker and coordinator session context sizes identical before treating any merged-save failure as a true shard-format bug.

### What was completed

- Added `tests/issue304_phase2_handoff`.
- Built the helper locally and on the DGX host after syncing the changed core files needed by the newer route-summary API surface.
- Ran the authoritative Mac coordinator + DGX worker topology with matching `ctx=16384`.
- Captured a staged merged payload, local load, greedy continuation comparison, forced-token trace comparison, and throughput split.

### Findings

1. The `DSV4`-first Phase 2 handoff path works end-to-end on the mixed Metal/CUDA route.
   - Distributed prefill completed on the `README.md` prompt.
   - Immediate merged stage succeeded.
   - Fresh local Metal session load succeeded from the merged payload.

2. The handoff checkpoint itself is exact enough to rule out a bad merged-save boundary.
   - Handoff logits matched exactly:
     - top1 match
     - top5 overlap `5/5`
     - top20 overlap `20/20`
     - `rms=0`
     - `max_abs=0`
     - nonfinite `0`

3. Greedy continuation from the resumed local session matched the distributed reference for the sampled window.
   - The helper compared `16` greedy tokens.
   - Result: exact match for all `16` steps.

4. The remaining observed drift is still post-load decode evolution, not stage/load correctness.
   - Forced-token trace still diverged at the first identical post-load eval step:
     - first bad step: `1`
     - forced token before bad step: `5`
     - top1 still matched (`420`)
     - top5 overlap `5/5`
     - top20 overlap `20/20`
     - `rms=0.0802887231`
     - `max_abs=0.507860184`

5. The expected throughput split was observed.
   - Distributed prefill: `38.183 s`, `374.99 tok/s`
   - Distributed decode: `16.28 tok/s`
   - Local decode after handoff: `30.32 tok/s`

### Implication

Phase 2 is no longer blocked on distributed gather, payload staging, or initial local resume. The feature concept is viable through the existing merged `DSV4` path. The remaining question is not whether every backend can produce bit-identical logits after resume; engine implementations can legitimately differ. The next question is whether distributed prefill followed by local decode matches fully local inference on the same decode backend and stays inside the existing official-vector/local-golden acceptance envelope.

### Open question carried forward

- `CUDA -> Metal` restores the immediate logits exactly, but diverges after the first identical forced post-load eval token.
- This remains unresolved and should be treated as a later investigation into post-load decode evolution, not as a blocker for considering Phases 0 and 1 complete.

## Phase 3: Handoff equivalence and drift isolation

### What was completed

- Added `tests/issue304_phase3_vectors` and built it locally and on `dgx-direct:~/ds4`.
- Verified the local and DGX GGUF hashes matched exactly:
  - `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Re-ran the local control gates:
  - `./ds4_test --logprob-vectors`: pass
  - `./ds4_test --local-golden-vectors`: pass
- Ran Phase 3 against the authoritative DGX/Mac topology in three aligned layout groups:
  - official vectors at `ctx=4096`, worker `DS4_METAL_PREFILL_CHUNK=2048`
  - official vectors at `ctx=16384`, worker `DS4_METAL_PREFILL_CHUNK=2048`
  - local golden vector at `ctx=5000`, worker `DS4_METAL_PREFILL_CHUNK=4096`
- Repeated each group five times with one-shot coordinator runs and short pauses between runs to avoid coordinator port reuse races.

### Findings

1. Phase 3 rejects the “cross-engine variance only” explanation.
   - The decode engine on the coordinator side stayed Metal for both the fresh local baseline and the resumed local handoff path.
   - Despite that same-backend comparison, the resumed handoff path still drifted materially from the fresh local baseline.

2. Official-vector acceptance is weaker than same-backend parity.
   - Every official-vector case passed the existing selected-token/top-logprob gate on all five runs.
   - The same runs still showed stable same-backend logit drift relative to a fresh local Metal baseline.
   - Worst observed same-backend parity over the five official runs:
     - `short_code_completion`: top20 overlap `17/20`, top64 `62/64`, `rms=0.356403291`, `max_abs=1.69900322`
     - `short_reasoning_plain`: top20 overlap `20/20`, top64 `62/64`, `rms=0.327032387`, `max_abs=1.77348614`
     - `short_italian_fact`: top20 overlap `19/20`, top64 `62/64`, `rms=0.21716401`, `max_abs=1.00239372`
     - `long_code_audit`: top5 overlap `4/5`, top20 `15/20`, top64 `55/64`, `rms=0.876981199`, `max_abs=4.63468361`

3. The long local-golden continuation fails same-backend parity decisively.
   - The resumed handoff still passed the existing local-golden fixture on every run:
     - top1 `4371`
     - top20 overlap `16-17/20`
     - top64 overlap `56/64`
     - top20 max abs up to `3.20196915`
   - But when compared directly to a fresh local Metal baseline over the next eight greedy steps, the worst-of-five parity was much looser:
     - top5 overlap `3/5`
     - top20 overlap `12/20`
     - top64 overlap `46/64`
     - `rms=2.1708498`
     - `max_abs=12.3051796`
     - top20 max abs `7.73109055`

4. Phase 3 also surfaced an operational constraint beyond `ctx`.
   - Merged `DSV4` staging requires worker and coordinator shard layouts to match on `prefill_cap`, not just on `ctx`.
   - For the official suite this required setting worker `DS4_METAL_PREFILL_CHUNK=2048` to match the strict local vector environment.
   - For the local-golden suite this required worker `DS4_METAL_PREFILL_CHUNK=4096`.

### Implication

Phase 3 did its job: the remaining drift is not just a CUDA-vs-Metal variance story. The handoff path itself differs from a same-backend fresh local Metal decode path after resumed evaluation begins. That remains an active caveat, but it no longer blocks the next engineering phase by itself; the next step is to build a practical fused residency-plus-workflow path and then judge the variance again under realistic use and benchmarks.

## Pointers

- [runbook.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/runbook.md)
- [compatibility-matrix.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/compatibility-matrix.md)
- [logit-comparisons.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/logit-comparisons.md)
- [shard-metadata.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/shard-metadata.md)

## Phase 3.5: Six-route variance matrix

### What was completed

- Added `tests/issue304_phase35_vectors` and `tests/issue304_phase35_matrix.py`.
- Synced the tree to `dgx-direct:~/ds4`, rebuilt locally and remotely, and ran the DGX-side helper and worker with `--power 50`.
- Completed worst@5 across all six routes for:
  - official vectors: `short_code_completion`, `short_reasoning_plain`, `short_italian_fact`, `long_code_audit`
  - local-golden frontier: `long_story_4096`

### Findings

1. Official-vector variance depends on the route/backend pair.
   - `short_code_completion` failed official acceptance on:
     - route `1` pure local CUDA
     - route `3` distributed generation `CUDA -> Metal`
     - route `5` resumed `CUDA -> Metal`
   - The same case passed on:
     - route `2` pure local Metal
     - route `4` distributed generation `Metal -> CUDA`
     - route `6` resumed `Metal -> CUDA`
   - The remaining official cases passed on all six routes.

2. Resumed payload routes remain the weakest official-route cells even when they pass the official gate.
   - `short_reasoning_plain`:
     - route `5`: `top64=61/64`, `rms=0.382280707`, `max_abs=2.11430359`
     - route `6`: `top20=18/20`, `top64=59/64`, `rms=0.504765928`, `max_abs=2.56024551`
   - `short_italian_fact`:
     - route `5`: `top5=4/5`, `top20=18/20`, `top64=56/64`, `rms=0.663591683`
     - route `6`: `top5=4/5`, `top20=18/20`, `top64=58/64`, `rms=0.688694775`
   - `long_code_audit`:
     - route `5`: `top5=4/5`, `top20=16/20`, `top64=50/64`, `rms=0.972514927`
     - route `6`: `top5=4/5`, `top20=15/20`, `top64=53/64`, `rms=0.903749108`

3. Pure CUDA generation also carries visible drift on longer prompts.
   - `long_code_audit` route `1` pure local CUDA still passed the official gate, but parity against its local reference was already loose:
     - `top5=4/5`
     - `top20=16/20`
     - `top64=52/64`
     - `rms=0.893455684`
     - `max_abs=4.59287167`
   - That matters because the resumed-route numbers are not emerging from nowhere; some longer-prompt variance is already present in the pure CUDA path.

4. The stored local-golden fixture is not “local Metal canonical”.
   - `long_story_4096` route `2` pure local Metal missed the fixture at worst@5:
     - `top5=3/5`
     - `top20=16/20`
     - `top64=46/64`
     - `top20_max_abs=5.2168293`
   - Routes `1`, `3`, `4`, `5`, and `6` all passed the coarse fixture, even though the resumed routes still showed loose parity against their direct-route references.

5. Phase 3 and Phase 3.5 together narrow the real blocker.
   - Phase 3 already showed that distributed-prefill-to-local Metal resume diverges from a fresh local Metal baseline.
   - Phase 3.5 adds that:
     - official acceptance can still hold across several visibly different parity profiles,
     - short-prompt failures are route-specific,
     - longer-prompt resumed routes are still among the weakest cells.

### Implication

The next classification boundary is clearer now:

- do not treat official-vector acceptance alone as proof of route equivalence,
- do not treat the stored local-golden fixture as a local-Metal oracle,
- and do not attribute all resumed-route drift to the handoff path alone without considering the existing longer-prompt CUDA variance.

The remaining work should separate:

- backend-specific generation variance on long prompts,
- fixture drift in `local-golden.vec`,
- and any additional error introduced specifically by distributed payload resume.

## Phase 5: CLI worker-owned local decode

### What was completed

- Landed the worker-owned `--local-decode` CLI/runtime path.
- Made `--local-decode` imply full worker residency and advertise through
  HELLO/route metadata.
- Replaced the Phase 5 handoff temp-file path with stream-based `DSVL`
  save/load helpers.
- Added coordinator catch-up from returned worker token ids for reusable
  sessions.
- Validated the path with the plain `ds4` CLI in both backend directions.

### Findings

1. The user-visible worker-owned path now exists without a harness.
   - One-shot greedy coordinator runs (`--temp 0`) can prefill over the
     distributed route, push the coordinator KV shard, and let the
     output-owning worker finish generation.

2. KV handoff cost is measurable but small relative to prefill.
   - `Metal -> CUDA`, full `README.md`:
     - prompt tokens `14,318`
     - KV bytes `105,725,160`
     - handoff `0.345 s`
     - prefill `603.15 tok/s`
   - `CUDA -> Metal`, fresh Metal worker, full `README.md`:
     - prompt tokens `14,524`
     - KV bytes `107,097,320`
     - handoff `0.350 s`
     - prefill `589.93 tok/s`

3. Local decode on the Mac side is close to the pure local baseline.
   - `CUDA -> Metal` worker-owned local generation reached `30.33 tok/s`.
   - Pure local Mac generation on the same `README.md` prompt reached
     `34.78 tok/s`.
   - That is close enough that Phase 5 no longer needs to justify itself on
     decode throughput alone.

4. The remaining surface gap is protocol usage, not handoff mechanics.
   - The worker-owned path now supports sampled one-shot coordinator runs
     through the existing `LOCAL_GENERATE` response path.
   - The normal token-by-token distributed eval/decode path now also
     delegates through worker-owned local decode by lazily activating the
     worker on first `ds4_session_eval()` after sync and then using
     forced-token one-step `LOCAL_GENERATE` calls for subsequent tokens.
   - This was validated on the plain CLI `--dump-logprobs` path and on
     `ds4_server` in both `Metal -> CUDA` and `CUDA -> Metal`.

5. The first reusable-session variance diagnosis was partly a stale-logits
   bookkeeping bug on the coordinator.
   - After worker-owned decode, the coordinator transcript/KV advanced but
     the coordinator often kept pre-handoff logits unless a full logits
     trace was requested.
   - Fixing that bug made the immediate post-catch-up frontier much closer
     to a fresh full-transcript session.

6. The remaining reusable-session drift now appears after follow-up sync,
   not immediately after catch-up.
   - On the DGX/Mac `0:21 -> 22:output` route, the post-turn1 reused-vs-fresh
     frontier now matches on top1 (`1116` vs `1116`) with `top5=5/5`,
     `top10=10/10`, `top20=20/20`, and `rms=0.27032664`.
   - After syncing the next user follow-up on top of that reused state, the
     frontier diverges again with a near-top1 flip:
     - reused top1 `8474` (`"They"`)
     - fresh top1 `267` (`"en"`)
     - `top5=4/5`, `top10=6/10`, `top20=16/20`
     - `rms=0.99836105`
   - That means the remaining Phase 5 reusable-session variance is no
     longer best described as "catch-up is broken". It is more narrowly a
     reused-state follow-up sync variance that appears after the next
     distributed prefill step.

7. The symmetric `CUDA -> Metal` sampled normal-eval route is milder than
   the `Metal -> CUDA` case.
   - `Metal -> CUDA`, sampled normal-eval multi-turn:
     - reused top1 `8474`, fresh top1 `267`
     - `top5=4/5`, `top10=8/10`, `top20=18/20`
     - turn-two tokens diverged
   - `CUDA -> Metal`, sampled normal-eval multi-turn with the same seed:
     - reused top1 `8474`, fresh top1 `8474`
     - `top5=4/5`, `top10=8/10`, `top20=15/20`
     - turn-two tokens still matched exactly
   - So the remaining reused-session follow-up-sync variance is present in
   both directions, but it is not equally severe on the sampled path.

8. The first `ds4-eval` local-decode matrix runs initially measured the
   wrong workflow.
   - The original Phase 5 `ds4-eval` matrix used the normal
     `ds4_session_sample()` + `ds4_session_eval()` loop even when the route
     advertised worker-owned local decode.
   - That meant the evaluator was exercising the "forced one-token
     `LOCAL_GENERATE`" path, not the worker-owned one-shot handoff path used
     by the plain `ds4` CLI benchmarks.
   - This explained the misleading early matrix result where
     `CUDA -> Metal --local-decode` looked materially slower than the plain
     CLI workflow.

9. `ds4-eval` now uses the proper worker-owned handoff path for eligible
   `--nothink` local-decode coordinator runs.
   - The evaluator was updated to call the distributed handoff API instead of
     always staying in the generic per-token decode loop.
   - A fresh `Q1` smoke on `CUDA -> Metal --local-decode --nothink` then
     passed through the intended handoff workflow:
     - route summary: `local 0:21 -> 192.168.1.218:51924 Q2 22:output`
     - `local_decode_expected: yes`
     - summary `local_decode_active_any_case: yes`
     - `generated_tokens: 539`
     - `decode_sec: 54.426`
     - `generated_tps: 9.903`

10. After correcting the evaluator workflow, the remaining throughput gap is
    in the current handoff RPC payload shape, not in evaluator control flow.
    - Plain CLI `CUDA -> Metal --local-decode` on the long README prompt had
      already shown `generation: 30.10 t/s` to `30.33 t/s`.
    - The corrected `ds4-eval` handoff smoke was still only about
      `9.9 t/s`.
    - Code inspection showed why: the worker local-generate RPC currently
      allocates, fills, and returns a full logits trace for every generated
      token, and the reported `decode_usec` includes that extra work.
    - So the remaining performance discrepancy is now best described as a
      protocol-overhead issue in the current `LOCAL_GENERATE` response path,
      not as evidence that the Mac local-decode backend is fundamentally
      slower than the plain local-decode CLI benchmark.

### Implication

Phase 5 has crossed the implementation threshold: the intended CLI workflow
is real, benchmarkable, and no longer dependent on the issue harnesses.
The next work should focus on:

- preserving correctness for reusable/sample-driven flows,
- making the evaluator and benchmark surfaces use the same worker-owned
  local-decode workflow where intended,
- and reducing local-generate protocol overhead before interpreting
  `ds4-eval` local-decode throughput as a backend limit.
