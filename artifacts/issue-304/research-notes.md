# Issue 304 Research Notes

This file tracks phase-wise findings for the staged investigation in `PLAN.md`.

## Phase status

| Phase | Status | Summary |
| --- | --- | --- |
| Phase 0 | Complete | The DGX/Mac baseline ran end-to-end, and the earlier merged-`DSV4` stage failure was narrowed to a worker/coordinator `ctx` mismatch rather than a backend or route-format incompatibility. |
| Phase 1 | Complete | The payload resume matrix is now filled: `Metal -> Metal`, `CUDA -> CUDA`, and `Metal -> CUDA` passed, while `CUDA -> Metal` preserved restore-point logits but diverged during subsequent greedy decode. |
| Phase 2 | Complete | Distributed prefill -> merged `DSV4` -> fresh local Metal load now validates end-to-end on the DGX/Mac route. Handoff logits and 16-token greedy continuation matched, but forced-token logits still diverged at the first post-load eval step. |
| Phase 3 | Not started | Re-scoped to prove distributed-prefill-to-local decode matches fully local inference on the same decode engine against the official-vector/local-golden correctness surfaces, while classifying any remaining cross-engine forced-logit drift. |
| Later phases | Not started | Engine residency, user-facing workflow, and pipelined KV return remain open after Phase 3 classification. |

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

### Current status

Not started.

### Revised goal

- Use `tests/test-vectors/official.vec` and the existing local golden-vector checks as the correctness reference.
- Compare distributed-prefill-to-local decode against fully local inference on the same decode backend before treating cross-engine drift as a defect.
- Keep forced-token traces as diagnostic evidence, but do not require bit-exact cross-engine logits if selected tokens and official/local-golden gates remain stable.

### Acceptance direction

- Distributed prefill -> merged `DSV4` -> local decode should match fully local inference on the same decode backend for the official-vector prompt set.
- The same path should satisfy the existing official top-logprob tolerance and local-golden drift thresholds, allowing the already documented `long_memory_archive` API/official graph caveat.
- Remaining `CUDA -> Metal` forced-logit drift should be classified as accepted backend variance only after the same-backend distributed handoff checks pass.

## Pointers

- [runbook.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/runbook.md)
- [compatibility-matrix.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/compatibility-matrix.md)
- [logit-comparisons.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/logit-comparisons.md)
- [shard-metadata.md](/Users/lobanov/Projects/ds4/artifacts/issue-304/shard-metadata.md)
