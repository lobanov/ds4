# Local generation with distributed prefill

## Goal

Use distributed prefill, but allow token generation to run on a single local node after prefill completes.

## Research Notes From Issue #304

Source: <https://github.com/antirez/ds4/issues/304> and its comment thread.

### Feasibility notes

- Existing distributed prefill already pipelines chunks between workers. The proposed extra work is returning KV cache chunks for the layers prefetched on the remote node back to the local decode node.
- With the current 4096-token distributed chunk size and the README estimate of about 26 GB KV for a 1M-token context, a full context is about 244 chunks and about 106 MB of KV per chunk.
- Activation transfer for one chunk is about 64 MB (`4096 tokens * hidden size 4096 * 4 bytes`), so shipping activation plus KV would raise the per-chunk payload from about 64 MB to about 170 MB.
- On a practical 2.5 GbE link at about 280 MB/s with about 1-2 ms RTT, that larger payload adds about 0.4-0.5 s of transfer latency.
- Because prefill is feed-forward and already pipelined, the working assumption is that efficient KV return pipelining would add that latency roughly once per prefill, not once per chunk.
- Based on this rough model, the feature looks feasible over commodity Ethernet and is unlikely to require Thunderbolt to be useful.
- The issue notes that reducing `--dist-activation-bits` reportedly does not materially improve performance even when the payload is halved, which suggests the current distributed path is not primarily bandwidth-bound at this scale.

### Direct-link test notes

- Test date from issue comments: June 1, 2026.
- Tested on `main@ba00a8a`.
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
- Topology:
  - DGX Spark worker: `./ds4 --role worker --layers 22:output --coordinator 10.77.0.1 1234`
  - MacBook M5 Max 128GB coordinator: `./ds4 --role coordinator --layers 0:21 --listen 10.77.0.1 1234 --prompt-file README.md --nothink -n 100`
- Measured result with Spark on `22:output`: prefill about `598.73 t/s`, generation about `14.98 t/s`.
- Measured result with Spark on `22:42`: prefill about `592.16 t/s`, generation about `15.81 t/s`.

### Conclusions from current testing

- Heterogeneous distributed inference already works across CUDA and Metal.
- Prefill benefits from splitting work across the two devices even over a direct 5 GbE link with about 1-2 ms ping.
- Decode remains slow when generation stays distributed, which preserves the case for "distributed prefill, local decode".
- If KV return and local resume work as intended, the target outcome is to keep about `600 t/s` prefill while improving generation to `30+ t/s` by decoding entirely on the Mac.

## Remaining unknowns

- Why `CUDA -> Metal` payloads restore the handoff logits exactly but still diverge after the first identical resumed decode step.
- Whether that resumed-decode drift lives in backend-specific KV interpretation, backend-specific decode-state evolution, or a smaller payload field that only matters after the next token is evaluated.
- How to let the coordinator participate in distributed prefill while still keeping a full local decode-capable engine resident without exhausting Metal memory budget.
- Whether the eventual handoff path should remain a merged `DSV4` checkpoint, move to an in-memory merged payload, or skip the merged payload entirely by streaming worker-owned KV shards back during prefill.

## Implementation direction

- Keep the merged `DSV4` handoff path as the current correctness boundary. Phase 2 validated that it can gather distributed shards, stage a payload, load locally, and recover the exact handoff checkpoint.
- Treat mixed-backend resumed-decode drift as the immediate blocker. Fix that before spending more time on transport redesign or pipelined KV return.
- Treat coordinator engine residency as a separate productization problem. Phase 2 proved the feature concept by releasing the distributed engine before local decode; the next residency work is about making that handoff practical in one user-visible workflow.
- Defer chunk-by-chunk KV return until after the resumed-decode correctness and residency story are both understood. At this point it is an optimization and UX improvement, not the first proof point.

## Relevant Code Module Map

### 1. Public session and distributed API boundary

- `ds4.h`
  - Defines the public `ds4_session` and `ds4_engine` API surface.
  - Distributed configuration lives in `ds4_distributed_options`.
  - The low-level distributed execution seam is already exposed via:
    - `ds4_session_eval_layer_slice()`
    - `ds4_session_eval_output_head_from_hc()`
    - `ds4_session_save_layer_payload()`
    - `ds4_session_load_layer_payload()`
  - The full-session payload ABI is defined here:
    - `DS4_SESSION_PAYLOAD_MAGIC` / `DS4_SESSION_PAYLOAD_VERSION`
    - `DS4_SESSION_LAYER_PAYLOAD_MAGIC` / `DS4_SESSION_LAYER_PAYLOAD_VERSION`
- Why it matters:
  - Any local-generation handoff should preserve these payload boundaries or extend them carefully.
  - This is the main header to consult when deciding whether the feature can be expressed as a normal session save/load versus a new distributed-only mechanism.

### 2. Core session implementation and KV payload format

- `ds4.c`
  - Owns normal session lifecycle, checkpoint timeline, prefill/decode execution, and serialized KV payloads.
  - Important functions for this effort:
    - `ds4_session_sync()` and `ds4_session_eval()`: normal prefill/decode entry points.
    - `ds4_session_save_payload()` / `ds4_session_load_payload()`: full topology-neutral session serialization.
    - `ds4_session_save_layer_payload()` / `ds4_session_load_layer_payload()`: per-layer-slice serialization used by distributed mode.
    - `ds4_session_payload_bytes()`: current payload sizing helper.
    - `ds4_session_save_snapshot()` / `ds4_session_load_snapshot()`: in-memory snapshot helpers.
    - `ds4_session_eval_layer_slice()`: executes a contiguous layer range and is the core primitive used by distributed inference.
  - Current constraint:
    - In-memory snapshots explicitly reject distributed sessions.
    - File payload save/load does support distributed sessions by delegating into `ds4_distributed.c`.
- Why it matters:
  - The local-decode handoff probably needs one of two things:
    - extend distributed sessions so they can produce an in-memory merged payload or snapshot, or
    - create a new handoff path that builds a local `ds4_session` from remote layer payloads without going through current snapshot helpers.

### 3. Distributed transport, routing, and shard orchestration

- `ds4_distributed.c`
  - This is the main implementation file for distributed inference.
  - It already states the intended model clearly: workers execute contiguous slices, save gathers worker-owned tensors into a normal DSV4 payload, and load splits a normal payload back across the route.
  - Important coordinator-side pieces:
    - `ds4_dist_session_create()`: sets up the coordinator session runtime.
    - `ds4_dist_session_route_ready()`: route completeness check.
    - `ds4_dist_session_sync()`: distributed prefill and prefix extension.
    - `ds4_dist_session_eval()`: distributed decode of one token.
    - `dist_coordinator_build_route_plan()` / `dist_coordinator_ensure_route()`: build the active worker chain.
    - `dist_coordinator_prefill_prompt_pipelined()`: pipelined prefill path.
    - `dist_coordinator_eval_span()`: coordinator-issued layer-slice work over one span.
    - `dist_coordinator_rebuild_from_transcript()`: replay/recovery path after route changes or transport failure.
  - Important save/load pieces:
    - `ds4_dist_session_save_payload()`: gathers local shard plus remote shards, then writes one normal session payload.
    - `ds4_dist_session_load_payload()`: reads one normal session payload and fans shards back out across the route.
    - `dist_kv_route_validate()` and the `dist_kv_*` helpers: verify shard layout compatibility and copy tensor sections.
  - Important worker-side pieces:
    - `dist_worker_handle_snapshot_save()`: serializes the worker-owned layer shard.
    - `dist_worker_handle_snapshot_load()`: restores the worker-owned layer shard.
    - `dist_send_snapshot_file_chunks()`: chunked wire transfer of snapshot payload data.
  - Important protocol pieces:
    - `DS4_DIST_MSG_SNAPSHOT_SAVE_REQ`
    - `DS4_DIST_MSG_SNAPSHOT_BEGIN`
    - `DS4_DIST_MSG_SNAPSHOT_CHUNK`
    - `DS4_DIST_MSG_SNAPSHOT_DONE`
    - `DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN`
- Why it matters:
  - This is the most likely place to implement "return KV during prefill" because it already owns:
    - per-worker KV shard save/load
    - chunked transport
    - route planning
    - pipelined prefill orchestration
  - The existing snapshot messages are request/response oriented around whole shard save/load, not incremental per-prefill-chunk KV return. That is the main architectural gap.

### 4. GPU backend abstraction used by both Metal and CUDA

- `ds4_gpu.h`
  - Defines the tensor-resident GPU API shared by the backend implementations.
  - Relevant categories:
    - command lifetime: `ds4_gpu_begin_commands()`, `ds4_gpu_end_commands()`, `ds4_gpu_synchronize()`
    - tensor I/O: `ds4_gpu_tensor_write()`, `ds4_gpu_tensor_read()`, `ds4_gpu_tensor_copy()`
    - prefill/decode attention and compressor kernels
  - Why it matters:
    - Cross-backend compatibility depends on whether the serialized layer payloads truly represent backend-neutral tensor state, not backend-private handles or layouts.
    - If local handoff fails only on one backend pair, the root cause will likely be below the distributed layer but above raw model logic.

### 5. Backend-specific graph/runtime implementations

- `ds4_metal.m`
  - Primary Metal runtime and graph executor.
  - Allocates the live GPU graph state, raw KV ring, compressed KV buffers, and scratch tensors.
  - Implements the actual prefill/decode graph encoding used by `ds4_session_eval_layer_slice()` and normal local inference.
- `ds4_cuda.cu`
  - CUDA backend implementation of the same tensor/runtime surface.
- Why they matter:
  - The issue’s key unknown is cross-backend cache portability, specifically whether CUDA-produced layer payloads can be loaded and resumed by Metal.
  - If the serialized layer payload format is correct but behavior still diverges, the bug will likely be in backend read/write conventions, tensor dtype conversions, or cache-state assumptions here.

### 6. Disk KV cache and staged payload persistence

- `ds4_kvstore.c` and `ds4_kvstore.h`
  - Manage on-disk KV cache files used by the CLI/server path.
  - `ds4_session_stage_payload()` and `ds4_session_write_staged_payload()` are used when persisting session payloads.
  - Load uses the same session payload ABI through `ds4_session_load_payload()`.
- Why it matters:
  - If the handoff is first implemented as "distributed prefill -> merged payload file -> local session load", these files are directly relevant.
  - They are less central if the goal is a zero-copy or in-memory chunked transfer path, but still define the durable payload contract.

### 7. CLI/server entry points and user-visible workflow

- `ds4_cli.c`
  - Normal CLI path that drives local sessions and can expose coordinator/worker mode options.
- `ds4_server.c`
  - Server runtime and persistent session/KV behavior.
  - Distributed mode is wired through the normal engine/session path rather than a separate application stack.
- `README.md`
  - Documents current distributed assumptions:
    - each worker owns its KV slice
    - prefill is pipelined
    - generation remains slower when distributed
    - save/load already uses the same normal KV file format by gathering and splitting shards
- Why they matter:
  - After implementation, these are the places that will need the workflow and operator guidance updated.
  - The README also documents the current conceptual model and is useful for checking whether the code still matches the intended design.

### 8. Existing validation surface

- `tests/ds4_test.c`
  - Broad regression coverage for session sync/eval, logits, and long-prefill behavior.
- `tests/cuda_long_context_smoke.c`
  - CUDA-side long-context kernel regression coverage.
- Why they matter:
  - There does not appear to be dedicated coverage yet for:
    - distributed shard portability across backends
    - distributed prefill followed by local-only decode resume
    - incremental remote KV return during pipelined prefill
  - New tests for this feature likely belong in `tests/ds4_test.c`, with at least one focused serialization/resume test before attempting a full distributed integration test.

## Most Relevant Files To Start In

- `ds4_metal.m` and `ds4_cuda.cu`: Phase 2 points first at backend-specific resumed decode behavior, so these are now the highest-priority files.
- `ds4.c`: inspect payload restore, decode entry, and session state setup around `ds4_session_load_payload()` and resumed `ds4_session_eval()`.
- `tests/issue304_phase1_matrix.c` and `tests/issue304_phase2_handoff.c`: keep using these focused harnesses to narrow the first post-load divergence.
- `ds4_distributed.c`: return here after backend resume correctness is understood, or when implementing the later residency/transport optimization phases.

## Research And Implementation Plan

This change should be run as an iterative research project, not as a straight feature implementation. The priority is to compound learning: each step should produce an artifact that either validates the next step, rejects an assumption, or narrows the design space.

### Operating principles

- Prefer the existing `DSV4` full-session payload as the first correctness boundary. The code already knows how to gather distributed shards into a topology-neutral session stream.
- Treat chunk-by-chunk KV return as an optimization, not the first milestone. Prove that "distributed prefill -> merged payload -> full local session load -> local decode" is correct before changing the prefill pipeline.
- Keep implementation changes small until a validation artifact says the next abstraction is justified.
- Capture every surprising result in this plan or a linked artifact before moving on. This includes failures, mismatched logits, transport timings, and rejected API options.
- Measure correctness before throughput. A fast handoff is not useful until the resumed local session is demonstrably equivalent enough to continue generation.
- Treat model weight residency as a separate correctness and feasibility problem. Phase 2 showed that releasing the distributed engine before local decode is enough to prove the feature concept, but not enough for a clean end-user workflow.

### Required research artifacts

- `artifacts/issue-304/compatibility-matrix.md`
  - Records `Metal -> Metal`, `CUDA -> CUDA`, `CUDA -> Metal`, and `Metal -> CUDA` payload load/resume outcomes.
  - Include commit SHA, model filename/hash, backend, context size, prefill chunk size, layer split, token count, payload byte count, and pass/fail notes.
- `artifacts/issue-304/logit-comparisons.md`
  - Captures next-token comparison between single-node prefill, distributed prefill + distributed decode, and distributed prefill + local decode after payload load.
  - Include top1 match, top5/top20 overlap, max absolute drift, RMS drift, and first 16 greedy tokens.
- `artifacts/issue-304/shard-metadata.md`
  - Captures route coverage and KV shape metadata: `layer_start`, `layer_end`, `token_count`, `token_hash`, `raw_live`, `raw_window`, `raw_cap`, `comp_cap`, per-layer `n_comp`, and per-layer `n_index_comp`.
- `artifacts/issue-304/perf-breakdown.md`
  - Captures prefill t/s, remote shard save time, KV bytes returned, merge/write time, local payload load time, and local decode t/s.
- `artifacts/issue-304/decision-log.md`
  - Records decisions with date, evidence, alternatives rejected, and follow-up questions.
- `artifacts/issue-304/runbook.md`
  - Records exact commands, host roles, environment variables, model paths, ports, and reproduction notes for every validation run.
- `artifacts/issue-304/failure-cases.md`
  - Records failures that should remain rejected, including stale worker state, bad token hash, layer range mismatch, context mismatch, and unsupported backend combinations.
- `artifacts/issue-304/engine-residency.md`
  - Records how model layer residency currently works, memory implications of each candidate design, and the chosen strategy for letting a local decode-capable engine participate in distributed prefill.
- `artifacts/issue-304/topology-decoupling.md`
  - Records follow-on design notes for decoupling coordinator/worker roles, prefill pipeline layer ownership, output-head/logit production, and local decode placement.

### Artifact conventions

- Create or update artifacts before moving to the next phase. If a phase is blocked, write down the blocker and the smallest next experiment.
- Include enough run context to reproduce results: commit SHA, branch, host names, OS, backend, model path/hash, layer split, context size, prefill chunk, activation bits, command line, and relevant environment variables.
- Prefer tables for measured data and short dated entries for decisions.
- If a code change adds a new flag, mode, or test, document both the intended command and one known-good invocation in `artifacts/issue-304/runbook.md`.
- If a validation result changes the implementation direction, update `artifacts/issue-304/decision-log.md` and the relevant phase notes in this plan.

### Phase 0: Baseline and instrumentation

Goal:

- Why: establish a factual baseline before changing behavior, so later regressions, speedups, memory costs, and correctness differences are attributable instead of guessed.
- What: produce reproducible commands, route/payload metadata, and timing measurements for current distributed prefill, distributed decode, and distributed payload save behavior.

Expected artifacts:

- Update `artifacts/issue-304/runbook.md` with exact baseline commands for the issue topology and one smaller reproducible topology.
- Update `artifacts/issue-304/shard-metadata.md` with the current distributed route, layer ownership, token counts, token hash, and payload metadata.
- Update `artifacts/issue-304/perf-breakdown.md` with current distributed prefill t/s, distributed decode t/s, and distributed payload save timing.
- Update `artifacts/issue-304/decision-log.md` only if the current save/stage behavior disproves the assumed `DSV4` first path.
- Update `artifacts/issue-304/research-notes.md` with the current pre-implementation findings, surviving assumptions, and any invalidated hypotheses.

Code touchpoints:

- `ds4_distributed.c`
  - `ds4_dist_session_sync()`: distributed prefill entry.
  - `dist_coordinator_prefill_prompt_pipelined()`: pipelined prefill behavior and completion point.
  - `dist_coordinator_eval_span()`: fallback prefill and current distributed decode path.
  - `ds4_dist_session_eval()`: one-token distributed decode.
  - `ds4_dist_session_save_payload()`: distributed gather into a normal `DSV4` payload.
  - `dist_kv_route_validate()`: shard coverage/layout validation before save/load.
  - `dist_save_remote_shard_to_file()`: coordinator-side remote shard request.
  - `dist_worker_handle_snapshot_save()`: worker-side shard export.
- `ds4.c`
  - `ds4_session_sync()`: public prefill entry and distributed delegation.
  - `ds4_session_eval_internal()`: public decode path and distributed delegation.
  - `ds4_session_stage_payload()`: tempfile-backed payload staging.
  - `ds4_session_save_payload()`: distributed save delegation.
- `ds4.h`
  - `ds4_distributed_options`: route/prefill/activation settings to record.
  - `ds4_session_payload_file`: staged payload handle to inspect.

Other entry points:

- CLI coordinator/worker commands from `README.md`.
- Existing issue topology:
  - worker: `./ds4 --role worker --layers 22:output --coordinator 10.77.0.1 1234`
  - coordinator: `./ds4 --role coordinator --layers 0:21 --listen 10.77.0.1 1234 --prompt-file README.md --nothink -n 100`
- Local or loopback topology for faster iteration, if supported by available hardware.
- `make test` only as a sanity check; it does not validate the distributed handoff by itself.

Work items:

- Capture the current distributed prefill/decode route for the issue topology and at least one smaller local topology.
- Add or script metadata dumps for route layer coverage, payload header fields, per-shard layer ranges, token count/hash, and raw/compressed row counts.
- Confirm whether `ds4_session_stage_payload()` already works on a distributed coordinator immediately after pipelined prefill.
- Record exact timings for distributed prefill, current distributed decode, and current distributed payload save.

Exit gate:

- The current system can produce a merged `DSV4` payload from a distributed coordinator after prefill, or the failure mode is captured with a concrete call stack/error.

### Phase 1: Prove whole-payload resume locally

Goal:

- Why: avoid inventing a distributed handoff mechanism before proving the existing durable session format can faithfully suspend and resume local graph state.
- What: demonstrate that a normal non-distributed session can save `DSV4`, load it into a fresh non-distributed session, and continue with measured logit/token equivalence across important cache boundary cases.

Expected artifacts:

- Update `artifacts/issue-304/compatibility-matrix.md` with local same-backend payload resume results, starting with `Metal -> Metal` on Mac and `CUDA -> CUDA` where CUDA is available.
- Update `artifacts/issue-304/logit-comparisons.md` with local save/load top-k, drift, and short greedy continuation results.
- Update `artifacts/issue-304/shard-metadata.md` with `DSV4` header fields and `DSVL` representative shard metadata.
- Update `artifacts/issue-304/runbook.md` with test commands and model/context settings.
- Update `artifacts/issue-304/research-notes.md` with Phase 1 findings, what the results rule out, and the remaining unknowns carried into Phase 2.

Code touchpoints:

- `tests/ds4_test.c`
  - Add or extend focused save/load resume tests.
  - Add a representative per-layer `DSVL` save/load smoke if the existing test harness can create a graph session with the required model.
- `ds4.c`
  - `ds4_session_save_payload()` / `ds4_session_load_payload()`: full `DSV4` save/load.
  - `ds4_session_save_layer_payload()` / `ds4_session_load_layer_payload()`: per-layer `DSVL` save/load.
  - `payload_write_tensor_span()` / `payload_read_tensor_span()`: tensor serialization boundary if drift or corruption appears.
  - `ds4_session_payload_bytes()`: payload sizing behavior; note distributed currently returns `0`.
  - `ds4_session_save_snapshot()` / `ds4_session_load_snapshot()`: memory snapshot helpers, but distributed sessions are currently rejected.
- `ds4_gpu.h`
  - `ds4_gpu_tensor_read()` / `ds4_gpu_tensor_write()` / `ds4_gpu_synchronize()`: backend-neutral tensor byte transfer boundary.
- `ds4_metal.m` and `ds4_cuda.cu`
  - Only inspect if payload bytes load but logits diverge unexpectedly.

Other entry points:

- `make test`
- `./ds4_test --metal-short-prefill`
- `./ds4_test --metal-tensor-equivalence`
- `./ds4_test --local-golden-vectors`
- `make cuda-regression` and `tests/cuda_long_context_smoke.c` for CUDA-side validation where available.

Work items:

- Add a focused local save/load resume test: prefill prompt in one local graph session, save `DSV4`, load into a fresh non-distributed graph session, then compare logits and a short greedy continuation.
- Add a per-layer `DSVL` save/load smoke for representative ranges: first layers, middle compressed layers, ratio-4 compressed layers, and final/output-adjacent layers.
- Run boundary prompts around likely cache transition points: `1`, `3`, `4`, `5`, `4095`, `4096`, `4097`, `raw_window - 1`, `raw_window`, and `raw_window + 1`.

Exit gate:

- A non-distributed local session can load a normal `DSV4` payload and continue with acceptable logit/token equivalence.
- Any numeric drift is measured and classified as expected backend precision drift or a correctness bug.

### Phase 2: Prove distributed prefill to local decode via existing payload path

Goal:

- Why: get the fastest end-to-end correctness signal for issue #304 by reusing the existing distributed shard gather and normal `DSV4` load path.
- What: prove or disprove that distributed prefill state can be saved as one merged `DSV4`, loaded into a full local non-distributed session, and decoded locally with measured equivalence, while recording whether the two-session/two-engine shape is only a proof harness.

Expected artifacts:

- Update `artifacts/issue-304/compatibility-matrix.md` with distributed prefill to local decode results for every tested backend pair.
- Update `artifacts/issue-304/logit-comparisons.md` with three-way comparisons: single-node local, distributed prefill + distributed decode, and distributed prefill + local decode after payload load.
- Update `artifacts/issue-304/perf-breakdown.md` with distributed prefill time, distributed save/stage time, payload bytes, local load time, and local decode t/s.
- Update `artifacts/issue-304/shard-metadata.md` with route coverage and all shard metadata from the handoff run.
- Update `artifacts/issue-304/failure-cases.md` for any mismatch that correctly rejects a handoff.
- Update `artifacts/issue-304/engine-residency.md` with peak memory observed for the proof path and whether running two engines is feasible only as a test harness.
- Update `artifacts/issue-304/research-notes.md` with the Phase 2 outcome, the current best explanation for any mismatch, and whether the payload-first path remains viable.

Code touchpoints:

- `ds4_distributed.c`
  - `ds4_dist_session_save_payload()`: main distributed gather path to validate.
  - `dist_kv_parse_layer_payload()`: shard header parsing.
  - `dist_kv_write_session_header()`: merged `DSV4` header creation.
  - `dist_prepare_shard_from_session_payload()`: useful reference for the reverse split.
  - `dist_worker_handle_snapshot_save()`: worker export and token hash validation.
  - `dist_send_snapshot_file_chunks()`: existing chunk transfer mechanism.
- `ds4.c`
  - `ds4_session_stage_payload()` and `ds4_session_write_staged_payload()`: tempfile-backed export path.
  - `ds4_session_load_payload()`: local full-session import.
  - `ds4_session_create()`: determine whether a separate full local session is needed.
  - `ds4_engine_open()`: current place where distributed `--layers` constrains model map spans.
  - `weights_model_map_spans()`: computes tensor spans for a layer slice.
  - `weights_layers_bound()`: verifies whether a requested layer range has loaded weights.
- `ds4_cli.c`
  - Candidate place for a temporary/manual CLI experiment if the current CLI cannot exercise save/load handoff directly.
- `ds4_kvstore.c`
  - Reference for staged payload persistence and load path if the prototype uses a KV cache file rather than a raw temp payload.

Other entry points:

- Manual coordinator/worker CLI commands.
- A temporary harness or test binary if the CLI cannot perform the handoff without invasive user-facing changes.
- Existing `ds4_session_stage_payload()` temp files under `/tmp/ds4-session-payload.XXXXXX`.
- Existing KV store path if testing through server/session persistence is easier.
- Memory reporting from OS tools and backend diagnostics to capture peak resident memory/accelerator VM pressure.

Work items:

- Run distributed prefill normally.
- Immediately stage/save the distributed coordinator session payload using the existing `ds4_session_save_payload()` or `ds4_session_stage_payload()` path.
- Create a separate full local, non-distributed graph session.
- Load the merged `DSV4` payload into that local session.
- Decode locally for at least 1-3 tokens.
- Compare against continuing distributed decode from the original distributed session and single-node local prefill/decode when feasible.
- Treat this two-session shape as a correctness experiment unless `artifacts/issue-304/engine-residency.md` proves memory is acceptable.

Key questions:

- Does the coordinator have enough local model state loaded to create the full local session, or is a second full engine/session required?
- Does `--layers` imply partial model loading that conflicts with local decode?
- Does `ds4_dist_session_save_payload()` safely run immediately after `dist_coordinator_prefill_prompt_pipelined()` completes?
- Are worker session IDs and token hashes stable between prefill completion and shard fetch?

Exit gate:

- The existing payload path can hand off distributed prefill state into a local-only decode session, or a specific blocker is captured with enough detail to choose the next implementation step.
- If the path works only by running two engines or reloading an engine, it must not proceed directly to user-facing implementation; Phase 3 must choose a viable layer-residency design first.

### Phase 3: Mixed-backend resumed decode root cause

Goal:

- Why: Phase 2 proved the merged handoff checkpoint is exact, but it also reproduced mixed-backend resumed-decode drift immediately after the first identical forced token. The next phase needs to turn that broad symptom into a concrete backend or restore-path bug.
- What: isolate why `CUDA -> Metal` diverges after the first resumed eval step even when the restored logits are identical and the sampled greedy path can stay stable.

Expected artifacts:

- Update `artifacts/issue-304/logit-comparisons.md` with finer-grained forced-token traces around the first divergent resumed step.
- Update `artifacts/issue-304/failure-cases.md` with every narrowed reproduction of post-load decode drift that remains rejected.
- Update `artifacts/issue-304/research-notes.md` with the first divergent state that can be localized reliably.
- Update `artifacts/issue-304/decision-log.md` if the evidence shows the bug is payload-ABI, backend-runtime, or decode-graph specific.

Code touchpoints:

- `tests/issue304_phase1_matrix.c`
  - Extend the focused resume helper so it can compare finer-grained post-load state, not just logits and greedy tokens.
- `tests/issue304_phase2_handoff.c`
  - Keep the distributed-prefill harness aligned with the smaller Phase 1 probes so the same divergence can be reproduced in both contexts.
- `ds4.c`
  - `ds4_session_load_payload()`: restore path to inspect for any field that is sufficient for step-0 logits but insufficient for step-1 decode evolution.
  - `ds4_session_eval()` and local decode entry path: first resumed decode step after load.
- `ds4_gpu.h`
  - Shared tensor read/write and synchronization boundary if the first divergent state turns out to be visible only below the session payload layer.
- `ds4_metal.m` and `ds4_cuda.cu`
  - Backend-specific decode-state assumptions, dtype conversion, raw/compressed KV interpretation, and graph state setup.

Work items:

- Extend `tests/issue304_phase1_matrix.c` so it can dump or compare finer-grained post-load state around the first resumed decode step.
- Compare `CUDA -> Metal` and `Metal -> Metal` immediately after one identical forced eval token, looking for the first tensor or cache-state difference that appears before logits drift becomes visible.
- Inspect `ds4_session_load_payload()` and resumed `ds4_session_eval()` in `ds4.c` for any state that is restored correctly enough for step-0 logits but not for step-1 decode evolution.
- Inspect `ds4_metal.m` and `ds4_cuda.cu` for backend-specific assumptions around raw KV rows, compressed KV rows, cache frontiers, dtype conversion, or decode-only graph state that may not be fully represented by the current payload ABI.

Exit gate:

- Either:
  - `CUDA -> Metal` forced-token drift is eliminated for the existing Phase 1 and Phase 2 probes, or
  - the first divergent restored state is localized well enough that the remaining work is a specific backend/runtime bug rather than a general handoff uncertainty.

### Phase 4: Coordinator residency and workflow viability

Goal:

- Why: Phase 2 currently proves the concept by releasing the distributed engine and then opening a full local engine. That is enough for research, but not enough for a practical one-shot coordinator workflow.
- What: choose a viable residency design that lets the coordinator do distributed prefill and then local decode without requiring a research-only two-engine transition.

Expected artifacts:

- Update `artifacts/issue-304/engine-residency.md` with the current layer-loading model, observed memory behavior, and candidate designs.
- Update `artifacts/issue-304/perf-breakdown.md` with any measured cost of wider model mapping, lazy layer activation, or switching between slice/full execution modes.
- Update `artifacts/issue-304/decision-log.md` with the chosen residency design and rejected alternatives.
- Update `artifacts/issue-304/failure-cases.md` with expected failures for missing full-decode layers, invalid layer range transitions, and memory-budget rejection.
- Update `artifacts/issue-304/runbook.md` with commands and environment settings used to measure memory.
- Update `artifacts/issue-304/research-notes.md` with the residency conclusions and the decision impact on later implementation phases.

Code touchpoints:

- `ds4.h`
  - `ds4_engine_options.load_layer_start` / `load_layer_end`: explicit model slice loading knobs.
  - `ds4_distributed_options.layers`: distributed route ownership, currently also influences local loaded layer range.
  - Potential future option fields if route ownership and local weight residency need to be decoupled.
- `ds4.c`
  - `ds4_engine_open()`: currently maps distributed `--layers` into `load_layer_start`, `load_layer_end`, `load_output`, and `load_output_optional`.
  - `weights_model_map_spans()`: computes model map spans for a sliced layer range.
  - `ds4_gpu_set_model_map_spans()` call site: constrains the backend-visible model mapping.
  - `ds4_gpu_set_model_map_range()` call site: maps the full model tensor data range.
  - `ds4_engine_preload_pro_q4_expert_tables()`: layer-sliced preload behavior that may need to match the selected residency strategy.
  - `accelerator_cache_model_tensors()`: optional model tensor cache behavior and memory pressure point.
  - `weights_layers_bound()`: runtime check that requested layer execution has weights.
  - `ds4_session_create()`: session graph allocation is based on the engine's loaded/bound weights.
  - `ds4_session_eval_layer_slice()`: distributed prefill slice execution must remain able to run only the route-owned layers even if more weights are resident.
  - `ds4_session_eval_internal()` and local graph decode path: local decode needs all transformer layers plus output head.
- `ds4_distributed.c`
  - `dist_coordinator_build_route_plan()`: route ownership should stay independent from local weight residency if the final design decouples them.
  - `dist_coordinator_prefill_prompt_pipelined()` and `dist_coordinator_eval_span()`: must continue issuing only the coordinator-owned distributed slice during prefill.
- `ds4_cli.c` and `README.md`
  - User-visible meaning of `--layers` if route ownership and residency are decoupled.
  - Current distributed docs state that `--layers` controls which tensors are mapped; update that assumption only if the chosen design actually decouples route ownership from residency.
- `ds4_gpu.h`, `ds4_metal.m`, and `ds4_cuda.cu`
  - Revisit these if the residency design requires dynamic mapping, lazy mapping, or remapping backend tensor views.

Candidate designs to analyze:

- Decouple route ownership from weight residency:
  - Keep `--layers` as the distributed route slice, but allow the coordinator to map full local decode weights at startup.
  - During distributed prefill, call `ds4_session_eval_layer_slice()` only for the coordinator route slice even though more weights are resident.
- Full-resident coordinator with sliced distributed execution:
  - Load the full model once on the coordinator, but still advertise/execute only the coordinator-owned distributed layer range for prefill.
- Lazy or expandable residency:
  - Start with route-slice mapping, then expand to full mapping before local decode without closing the engine.
  - Only keep this path if backend model map spans and cached/preloaded tensors can be expanded safely and fast enough.
- Dual session over one engine:
  - Use one full-resident engine with separate distributed-prefill and local-decode sessions.
  - This may avoid duplicate weights but still duplicates KV/session graph memory; measure before selecting.
- Two engines or reload:
  - Keep only as a correctness harness or fallback diagnostic, not the final workflow unless the measurements force it.

Work items:

- Trace exactly how distributed `--layers` becomes loaded model spans today.
- Measure memory for route-slice engine, full engine, dual-session-over-one-engine, and the current two-engine proof harness where feasible.
- Verify whether a full-resident coordinator can still execute only its distributed slice during prefill by using the existing layer-slice APIs.
- Determine whether session graph allocation assumes only a subset of layers is bound, or whether it can safely hold KV for all layers when the engine has full weights.
- Determine whether output head residency is required on the coordinator for local decode and how `load_output_optional` interacts with that.
- Decide whether the implementation should decouple "route layer range" from "loaded layer range" in `ds4_engine_options` / `ds4_distributed_options`.

Exit gate:

- A final implementation path exists that does not require loading/unloading the whole engine at handoff time and does not require two full model-weight copies in memory.
- The chosen strategy is captured in `artifacts/issue-304/engine-residency.md` and `artifacts/issue-304/decision-log.md`.
- If no viable strategy is found, stop before advancing and record the smallest code experiment needed to test dynamic or expanded residency.

### Phase 5: Choose API shape and implement the handoff path

Goal:

- Why: once resumed-decode correctness and residency constraints are understood, the next step is turning the research-only path into a real session/CLI/server workflow.
- What: choose the smallest API shape that matches the validated behavior, implement it, and verify that distributed prefill can transition to local-only decode without manual harness glue.

Expected artifacts:

- Update `artifacts/issue-304/decision-log.md` with the selected API shape, alternatives rejected, and evidence from Phases 2-4.
- Update `artifacts/issue-304/runbook.md` with the expected developer/operator workflow for the selected path.
- Update `artifacts/issue-304/logit-comparisons.md` and `artifacts/issue-304/perf-breakdown.md` with post-implementation correctness and timing.
- Update `artifacts/issue-304/failure-cases.md` with user-visible negative cases and observed diagnostics.
- Update `artifacts/issue-304/research-notes.md` with implementation-phase findings and deltas from the earlier hypotheses.
- Update `README.md` only once the workflow is stable enough to describe honestly.

Code touchpoints:

- `ds4.h`
  - New helper declaration or option fields if the feature is exposed beyond the research harness.
- `ds4.c`
  - Core handoff flow and local session creation/load boundaries.
- `ds4_distributed.h`
  - Distributed-specific declarations if the chosen API remains internal to coordinator sessions.
- `ds4_distributed.c`
  - Coordinator gather/handoff path and failure diagnostics for stale/missing shards.
- `ds4_cli.c` and `ds4_server.c`
  - User-visible wiring only after the core session path works.
- `ds4_kvstore.c`
  - Only if the selected path routes through persisted/staged KV payloads.
- `tests/ds4_test.c`
  - Regression coverage for local handoff behavior where feasible.

Likely options:

- Whole-payload handoff:
  - expose a helper that saves/stages a distributed session payload and loads it into a provided local session.
- In-memory payload handoff:
  - extend distributed sessions so `ds4_session_save_snapshot()` or a new memory-backed payload helper can gather remote shards.
  - This avoids temp files, but requires solving the current distributed snapshot restriction.
- Explicit shard merge:
  - expose a multi-`DSVL` merge path that reconstructs a local session from layer shards.
  - This is more invasive and should only be chosen if whole `DSV4` handoff proves too slow or too rigid.

Decision criteria:

- Correctness equivalence after resume.
- Compatibility with the chosen residency strategy.
- Operator complexity in CLI/server workflows.
- Ability to support cross-backend handoff.
- Measured handoff overhead relative to prefill speedup.
- Amount of new protocol surface in `ds4_distributed.c`.

Work items:

- Decide whether the first implementation should be whole-payload, in-memory payload, or explicit shard merge.
- Record that decision and evidence before adding user-facing wiring.
- Add the minimal session-level handoff mechanism selected from that decision.
- Keep the distributed prefill session and local decode session distinct unless evidence shows same-session mutation is safer.
- Ensure the local session receives full token timeline, logits, raw SWA KV rows in logical order, compressed KV rows, compressor frontier state, and indexer state.
- Add CLI/server wiring only after the core session path works.
- Document the operator requirement for full local decode weights if the coordinator's distributed `--layers` setting does not load them.

Exit gate:

- A user-visible path can run distributed prefill and then local-only decode with measured speedup and correctness artifacts.

### Phase 6: Optimize with pipelined KV return

Goal:

- Why: once correctness is established, the remaining risk is that post-prefill shard fetch/merge/load latency erases the practical benefit of faster distributed prefill.
- What: measure handoff overhead, then add incremental or pipelined KV return only if it materially reduces latency while preserving token/hash ordering, backend synchronization, and resume equivalence.

Expected artifacts:

- Update `artifacts/issue-304/perf-breakdown.md` with before/after handoff latency and transfer byte counts.
- Update `artifacts/issue-304/logit-comparisons.md` with correctness results for incremental KV return.
- Update `artifacts/issue-304/shard-metadata.md` with per-chunk or per-window KV state metadata if chunked return is implemented.
- Update `artifacts/issue-304/decision-log.md` with protocol decisions and why existing snapshot framing was or was not sufficient.
- Update `artifacts/issue-304/failure-cases.md` with in-flight/race/stale chunk rejection cases.
- Update `artifacts/issue-304/research-notes.md` with the optimization-phase findings and whether chunked return is justified versus whole-payload handoff.

Code touchpoints:

- `ds4_distributed.c`
  - `dist_coordinator_prefill_prompt_pipelined()`: main place to coordinate pipelined prefill and KV return.
  - `dist_prefill_sender_main()`: sends queued hidden states to workers.
  - `dist_prefill_result_reader_main()`: receives chunk ACKs/final result; likely place to coordinate returned KV state.
  - `dist_worker_process_work_payload()`: worker receives/evaluates prefill work.
  - `dist_worker_handle_snapshot_save()`: existing shard export reference.
  - `dist_send_snapshot_file_chunks()`: existing chunked transfer reference.
  - `dist_write_frame_header()` / `dist_read_frame_header()`: protocol framing if new messages are required.
  - `DS4_DIST_MSG_SNAPSHOT_*`: existing protocol message family to reuse or extend.
- `ds4.c`
  - `ds4_session_save_layer_payload()`: current whole-shard export; may need range/window-aware variant if incremental return is not expressible through existing payloads.
  - `ds4_session_load_layer_payload()`: current layer-shard restore.
  - Graph synchronization points before tensor reads.
- `ds4_gpu.h`
  - `ds4_gpu_synchronize()` and tensor read APIs if chunk export must avoid racing kernels.
- `ds4_metal.m` and `ds4_cuda.cu`
  - Only if incremental reads need backend-specific synchronization or buffer layout fixes.

Other entry points:

- Distributed coordinator/worker wire protocol.
- Network timing measurements on 2.5 GbE, 5 GbE, and any Thunderbolt/direct-link setup available.
- Existing `--dist-activation-bits`, `--dist-prefill-chunk`, and `--dist-prefill-window` flags.

Work items:

- Extend `ds4_distributed.c` only after Phase 5 shows payload save/load overhead is material.
- Investigate returning worker-owned KV alongside prefill chunk boundaries.
- Define when a chunk's KV is safe to export relative to in-flight kernels and worker forwarding.
- Preserve ratio-4 compressor/indexer frontier state; do not return only completed compressed rows.
- Reuse existing snapshot chunk framing where possible: `DS4_DIST_MSG_SNAPSHOT_SAVE_REQ`, `DS4_DIST_MSG_SNAPSHOT_BEGIN`, `DS4_DIST_MSG_SNAPSHOT_CHUNK`, and `DS4_DIST_MSG_SNAPSHOT_DONE`.
- Add new protocol messages only if existing request/response snapshot framing cannot express incremental return cleanly.

Exit gate:

- Incremental KV return reduces measured handoff latency without changing resumed logits/tokens beyond the accepted drift envelope.

### Phase 7: Failure handling and recovery

Goal:

- Why: a local decode handoff turns distributed worker KV into trusted local generation state, so stale, incomplete, or mismatched shards must fail closed.
- What: validate and harden rejection paths for token/hash, model, route, layer range, context, output-head, reconnect, and shard metadata mismatches, with reproducible negative artifacts.

Expected artifacts:

- Update `artifacts/issue-304/failure-cases.md` with each negative case, expected rejection, observed error text, and reproduction command.
- Update `artifacts/issue-304/runbook.md` with fault-injection or manual failure procedures.
- Update `artifacts/issue-304/decision-log.md` if a failure mode forces an API or protocol change.
- Update `artifacts/issue-304/research-notes.md` with any failure findings that materially change the implementation direction.

Code touchpoints:

- `ds4_distributed.c`
  - `dist_kv_route_validate()`: route/layer coverage validation.
  - `dist_worker_handle_snapshot_save()`: token hash and worker session validation.
  - `dist_worker_handle_snapshot_load()`: load rejection behavior.
  - `dist_coordinator_rebuild_from_transcript()`: route rebuild behavior before/after handoff.
  - `dist_coordinator_ensure_route()` and route planning helpers.
  - Error propagation through coordinator save/load and eval paths.
- `ds4.c`
  - `ds4_session_load_payload()`: normal payload rejection behavior.
  - `ds4_session_load_layer_payload()`: shard metadata validation.
- `tests/ds4_test.c`
  - Add focused negative tests where they can run without full distributed hardware.

Other entry points:

- Worker restart/reconnect workflow.
- Mismatched coordinator/worker layer split.
- Mismatched context size or model id.
- Token transcript mutation between prefill completion and shard fetch.

Work items:

- Reject shard fetches when token count/hash, model id, layer range, context, raw window, or compression capacity do not match.
- Test worker reconnect or route rebuild before shard fetch.
- Test final-worker/output-head ownership cases.
- Capture failure artifacts: exact load error, shard header dump, route plan, and first mismatching layer/range.

Exit gate:

- Known stale/missing/mismatched shard cases fail closed with useful diagnostics.

### Phase 8: Topology decoupling follow-on

Goal:

- Why: practical deployments may want GPU-rich machines to do prefill and memory-rich machines to do generation, so control-plane role, prefill layer ownership, output-head/logit ownership, KV return destination, and decode owner should not remain permanently coupled.
- What: document the current coordinator-first topology constraints, determine whether topology decoupling is required for the first production implementation or can be deferred, and outline a route/protocol direction if it must be solved.

This is intentionally after Phase 7 because the immediate feature can be proven with the current route model. Do not let this phase block the first correct local-generation handoff unless the chosen Phase 3 residency design makes topology decoupling unavoidable.

Expected artifacts:

- Update `artifacts/issue-304/topology-decoupling.md` with the current topology constraints, desired future topologies, and candidate protocol/API changes.
- Update `artifacts/issue-304/decision-log.md` with whether topology decoupling is deferred, partially required, or required for the first production implementation.
- Update `artifacts/issue-304/perf-breakdown.md` with measurements that motivate topology changes, especially GPU-prefill throughput versus memory-bandwidth-local decode throughput.
- Update `artifacts/issue-304/runbook.md` with any topology experiments and command lines.
- Update `artifacts/issue-304/research-notes.md` with topology-related findings and whether they affect the first implementation scope.

Current constraints to capture:

- The coordinator route must currently start at layer `0`.
- The coordinator runs the early layer slice during distributed prefill.
- Workers cover later contiguous slices.
- `--layers N:output` makes the final worker own the output head and return logits.
- `--layers N:42` keeps the output head/logit production on the coordinator, but the worker still owns the later transformer slice up to layer 42.
- Coordinator/worker roles currently imply control-plane ownership, prompt/session ownership, route construction, and a position in the layer pipeline.

Desired future topology properties:

- Control-plane coordinator should be separable from prefill execution role.
- The machine that performs local decode should be separable from the machine that runs the earliest prefill layers.
- Prefill pipeline stages should be assignable to the machines with the best GPU throughput, not necessarily to the decode/sampling machine.
- Decode/sampling should be placeable on the machine with the best memory bandwidth and enough resident full-model/KV state.
- Output-head/logit production should be an explicit route property rather than an implicit consequence of `N:output` versus `N:42`.
- KV ownership and KV return destination should be explicit enough to support "GPU prefill workers -> memory-rich local decoder" without forcing awkward layer ownership.

Code touchpoints:

- `ds4_distributed.c`
  - `dist_coordinator_build_route_plan()`: currently enforces route coverage and coordinator-first execution.
  - `dist_coordinator_ensure_route()`: route readiness and rebuild entry point.
  - `dist_coordinator_prefill_prompt_pipelined()`: assumes coordinator local slice starts the pipeline.
  - `dist_coordinator_eval_span()`: current span execution over the route.
  - `dist_coordinator_send_remote_work_on_fd()`: work dispatch format and first remote hop.
  - `dist_prefill_sender_main()` / `dist_prefill_result_reader_main()`: current pipelined prefill sender/result coupling.
  - `dist_worker_process_work_payload()`: worker route validation, forwarding, output-logits handling, and final-result behavior.
  - `dist_route_validate_blob()`, `dist_route_get_entry()`, and return-target helpers: current route representation and result destination constraints.
  - Worker `HELLO` registration structs and parsing: advertised layer range and output-head ownership.
- `ds4_distributed.h`
  - Coordinator/worker API boundary if roles become more specific than control-plane coordinator and execution worker.
- `ds4.h`
  - `ds4_distributed_options.layers`: currently combines route layer ownership and local model loading concerns.
  - Future option shapes if route ownership, output-head ownership, KV return destination, and decode destination are split.
- `ds4_cli.c`
  - CLI flag parsing and help text for future topology descriptions.
- `README.md`
  - Distributed docs currently describe `A -> B -> C -> back to A`, coordinator early layers, and final-worker output-head behavior.

Other entry points:

- Existing CLI forms:
  - `--role coordinator --layers 0:M`
  - `--role worker --layers N:output`
  - `--role worker --layers N:42`
- Future topology experiments:
  - control coordinator with no prefill layers
  - GPU-prefill node owns early layers while decode node owns full local generation state
  - final prefill worker returns hidden state or KV to a separate decode destination
  - output head computed on a non-final/non-prefill participant

Candidate directions to analyze:

- Split roles into control role and execution role:
  - coordinator owns prompt/session/control plane but may not execute layer `0`.
  - execution participants advertise layer ranges and capabilities independently.
- Make the route graph explicit:
  - represent ordered prefill stages, output/logit stage, return target, and KV return target separately.
  - keep contiguous layer coverage initially, but do not hard-code coordinator-first.
- Add an explicit decode owner:
  - define which participant will receive merged KV and run local decode.
  - this may be the coordinator, but should not be required forever.
- Add explicit output-head ownership:
  - replace `N:output`/`N:42` as the only mechanism with a capability/route field.
  - preserve the current flags as shorthand for the simple topology.
- Keep current topology for Phases 5 and 6:
  - use this option if local-generation handoff works and topology flexibility can be deferred without compromising correctness.

Work items:

- Document exactly where the current route requires the coordinator to own layer `0`.
- Document how output logits are requested and validated today.
- Identify which protocol fields would need to change to support a route whose first execution stage is remote.
- Identify whether hidden-state return, logits return, and KV return need separate destinations.
- Determine whether topology changes are necessary for the first production implementation or should be split into a later issue.
- Preserve compatibility with existing `--layers 0:M` / `N:output` workflows as shorthand if new topology controls are added.

Exit gate:

- The follow-on topology problem is documented with a concrete design direction or explicitly deferred.
- If deferred, the plan records why the current coordinator-first topology is acceptable for the first local-generation implementation.
- If required, Phases 4 and 5 must be revisited before user-facing implementation.

### Phase 9: Documentation and durable learnings

Goal:

- Why: this change crosses distributed transport, backend serialization, model residency, KV persistence, and user workflow; undocumented learnings will be expensive to rediscover.
- What: update user docs, runbooks, decision logs, performance notes, failure cases, and this plan so the final implementation state and remaining deferred work are clear to the next engineer.

Expected artifacts:

- Update `README.md` with the final user-visible workflow, constraints, and expected performance behavior.
- Update `artifacts/issue-304/runbook.md` with final known-good commands.
- Update `artifacts/issue-304/decision-log.md` with final architecture and remaining deferred work.
- Update `artifacts/issue-304/perf-breakdown.md` with final benchmark numbers.
- Update `artifacts/issue-304/topology-decoupling.md` if topology flexibility is deferred or partially implemented.
- Update `artifacts/issue-304/research-notes.md` with a final phase-by-phase summary of what was learned and what remains deferred.
- Keep this `PLAN.md` accurate if parts of the staged plan were skipped or invalidated.

Code/documentation touchpoints:

- `README.md`
  - Distributed inference documentation.
  - Any new CLI/server workflow.
- `ds4_cli.c`
  - Help text and flag descriptions if CLI flags were added.
- `ds4_server.c`
  - API/session documentation if server behavior changed.
- Tests added in prior phases.

Other entry points:

- GitHub issue #304 follow-up comment or PR description.
- Any local scripts/harnesses created for validation.
- Artifact files under `artifacts/issue-304/`.

Work items:

- Update `README.md` once the user-visible workflow exists.
- Keep issue-specific research notes in `artifacts/issue-304/`.
- Keep this plan current as decisions are made.
- If an unknown unknown changes the direction, add the new fact, the decision it invalidated, and the replacement hypothesis before continuing implementation.

### Initial recommended path

Start with Phase 1 and Phase 2 using the existing `DSV4` payload path. This gives the fastest correctness signal with the least new code. Only after that should the work move into API design or pipelined KV return.
