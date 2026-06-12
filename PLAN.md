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

- To what extent the currently observed route/backend variance would materially hurt realistic benchmarks or user-visible quality once a practical handoff workflow exists.
- Whether `CUDA -> Metal` resumed-decode drift is only normal engine/backend implementation variance, or whether it indicates payload/handoff state that would make distributed-prefill-to-local differ from fully local inference on the same decode engine.
- Whether distributed prefill -> merged `DSV4` -> local decode matches the official-vector and local-golden correctness surfaces already used for fully local inference strongly enough to matter in real use, rather than only as a research comparison.
- How to let the coordinator participate in distributed prefill while still keeping a full local decode-capable engine resident without exhausting Metal memory budget.
- Whether the eventual handoff path should remain a merged `DSV4` checkpoint, move to an in-memory merged payload, or skip the merged payload entirely by streaming worker-owned KV shards back during prefill.

## Implementation direction

- Keep the merged `DSV4` handoff path as the current correctness boundary. Phase 2 validated that it can gather distributed shards, stage a payload, load locally, and recover the exact handoff checkpoint.
- Treat mixed-backend resumed-decode drift as a risk characterization problem, not automatically a bug and not the current implementation bottleneck. Record it, guard against obvious regressions, and revisit only when practical workflow or benchmark evidence says it matters.
- Use `tests/test-vectors/official` / `official.vec` and the local golden-vector surface as guardrails, not as proofs of end-user equivalence.
- Shift the next implementation focus to practical handoff plumbing: turn the current proof path into a usable workflow that can be benchmarked and exercised realistically.
- Treat coordinator engine residency and practical handoff plumbing as one fused productization problem. Phase 2 proved the feature concept by releasing the distributed engine before local decode; the next work is about making that handoff runnable and benchmarkable in one user-visible workflow.
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

- `tests/ds4_test.c` and `tests/test-vectors/official.vec`: Phase 3 should reuse the official-vector and local-golden correctness gates as the primary acceptance surface.
- `tests/issue304_phase1_matrix.c` and `tests/issue304_phase2_handoff.c`: keep using these focused harnesses to isolate and classify cross-engine drift.
- `ds4.c`: inspect payload restore, decode entry, and session state setup around `ds4_session_load_payload()` and resumed `ds4_session_eval()`.
- `ds4_metal.m` and `ds4_cuda.cu`: inspect only if evidence points to backend-specific decode behavior that needs classification or documentation.
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

## Current pivot after Phase 3.5

- Phase 3 and Phase 3.5 produced enough evidence to characterize route/backend variance, but not enough to prove product harm.
- Practical next work should therefore focus on making distributed-prefill-to-local-decode usable as an actual workflow that can later be benchmarked on realistic prompts and sessions.
- Deeper variance forensics are deferred unless one of these happens:
  - practical handoff plumbing lands and realistic benchmarks show meaningful regressions,
  - resumed routes underperform direct generation in user-visible ways,
  - CUDA inference changes materially and the route matrix needs to be re-run,
  - product design starts depending on resumed routes being treated as equivalent.
- Until then:
  - official-vector and local-golden checks remain useful guardrails,
  - the existing variance findings remain documented and unresolved,
  - but they do not block the next round of handoff/product plumbing work.

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
- Treat this two-session shape as a correctness experiment. The fused Phase 4 owns the decision about whether residency plus workflow plumbing can make it practical for user-facing use.

Key questions:

- Does the coordinator have enough local model state loaded to create the full local session, or is a second full engine/session required?
- Does `--layers` imply partial model loading that conflicts with local decode?
- Does `ds4_dist_session_save_payload()` safely run immediately after `dist_coordinator_prefill_prompt_pipelined()` completes?
- Are worker session IDs and token hashes stable between prefill completion and shard fetch?

Exit gate:

- The existing payload path can hand off distributed prefill state into a local-only decode session, or a specific blocker is captured with enough detail to choose the next implementation step.
- If the path works only by running two engines or reloading an engine, it must not proceed directly to user-facing implementation; Phase 4 must choose a viable residency-plus-workflow design first.

### Phase 3: Handoff equivalence and drift isolation

Goal:

- Why: Phase 2 proved the merged handoff checkpoint is exact and local greedy decode can match distributed reference, but forced-token logits can drift after resumed eval. That drift is acceptable if it is just normal backend implementation variance; it is not acceptable if distributed prefill -> local decode differs from fully local inference on the same decode engine or fails the existing official-vector/local-golden correctness gates.
- What: isolate the drift class and prove that distributed-prefill-to-local decode matches fully local decode on the same backend for `tests/test-vectors/official` / `official.vec` and local golden-vector style checks.

Expected artifacts:

- Update `artifacts/issue-304/logit-comparisons.md` with official-vector comparisons for fully local inference and distributed-prefill-to-local decode on the same decode backend.
- Update `artifacts/issue-304/compatibility-matrix.md` with a Phase 3 row classifying drift as accepted engine variance or rejected handoff-specific mismatch.
- Update `artifacts/issue-304/failure-cases.md` with every drift case that remains rejected because it changes local-backend behavior or fails official/local-golden gates.
- Update `artifacts/issue-304/research-notes.md` with the classification result and any remaining caveat about cross-engine numeric differences.
- Update `artifacts/issue-304/decision-log.md` if the evidence shows the drift is acceptable engine variance, payload-ABI related, backend-runtime related, or decode-graph specific.

Code touchpoints:

- `tests/ds4_test.c`
  - Existing `--logprob-vectors` and `--local-golden-vectors` checks define the local correctness surface to reuse or mirror.
- `tests/test-vectors/official.vec` and `tests/test-vectors/official/*.official.json`
  - Official API top-logprob vectors are the reference surface; the hosted API exposes top-logprobs rather than full logits.
- `tests/issue304_phase1_matrix.c`
  - Extend the focused resume helper only as needed to classify whether cross-engine drift changes accepted top-token/top-logprob behavior.
- `tests/issue304_phase2_handoff.c`
  - Add or reuse a mode that can run the official-vector prompts through distributed prefill -> merged payload -> local decode and compare against fully local decode on the same backend.
- `ds4.c`
  - Inspect `ds4_session_load_payload()` and resumed `ds4_session_eval()` only if distributed handoff differs from fully local inference on the same backend.
- `ds4_gpu.h`
  - Shared tensor read/write and synchronization boundary if same-backend handoff mismatch appears below the session payload layer.
- `ds4_metal.m` and `ds4_cuda.cu`
  - Backend-specific decode-state assumptions, dtype conversion, raw/compressed KV interpretation, and graph state setup; cross-engine differences here are acceptable if they do not invalidate the same-backend handoff checks.

Work items:

- Define the Phase 3 acceptance envelope from existing local checks:
  - `./ds4_test --logprob-vectors`
  - `./ds4_test --local-golden-vectors`
  - the existing skipped `long_memory_archive` official-vector caveat
- Add a focused distributed-handoff vector check that runs the official-vector prompts through distributed prefill, stages merged `DSV4`, loads into local decode, and compares against fully local inference on the same decode backend.
- Compare distributed-prefill-to-local output to the official top-logprob vectors using the same selected-token and top-logprob tolerance rules as `--logprob-vectors`.
- Compare distributed-prefill-to-local output to fully local output for the same prompt/frontier before interpreting any cross-engine drift as a defect.
- Keep the forced-token trace probe, but use it to classify and bound backend variance rather than requiring bit-exact agreement across engines.
- Inspect `ds4_session_load_payload()`, `ds4_session_eval()`, and backend decode paths only if the same-backend fully-local vs distributed-handoff comparison fails or official/local-golden gates regress.

Exit gate:

- Distributed prefill -> local decode matches fully local inference on the same decode backend for the official-vector prompt set, within the existing official/local-golden acceptance envelope.
- Any remaining `CUDA -> Metal` forced-token drift is classified as accepted engine/backend variance, or is localized as a specific handoff defect with a reproduction.
- If official-vector or local-golden checks fail only for distributed handoff, record that mismatch and carry it forward as an active caveat; Phase 4 may still proceed if the practical workflow questions remain more urgent than deeper variance forensics.

### Phase 3.5: Six-route variance matrix and worst@5 envelope

Goal:

- Why: Phase 3 showed that one distributed-prefill-to-local-decode route can drift materially from a fresh same-backend local baseline on longer prompts. That result is not enough to separate three different causes cleanly:
  - ordinary `CUDA <-> Metal` implementation variance,
  - variance introduced by distributed generation itself,
  - variance introduced by distributed prefill followed by resumed generation from a merged `DSV4` payload.
- What: build a like-for-like measurement rig that runs the same prompt sets across six execution routes, records both official top-logprob variance and local logit/greedy variance, and reports `worst@5` for every case and route.

Required route matrix:

- `1.` pure local `CUDA` prefill + generation
- `2.` pure local `Metal` prefill + generation
- `3.` distributed prefill + generation `CUDA -> Metal`
- `4.` distributed prefill + generation `Metal -> CUDA`
- `5.` distributed prefill `CUDA -> Metal`, then resumed pure `Metal` generation from merged `DSV4`
- `6.` distributed prefill `Metal -> CUDA`, then resumed pure `CUDA` generation from merged `DSV4`

Measurement contract:

- Every route must run with `--repeat 5` semantics and emit:
  - one result record per repeat,
  - one `worst@5` summary record per case.
- Every official-vector case must emit:
  - selected-token pass/fail,
  - missing official top-token count,
  - max logprob delta,
  - route metadata,
  - timing metadata.
- Every local variance case must emit:
  - top1 match or first mismatch step,
  - top5/top20/top64 overlap,
  - RMS drift,
  - max absolute drift,
  - top20 max absolute drift,
  - greedy token divergence,
  - route metadata,
  - timing metadata.
- Each distributed or resumed route must also record:
  - coordinator host/backend,
  - worker host/backend,
  - layer split,
  - `ctx`,
  - `prefill_chunk`,
  - `prefill_window`,
  - `activation_bits`,
  - `prefill_cap`,
  - payload bytes where applicable,
  - model hash on both hosts.

Expected artifacts:

- Update `artifacts/issue-304/compatibility-matrix.md` with one row per route group and context bucket, using `worst@5` as the authoritative summary.
- Update `artifacts/issue-304/logit-comparisons.md` with:
  - official-vector `worst@5` variance tables for all six routes,
  - local/golden `worst@5` variance tables for all six routes,
  - explicit comparisons between:
    - pure local `CUDA` vs pure local `Metal`,
    - distributed generation vs pure local on the same generation backend,
    - resumed generation vs pure local on the same generation backend.
- Update `artifacts/issue-304/perf-breakdown.md` with route-specific prefill, staging, load, and generation timing envelopes from the same runs.
- Update `artifacts/issue-304/shard-metadata.md` with the authoritative route/layout metadata for every distributed and resumed route.
- Update `artifacts/issue-304/runbook.md` with exact local and remote commands, including source sync, remote build, payload copy, worker startup, and one-shot repeat structure.
- Update `artifacts/issue-304/research-notes.md` with the classification outcome:
  - whether resumed generation adds variance beyond pure backend differences,
  - whether distributed generation adds variance beyond pure local generation,
  - whether any route is acceptable under the official and local envelopes.
- Update `artifacts/issue-304/decision-log.md` with the final interpretation of the six-route matrix.
- Update `artifacts/issue-304/failure-cases.md` with any route/case combination that remains rejected after `worst@5`.

Code touchpoints:

- `tests/issue304_phase1_matrix.c`
  - Reuse for pure local save/load, forced-token trace capture, and same-prompt local backend comparisons.
- `tests/issue304_phase2_handoff.c`
  - Reuse for distributed prefill + generation and staged merged-payload export.
- `tests/issue304_phase3_vectors.c`
  - Reuse its official/local-golden parsing, parity metrics, and `worst@5` summary shape as the base reporting contract.
- New helper: `tests/issue304_phase35_vectors.c`
  - Add a single-host vector runner with three modes:
    - `local`
    - `distributed`
    - `payload`
  - It must support both:
    - `--suite official`
    - `--suite local-golden`
  - It must emit one stable NDJSON record schema across all three modes.
- New orchestrator: `tests/issue304_phase35_matrix.py`
  - Own cross-host orchestration rather than embedding SSH/rsync logic into the C helpers.
  - Responsibilities:
    - compare local and remote GGUF hashes before every run,
    - sync source to `dgx-direct:~/ds4` when code differs,
    - build local and remote helpers,
    - start/stop opposite-host workers,
    - run five one-shot repeats per case to avoid coordinator port reuse races,
    - copy merged payloads between hosts for resumed routes,
    - collect raw NDJSON into an artifact directory.
- `Makefile`
  - Add build targets for `tests/issue304_phase35_vectors`.

Execution strategy:

- Start by standardizing measurement shape, not by expanding route logic inside the existing helpers.
- Keep cross-host control in the orchestrator and keep the C helper focused on:
  - tokenization,
  - official/golden checks,
  - logit parity metrics,
  - route-local session control.
- Use the current trusted helpers to bootstrap the new runner:
  - `issue304_phase3_vectors` defines the official/golden comparison rules,
  - `issue304_phase2_handoff` defines distributed route setup and payload staging,
  - `issue304_phase1_matrix` defines the local forced-trace comparison shape.
- Treat one-shot coordinator runs with short pauses as mandatory for distributed routes; do not regress to a persistent in-process repeat loop for route-level validation.

Work items:

- Add `tests/issue304_phase35_vectors.c` with:
  - `local` mode for pure local prefill + generation on the host backend,
  - `distributed` mode for distributed prefill + generation on the coordinator host,
  - `payload` mode for loading a merged `DSV4` payload and continuing generation on the host backend,
  - official and local-golden suite support,
  - per-repeat and `worst@5` NDJSON output.
- Add `tests/issue304_phase35_matrix.py` to orchestrate:
  - route selection,
  - local/remote build,
  - worker lifecycle,
  - payload shipping,
  - result collection.
- Define a route manifest inside the orchestrator for the six required paths, including authoritative host role, backend role, layer split, and payload direction.
- Reuse the current context buckets as the first required coverage set:
  - `4096`
  - `5000`
  - `16384`
- Run official vectors for all six routes where the route can produce normal token-by-token top-logprobs.
- Run local-golden and prompt-length variance checks for all six routes.
- Record a separate comparison table that normalizes each route against the pure local run on the same generation backend.
- Only after the rig works reliably should we consider broadening prompt coverage beyond the existing official/local-golden cases.

Key questions:

- Can the distributed-generation routes use the same official top-logprob comparison rules directly, or do they need a coordinator-local top-logprob extraction shim when the output owner is remote?
- Does the opposite-host resumed route require any payload normalization beyond the current merged `DSV4` file copy?
- Are there backend-specific environment constraints, such as `DS4_METAL_PREFILL_CHUNK`, that must become part of the route manifest rather than ad hoc runbook notes?
- Do any routes need distinct frontier definitions beyond the current official and local-golden prompt surfaces to make prompt-length variance comparisons fair?

Exit gate:

- We have authoritative `worst@5` results for all six routes across the required context buckets.
- Official top-logprob variance and local logit/greedy variance are both available in one consistent result schema.
- The evidence can answer, with numbers, whether resumed generation introduces materially more drift than pure `CUDA <-> Metal` variance and whether distributed generation introduces additional drift beyond pure local generation.

### Phase 4: Final-worker residency and workflow viability

Goal:

- Why: Phase 2 currently proves the concept by releasing the distributed engine and then opening a full local engine. That is enough for research, but not enough for a practical one-shot coordinator workflow, and realistic resumed-route benchmarking depends on turning that proof path into a usable workflow.
- What: choose and implement a viable residency-plus-plumbing design that lets the final worker participate in distributed prefill over its later-layer route slice, then continue local decode as the full-resident generation node through a repeatable operator-facing workflow.
- Scope: this phase is not another variance-forensics phase. Phase 3 and Phase 3.5 variance remains an active caveat, but Phase 4 should answer whether a practical, benchmarkable handoff workflow can exist with acceptable residency and operator complexity.

Closeout note:

- Phase 4 is now complete.
- The practical answer is yes: a final-worker full-resident handoff workflow exists, works in both backend directions, and is benchmarkable.
- Ongoing caveat: strict `CUDA -> Metal` parity checks should use a fresh Metal worker process because reused-worker reruns can still produce Phase-3.5-class near-top1 variance.

Expected artifacts:

- Update `artifacts/issue-304/engine-residency.md` with the current layer-loading model, observed memory behavior, and candidate designs.
- Update `artifacts/issue-304/perf-breakdown.md` with any measured cost of wider model mapping, lazy layer activation, or switching between slice/full execution modes.
- Update `artifacts/issue-304/decision-log.md` with the chosen residency design and rejected alternatives.
- Update `artifacts/issue-304/failure-cases.md` with expected failures for missing full-decode layers, invalid layer range transitions, and memory-budget rejection.
- Update `artifacts/issue-304/runbook.md` with commands and environment settings used to measure memory.
- Update `artifacts/issue-304/research-notes.md` with the residency conclusions and the decision impact on later implementation phases.
- Update `artifacts/issue-304/runbook.md` with the intended developer/operator workflow for the first practical handoff path.
- If a temporary probe flag or harness is added, document it as experimental and keep it out of README until Phase 5 chooses the user-facing API shape.

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

- Full-resident final worker with sliced distributed execution:
  - Load the full model once on the final worker, but still advertise/execute only the worker-owned later-layer route range for distributed prefill.
  - Treat this as the preferred first experiment because it avoids engine unload/reload and avoids two model-weight copies.
  - During distributed prefill, call `ds4_session_eval_layer_slice()` only for the worker route slice even though more weights are resident.
- Decouple route ownership from weight residency:
  - Keep `--layers` as the distributed route slice, but stop forcing final-worker model mapping to that slice when a local-decode residency mode is selected.
  - Preserve the coordinator-first route model and coordinator early-layer execution; broader topology flexibility stays deferred.
  - Prefer an internal/test-only option or helper path first; defer public CLI naming to Phase 5.
- Dual session over one full-resident engine:
  - Use one full-resident final-worker engine with separate distributed-prefill and local-decode sessions.
  - This avoids duplicate weights but may duplicate KV/session graph memory; measure it before selecting.
  - Keep the distributed session and local decode session distinct unless evidence shows same-session mutation is safer.
- Lazy or expandable residency:
  - Start with route-slice mapping, then expand to full mapping before local decode without closing the engine.
  - Only keep this path if backend model map spans and cached/preloaded tensors can be expanded safely and fast enough.
- Two engines or reload:
  - Keep only as a correctness harness or fallback diagnostic, not the final workflow unless the measurements force it.

Work items:

1. Re-state the current residency contract in `engine-residency.md`.
   - Trace exactly how distributed `--layers` becomes loaded model spans today.
   - Record how `load_output_optional`, `N:output`, and `N:42` affect output-head residency.
   - Record that merged save/load still requires matching `ctx` and `prefill_cap`.

2. Build the smallest full-resident-final-worker probe.
   - Decouple final-worker route ownership from final-worker loaded spans only for the probe.
   - Keep coordinator-first topology and coordinator early-layer execution unchanged.
   - Verify the final worker can map full weights and output head while advertising/executing only `N:output` during distributed prefill.
   - Verify `ds4_session_eval_layer_slice()` works on a full-resident worker for the route-owned slice.

3. Measure residency and graph/session memory.
   - Compare route-slice final worker, full-resident final worker, dual-session-over-one-engine, and current two-engine proof harness.
   - Capture peak process memory, backend allocation behavior, payload bytes, stage/load time, and local decode t/s.
   - Record whether dual sessions over one engine duplicate only KV/session graph memory or trigger any unexpected model-weight duplication.

4. Prototype the first practical workflow with the existing `DSV4` boundary.
   - Prefer staged merged `DSV4` first; do not add in-memory or incremental KV return unless the staged path blocks viability.
   - Run distributed prefill in the distributed session.
   - Transfer or merge the coordinator-owned early-layer KV state into the final worker's full local decode session.
   - Prefer the existing merged `DSV4` boundary first if it is the lowest-risk way to prove the workflow.
   - Load the handoff state into a local decode session backed by the same full-resident final-worker engine.
   - Continue local decode and emit diagnostics for missing full-decode residency, output-head absence, layout mismatch, stale worker state, or memory-budget failure.

5. Run focused validation, not a new broad variance campaign.
   - Re-run the local guardrails that protect payload/session behavior.
   - Re-run the Phase 2-style handoff check on the DGX/Mac topology.
   - After the implementation is complete, validate both `CUDA -> Metal` and `Metal -> CUDA` routes.
   - Re-run a reduced Phase 3.5 sample only if the Phase 4 plumbing changes logits, payload layout, or backend mapping behavior.
   - Record any change in official/local behavior as a caveat, but do not require full resumed-route equivalence before proving workflow viability.

6. Make the Phase 5 decision.
   - Choose whether Phase 5 should expose whole-payload handoff, an in-memory payload helper, or an explicit shard merge.
   - Record rejected alternatives and the measured reason.
   - Do not update README or finalize CLI names until this decision is made.

Exit gate:

- A practical handoff path can run distributed prefill, stage/load the merged payload, and continue local-only decode without closing and reopening the model at handoff time.
- The preferred path uses one full-resident final-worker engine and does not hold two full model-weight copies in memory.
- If the preferred path fails, the smallest acceptable temporary compromise is explicitly recorded with measured memory/runtime costs.
- The chosen strategy is captured in `artifacts/issue-304/engine-residency.md` and `artifacts/issue-304/decision-log.md`.
- The runbook contains one known-good Phase 4 command sequence and one memory-measurement procedure.
- Completed implementation validation records both `CUDA -> Metal` and `Metal -> CUDA` route results in the existing issue artifacts.
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

- Same-session final-worker handoff:
  - keep the final worker's later-layer KV resident in its existing distributed worker session,
  - transfer only the coordinator-owned earlier-layer shard into that same worker session,
  - and continue local decode in-process on the final worker.
  - This is now the preferred Phase 5 implementation direction because Phase 4 proved it viable.
- Whole-payload handoff:
  - keep merged `DSV4` staging as a correctness harness, debug surface, and fallback diagnostic.
  - Do not treat it as the preferred user-facing implementation unless the same-session path proves too brittle.
- In-memory payload handoff:
  - only pursue if the same logical whole-payload path is still needed after Phase 5 and temp-file staging is the only material problem.
- Explicit shard merge:
  - treat this as the practical optimization path around the coordinator-owned missing shard, not as a requirement to reconstruct the full session from scratch.
  - Prefer it only in the narrow form already validated by Phase 4: worker keeps its own later-layer KV in place while loading the coordinator's earlier-layer shard.

Decision criteria:

- Correctness equivalence after resume.
- Compatibility with the chosen residency strategy.
- Operator complexity in CLI/server workflows.
- Ability to support cross-backend handoff.
- Measured handoff overhead relative to prefill speedup.
- Amount of new protocol surface in `ds4_distributed.c`.
- Fit with the validated same-session final-worker continuation shape.

Work items:

- Decide the smallest API shape for the already-chosen same-session final-worker continuation path.
- Record that decision and evidence before adding user-facing wiring.
- Add the minimal session-level handoff mechanism selected from that decision.
- Prefer same-session worker continuation; keep whole-payload `DSV4` as a debug/fallback path rather than the primary implementation.
- Ensure the final worker session receives full token timeline, logits, raw SWA KV rows in logical order, compressed KV rows, compressor frontier state, and indexer state.
- Add CLI/server wiring only after the core session path works.
- Document the operator requirement for full decode weights on the final worker and the current validation rule for strict `CUDA -> Metal` parity checks: restart the Metal worker first.

Exit gate:

- A user-visible path can run distributed prefill and then local-only decode with measured speedup and correctness artifacts.

### Phase 5.5: Bound accuracy follow-up and reusable-session parity

Goal:

- Why: Phase 5 closed the fresh-worker worker-owned local-decode workflow, but the research artifacts still record unresolved accuracy caveats around reused-session follow-up sync, route-dependent parity drift, and weak attribution between backend variance, fixture drift, and handoff-specific error.
- What: tighten the acceptance surface for post-Phase-5 accuracy work, improve reusable-session follow-up behavior where practical, and classify the remaining misses well enough that later phases are not blocked on open-ended parity chasing.

Expected artifacts:

- Update `artifacts/issue-304/compatibility-matrix.md` with a Phase 5.5 section covering the fixed reusable-session matrix for both backend directions and both greedy/sampled follow-up runs.
- Update `artifacts/issue-304/logit-comparisons.md` with reusable-session frontier metrics and any fresh-route versus reused-route parity deltas needed to attribute surviving drift.
- Update `artifacts/issue-304/failure-cases.md` with any Phase 5.5 accuracy miss that remains a true rejection, and remove any case that is reclassified as bounded backend variance or fixture drift.
- Update `artifacts/issue-304/decision-log.md` with the chosen Phase 5.5 acceptance thresholds, the final attribution buckets, and the explicit stop rule used to close the phase.
- Update `artifacts/issue-304/research-notes.md` with what changed after the Phase 5 closeout, especially reusable-session findings and any fixture/baseline revisions.
- Update `artifacts/issue-304/ds4-eval-matrix.md` only if the authoritative evaluator score delta changes materially after Phase 5.5 fixes.

Code touchpoints:

- `ds4_distributed.c`
  - Worker-owned local-decode activation, follow-up sync, coordinator catch-up, token-hash/session-integrity checks, and reused-session state transitions.
- `ds4.c`
  - Session sync/eval/logits bookkeeping only if the reusable-session mismatch localizes below the distributed handoff layer.
- `ds4_cli.c`, `ds4_server.c`, and `ds4_eval.c`
  - Only if one frontend still bypasses the intended worker-owned path or preserves stale state across reused turns.
- `tests/ds4_test.c`
  - Extend the distributed local-decode regression surface for reusable-session follow-up parity or integrity failures.
- Existing issue-304 helpers
  - Reuse the focused Phase 3/3.5/5 harnesses before creating new broad matrices.

Acceptance surface:

- Fresh-worker one-shot worker-owned local decode remains the authoritative correctness baseline for the shipped workflow.
- Reused-session follow-up accuracy is measured on a fixed matrix:
  - `Metal -> CUDA` and `CUDA -> Metal`,
  - one greedy follow-up path,
  - one sampled follow-up path with fixed seeds,
  - and the same prompt/transcript inputs across reused-versus-fresh comparisons.
- Official-vector acceptance remains necessary but not sufficient.
- The stored `local-golden` fixture is not treated as a pure local-Metal oracle unless it is first recalibrated and revalidated.
- Bit-exact cross-backend logits are not a requirement for closure.

Work items:

- Reproduce the current reused-session follow-up drift on the fixed Phase 5.5 matrix and record the exact frontier metrics before changing behavior.
- Audit the worker-owned follow-up flow for stale logits, stale coordinator KV, token timeline/hash drift, and worker reconnect contamination before treating any mismatch as backend variance.
- Compare reused-route follow-up results against:
  - the matching fresh full-transcript route on the same backend,
  - the direct-generation backend baseline where relevant,
  - and the known Phase 3.5 backend-variance/fixture-drift surfaces.
- Fix any concrete state-management defect that explains reused-session drift, starting with the smallest coordinator/worker bookkeeping change that improves parity without weakening fail-closed integrity checks.
- If a mismatch remains after state-management fixes, classify it into exactly one bucket:
  - backend-specific long-prompt variance,
  - `local-golden` fixture drift,
  - or handoff/reuse-specific variance.
- Preserve hard rejection for token-hash, payload-layout, reconnect, stale-session, or stale-KV integrity failures even if user-visible text still looks coherent.

Exit gate:

- Fresh-worker one-shot worker-owned local decode still passes the existing authoritative Phase 5 surface, including the full `ds4-eval --nothink` workflow without new official-vector regressions.
- The fixed reusable-session Phase 5.5 matrix is fully populated for both backend directions and both greedy/sampled follow-up paths.
- Reused-session follow-up behavior is either:
  - improved enough to meet the declared Phase 5.5 thresholds on that matrix,
  - or shown to be bounded non-integrity variance that does not produce prompt-control-flow breakage or clear answer-quality regressions.
- Every surviving accuracy miss on the fixed matrix is attributed to exactly one bucket: backend variance, fixture drift, or handoff/reuse-specific variance.
- No open reusable-session issue remains that points to stale coordinator KV, token-hash/session-integrity failure, payload-layout mismatch, or worker-reconnect contamination.
- Do not block closure on bit-exact parity, elimination of every near-top1 flip, or turning the current `local-golden` fixture into a perfect canonical oracle.

Closeout note:

- Phase 5.5 is complete.
- Reused-session follow-up differences were attributed to bounded
  prefill-vs-decode numerical trajectory variance, not to handoff or
  reused-state integrity failure.
- The decisive proof was:
  - replay diagnostics on the distributed matrix, where reused sessions
    matched fresh decode-replay references exactly,
  - plus local-only Metal and CUDA reproductions showing the same
    prefill-vs-decode split without any distributed handoff path.

### Phase 6: Profile performance bottlenecks across context lengths and session shapes

Goal:

- Why: Phase 5 and Phase 5.5 closed the correctness and attribution work enough to benchmark the practical workflow, and current evidence does not show KV handoff as the dominant cost.
- What: build an authoritative performance profile for distributed-prefill plus worker-owned local decode across representative context lengths and single-turn/multi-turn usage, then use that profile to decide whether any optimization work, including pipelined KV return, is actually warranted.

Expected artifacts:

- Update `artifacts/issue-304/perf-breakdown.md` with a benchmark matrix that records:
  - route direction,
  - prompt or transcript token count,
  - context size,
  - session shape,
  - distributed prefill sec and tok/s,
  - KV handoff sec, bytes, and MiB/s,
  - worker local-decode sec and tok/s,
  - first-token latency where measurable,
  - follow-up sync sec for multi-turn runs,
  - and end-to-end wall time per turn.
- Update `artifacts/issue-304/runbook.md` with the canonical commands used for the benchmark matrix.
- Update `artifacts/issue-304/decision-log.md` with bottleneck attribution for each workload class and the explicit rule for whether KV pipelining stays deferred or is reopened later.
- Update `artifacts/issue-304/research-notes.md` with a concise summary of what workloads are prefill-bound, handoff-bound, decode-bound, or follow-up-sync-bound.

Code touchpoints:

- Prefer no transport or protocol redesign in this phase.
- Reuse existing timing surfaces first:
  - CLI `--debug` KV handoff timing,
  - existing issue-304 handoff helpers,
  - and the existing Phase 5 multi-turn helper.
- Touch code only for measurement support if current timing is insufficient:
  - `ds4_distributed.c` for finer-grained handoff or sync timing,
  - `ds4_cli.c` to expose stage timing cleanly in the user-visible workflow,
  - and existing issue-304 helpers or scripts to automate the benchmark matrix.
- Do not implement pipelined or chunked KV return in this phase unless the plan is explicitly revised after profiling.

Other entry points:

- Network timing measurements on the available DGX/Mac topology and any direct-link setup already used for earlier issue-304 runs.
- Existing `--dist-activation-bits`, `--dist-prefill-chunk`, and `--dist-prefill-window` flags only as benchmark parameters, not as new feature surface.
- The fixed Phase 5.5 follow-up transcript pair for reused-session measurements.

Work items:

- Populate a minimum benchmark matrix for both `Metal -> CUDA` and `CUDA -> Metal`.
- Measure at least two session shapes:
  - single-turn one-shot handoff,
  - and multi-turn reused-session follow-up on the worker-owned local-decode path.
- Measure at least four context buckets where feasible:
  - short prompt (`~10-100` tokens),
  - medium prompt (`~1k` tokens),
  - long prompt (`~4k+` tokens),
  - and the current very-long `README.md` frontier at `ctx=16384`.
- Record relevant pure-local decode-backend baselines and continued distributed-decode baselines where those numbers already exist or can be collected cheaply with the current tools.
- Classify each measured scenario into exactly one dominant bottleneck bucket:
  - distributed prefill compute,
  - KV handoff transport,
  - local decode compute,
  - or follow-up sync or prefill.
- Keep `ds4-eval` out of scope for this phase; evaluator-specific performance work is covered separately.

Scope note:

- Phase 6 is not a transport redesign phase.
- It exists to decide, with measurements, whether pipelined KV return is still worth pursuing after the practical worker-owned local-decode workflow is in place.

Exit gate:

- The benchmark matrix is populated for the required session shapes and context buckets, with at least one authoritative passing run per route direction where the topology supports it.
- `artifacts/issue-304/perf-breakdown.md` reports stage-level timing, not just aggregate runtime, for prefill, handoff, decode, and follow-up sync where applicable.
- `artifacts/issue-304/decision-log.md` identifies the dominant bottleneck for each workload class and names the next optimization target, if any.
- KV pipelining is explicitly classified into one of two outcomes:
  - remains deferred because KV handoff is a minor tail relative to end-to-end runtime,
  - or is reopened as a later optimization because profiling shows it materially limits user-visible latency.
- Use this default materiality rule unless new data forces a stronger one:
  - reopen KV pipelining only if handoff is consistently a major contributor, such as roughly `>10%` of end-to-end latency on long-prompt runs or roughly `>25%` of first-token latency on short-prompt runs, or if it is the dominant bottleneck in multi-turn follow-up behavior.
- Do not close the phase with raw timing tables alone; the closeout must include an explicit bottleneck conclusion and a recommended follow-on target.

### Phase 7-8: Failure hardening and closeout documentation

Goal:

- Why: the worker-owned local-decode workflow is now implemented and profiled, so the remaining issue-304 work is to make its failure boundaries explicit, fail closed on stale or mismatched state, and leave a durable operator and engineering record of what shipped versus what stayed deferred.
- What: run one combined closeout pass that hardens rejection and recovery behavior, normalizes the authoritative failure matrix, and updates user-facing and research documentation so the final implementation state is unambiguous.

Scope and non-goals:

- This phase is about the current worker-owned local-decode workflow only.
- Do not treat bounded Phase 3.5 or Phase 5.5 numerical variance as a fresh correctness blocker unless new evidence shows stale state, payload corruption, or protocol misuse.
- Do not reopen pipelined KV return here; Phase 6 already re-scoped that decision.
- Do not let topology decoupling or coordinator-owned local decode work block this closeout phase.

Expected artifacts:

- Update `artifacts/issue-304/failure-cases.md` into the authoritative failure matrix for the shipped workflow.
- Update `artifacts/issue-304/runbook.md` with exact negative-path and recovery commands, alongside the known-good Phase 5 and Phase 6 commands.
- Update `artifacts/issue-304/decision-log.md` with the final issue-304 closeout decisions, especially which failures remain hard rejects and which open questions are now deferred.
- Update `artifacts/issue-304/research-notes.md` with a final closeout summary for the issue: implemented workflow, accepted caveats, preserved rejection cases, and deferred follow-on work.
- Update `README.md` with the final user-visible workflow, required invariants, fail-closed behavior, and the practical "what is deferred" notes.
- Keep this `PLAN.md` accurate if any closeout finding invalidates a staged assumption.

Code touchpoints:

- `ds4_distributed.c`
  - `dist_kv_route_validate()`: route and layout mismatch handling.
  - `dist_worker_handle_snapshot_save()`: token hash, session identity, and worker-state validation on shard export.
  - `dist_worker_handle_snapshot_load()`: streamed shard-load rejection behavior.
  - `dist_worker_handle_local_generate()`: post-handoff token hash validation on delegated local decode.
  - `dist_coordinator_rebuild_from_transcript()`: replay and rebuild path after route loss or worker replacement.
  - `dist_coordinator_ensure_route()` and route planning helpers.
  - Error propagation through coordinator sync, handoff, save/load, and eval paths.
- `ds4.c`
  - `ds4_session_load_payload()`: normal full-payload rejection behavior.
  - `ds4_session_load_layer_payload()` and streaming layer-payload helpers: shard metadata and payload-shape validation.
- `tests/ds4_test.c`
  - Extend focused negative and CLI-parse cases that can run without a full distributed topology.
- Existing issue-304 helpers under `tests/`
  - Reuse or add one dedicated negative-path harness for reproducible multi-host rejection and reconnect coverage if `ds4_test` alone is too narrow.

Other entry points:

- Worker restart or reconnect workflow.
- Missing or non-advertised local-decode worker capability.
- Final worker that does not own `N:output`.
- Mismatched coordinator/worker context or layout metadata.
- Model id, model family, or layer-count mismatch at worker registration.
- Token transcript mutation or stale session state between prefill completion and shard fetch or delegated decode.
- Historical fixed failures that should stay documented but should not be reclassified as open blockers, especially the old `60s` distributed socket timeout and the unsafe second CUDA residency request.

Work items:

- Normalize the failure taxonomy into explicit buckets:
  - configuration rejects,
  - route or capability rejects,
  - shard and session integrity rejects,
  - reconnect or replay recovery paths,
  - historical fixed failures.
- Preserve hard fail-closed behavior for token count or hash mismatch, model or layer mismatch, layout mismatch, payload byte mismatch, stale epoch traffic, and missing output-head ownership.
- Verify that reconnect or route-loss cases either:
  - rebuild from transcript,
  - report an incomplete route,
  - or reject the handoff,
  - but never silently reuse stale worker KV.
- Add or tighten diagnostics where the current error text is too generic to identify the broken boundary.
- Capture for each preserved negative case:
  - expected rejection,
  - observed error text,
  - reproduction command,
  - and whether the case is a hard reject, a replay or recovery case, a historical fixed failure, or accepted variance.
- Update `README.md` to document:
  - that `--local-decode` is worker-only,
  - that it requires `--layers N:output`,
  - that distributed participants must match commit and layout expectations,
  - that disconnects and stale-state mismatches fail closed,
  - and that topology decoupling and KV-pipeline work remain deferred.
- Close the issue-specific documentation loop by aligning `README.md`, `runbook.md`, `failure-cases.md`, `research-notes.md`, `decision-log.md`, and this plan on the same final status language.

Exit gate:

- The current worker-owned local-decode workflow has an authoritative documented failure matrix, not just ad hoc notes.
- Known integrity failures fail closed with useful diagnostics and no silent stale-state reuse path remains accepted.
- At least one reproducible reconnect or route-recovery scenario is documented alongside the normal known-good commands.
- The final docs clearly separate:
  - shipped workflow,
  - accepted numerical caveats,
  - preserved hard rejection cases,
  - and deferred future work.
- Issue 304 can be treated as closed on the current workflow, with topology decoupling and later optimization work explicitly deferred rather than left ambiguous.

### Phase 9: Topology decoupling follow-on

Goal:

- Why: practical deployments may want the best GPU machine to own early prefill while a different machine owns later layers and generation, so control-plane role, prefill layer ownership, output-head/logit ownership, KV return destination, and decode owner should not remain permanently coupled.
- What: document the current coordinator-first topology constraints, then design a route/protocol direction that allows any role to own generation effectively, including cases where a worker prefills early layers while the coordinator owns later layers and generation.

This is intentionally after the combined Phase 7-8 closeout so the first production handoff path, its caveats, and its preserved rejection boundaries are documented before route structure is reopened. Do not let this phase block the first correct local-generation handoff unless later product requirements prove the current topology insufficient.

Phase 4 conclusion for this phase:

- Topology decoupling is still deferred for the first implementation.
- The current coordinator-first topology is acceptable for the first production implementation as long as the final worker owns `N:output`, keeps full decode residency, and becomes the local decode owner after prefill.
- Phase 9 should therefore start from a documented working baseline and then ask what protocol/API changes are needed to let generation ownership move independently of today's coordinator/worker split.

Expected artifacts:

- Update `artifacts/issue-304/topology-decoupling.md` with the current topology constraints, desired future topologies, and candidate protocol/API changes.
- Update `artifacts/issue-304/decision-log.md` with whether topology decoupling remains deferred, becomes partially required, or becomes required for the next implementation stage.
- Update `artifacts/issue-304/perf-breakdown.md` with measurements that motivate topology changes, especially GPU-prefill throughput versus memory-bandwidth-local decode throughput.
- Update `artifacts/issue-304/runbook.md` with any topology experiments and command lines.
- Update `artifacts/issue-304/research-notes.md` with topology-related findings and whether they affect the implementation scope after the first production path.

Current constraints to capture:

- The coordinator route must currently start at layer `0`.
- The coordinator runs the early layer slice during distributed prefill.
- Workers cover later contiguous slices.
- `--layers N:output` makes the final worker own the output head and return logits.
- `--layers N:42` keeps the output head/logit production on the coordinator, but the worker still owns the later transformer slice up to layer 42.
- Coordinator/worker roles currently imply control-plane ownership, prompt/session ownership, route construction, and a position in the layer pipeline.

Desired future topology properties:

- Control-plane coordinator should be separable from prefill execution role.
- Any role should be able to own generation if it has the required later layers, output head, and full KV state.
- Early prefill ownership should be assignable to a worker while the coordinator owns later layers and generation.
- The machine that performs local decode should be separable from the machine that runs the earliest prefill layers.
- Prefill pipeline stages should be assignable to the machines with the best GPU throughput, not necessarily to the decode/sampling machine.
- Decode/sampling should be placeable on the machine with the best memory bandwidth and enough resident full-model/KV state.
- Output-head/logit production should be an explicit route property rather than an implicit consequence of `N:output` versus `N:42`.
- KV ownership and KV return destination should be explicit enough to support "GPU-prefill workers -> memory-rich decoder" without forcing awkward layer ownership.

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
  - worker owns early layers while coordinator owns later layers and generation
  - GPU-prefill node owns early layers while a different decode node owns full local generation state
  - final prefill worker returns hidden state or KV to a separate decode destination
  - output head computed on a non-final/non-prefill participant

Candidate directions to analyze:

- Split roles into control role and execution role:
  - coordinator owns prompt/session/control plane but may not execute layer `0`.
  - execution participants advertise layer ranges and capabilities independently.
- Make the route graph explicit:
  - represent ordered prefill stages, output/logit stage, generation owner, return target, and KV return target separately.
  - keep contiguous layer coverage initially, but do not hard-code coordinator-first.
- Add an explicit generation owner / decode owner:
  - define which participant will receive merged KV and run local decode.
  - allow that participant to be a worker or the coordinator.
- Add explicit output-head ownership:
  - replace `N:output`/`N:42` as the only mechanism with a capability/route field.
  - preserve the current flags as shorthand for the simple topology.
- Keep current topology for Phases 5 and 6:
  - use this option if local-generation handoff works and topology flexibility can stay deferred without compromising the first implementation.

Work items:

- Document exactly where the current route requires the coordinator to own layer `0`.
- Document how output logits are requested and validated today.
- Identify which protocol fields would need to change to support a route whose first execution stage is remote.
- Identify which protocol fields would need to change to let generation ownership live on a different role than the final prefill worker.
- Identify whether hidden-state return, logits return, and KV return need separate destinations.
- Determine whether topology changes are necessary for the next implementation stage or should remain a later issue.
- Preserve compatibility with existing `--layers 0:M` / `N:output` workflows as shorthand if new topology controls are added.

Exit gate:

- The follow-on topology problem is documented with a concrete design direction or explicitly deferred again.
- If deferred, the plan records why the current coordinator-first topology is still acceptable after the first implementation lands.
- If required, the plan identifies exactly which earlier assumptions in Phases 5-7 would need to be revisited.

### Phase 10: Adaptive in-session routing for large appended frontiers

Goal:

- Why: agentic coding sessions do not grow smoothly. Many turns add only a small prompt delta, but some turns append whole files or large tool outputs. A fixed "always distributed prefill" or "always local prefill" policy is therefore likely to leave latency on the table.
- What: design and validate a routing policy that decides, per turn, whether the new uncached frontier should be prefetched locally on the decode owner or through distributed prefill, using rolling measurements gathered from earlier prefills in the same live instance and topology.

This phase assumes the reused-session state path is already correct enough to move KV ownership in either direction. The question here is not state integrity, but whether the next uncached frontier should stay local or switch to distributed prefill because the expected latency win is material.

Expected artifacts:

- Update `artifacts/issue-304/perf-breakdown.md` with crossover measurements for local-only versus distributed catch-up on appended frontiers.
- Update `artifacts/issue-304/runbook.md` with the commands or harness needed to sweep appended-frontier sizes in a reused session.
- Update `artifacts/issue-304/decision-log.md` with the selected routing rule, fallback rule, and any reasons a dynamic policy was rejected or narrowed.
- Update `artifacts/issue-304/research-notes.md` with the measured crossover behavior and whether the best policy is route-specific, topology-specific, or backend-specific.
- Create `artifacts/issue-304/adaptive-routing.md` if the design or telemetry model is substantial enough that it should not be buried inside the general notes.

Code touchpoints:

- `ds4_distributed.c`
  - Reused-session catch-up entry points and route rebuild logic.
  - Timing/telemetry surfaces for sync, prefill, handoff, and rebuild overhead.
  - The route-selection point where a reused session decides whether the next uncached frontier stays local or becomes a distributed prefill.
- `ds4.c`
  - Session-side telemetry storage if the rolling cost model lives on the session or instance.
  - Local-only catch-up timing surfaces if they are not already exposed cleanly.
- `ds4.h`
  - Any state needed for rolling performance estimates, confidence bands, or hysteresis.
- `ds4_cli.c`
  - Optional debug flags for forcing local-only catch-up, forcing distributed catch-up, dumping routing decisions, or disabling adaptive routing for controlled comparison.
- `tests/issue304_phase5_multiturn.c` or a follow-on helper
  - Reused-session benchmark harness that varies only the appended uncached frontier.

Other entry points:

- The current direct-link DGX/Mac topology and any later topologies that materially change the local-versus-distributed crossover point.
- Session shapes where the transcript is mostly cached and a new file read or tool dump appends `100s` to `1000s` of new tokens at once.
- Existing route directions separately:
  - `CUDA -> Metal`
  - `Metal -> CUDA`

Candidate policy shape:

- Base the decision on new uncached tokens, not total session length.
- Keep a rolling in-instance cost model using recent measured turns:
  - local-only catch-up throughput
  - distributed prefill throughput
  - follow-up sync sec
  - KV handoff sec and throughput
  - route rebuild or route-formation overhead where applicable
- Estimate:
  - `T_local(new_tokens)`
  - `T_dist(new_tokens)`
- Choose distributed only when:
  - `T_dist + margin < T_local`
- Use hysteresis so the session does not flap around the crossover point.
- Bucket or separate the model by route direction and topology rather than assuming one universal threshold.

Scope note:

- Do not hard-code a single global threshold like `N=1000` unless the measurements show that a simple threshold is both stable and good enough.
- Prefer an online cost model calibrated from the current live instance over a static token heuristic.
- If the adaptive policy is too noisy or brittle, phase down to a per-route threshold with explicit logging rather than shipping an opaque predictor.

Work items:

- Add the telemetry needed to estimate local-only versus distributed catch-up cost from live turns.
- Build a reused-session benchmark sweep that appends controlled frontiers, for example:
  - `64`
  - `128`
  - `256`
  - `512`
  - `1024`
  - `2048`
  - `4096`
  tokens of new uncached content.
- Measure the crossover point separately for:
  - `CUDA -> Metal`
  - `Metal -> CUDA`
- Compare at least these policies:
  - always local catch-up
  - always distributed catch-up
  - static threshold policy
  - rolling-cost adaptive policy
- Record how often the adaptive policy would have chosen the wrong path relative to observed wall time.
- Add debug output that makes each routing decision auditable:
  - observed stats used
  - predicted local cost
  - predicted distributed cost
  - selected path
  - post-hoc actual cost
- Define the safe fallback behavior when recent stats are missing, stale, or inconsistent.

Exit gate:

- There is an authoritative crossover matrix for appended uncached frontiers on the intended topology, split by route direction.
- The plan records whether the best practical rule is:
  - dynamic rolling-cost routing,
  - a simple per-route threshold,
  - or no adaptive switching because the crossover is too small or unstable.
- The chosen rule is explainable from measured costs and has auditable debug output.
- If adaptive routing remains justified, the plan identifies exactly where the decision is made in the session lifecycle and what fallback is used when telemetry is insufficient.
- If adaptive routing is rejected, the artifacts explicitly say why:
  - crossover too small,
  - telemetry too noisy,
  - excessive route flapping risk,
  - or complexity not justified by measured wins.

### Current recommended path

Start Phase 4 with the full-resident-final-worker probe and keep the merged `DSV4` payload as the first handoff boundary. Do not move into user-facing API shape, pipelined KV return, or topology decoupling until the final-worker residency workflow is either proven viable or rejected with measured memory/runtime evidence.
