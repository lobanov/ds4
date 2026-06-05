# Issue 304 Phase 5 Plan: API Shape And User-Facing Handoff Workflow

This plan converts the Phase 4 research-only handoff into a usable, benchmarkable,
operator-facing workflow. It is intentionally narrow: it does not reopen Phase 3
variance forensics, does not change residency, and does not change route topology.
It is the smallest step that turns `tests/issue304_phase4_handoff` into something
a real frontend can call.

## 2026-06-05 implementation status

Phase 5 is now substantially implemented in the main CLI/runtime path.

Implemented:

- `--local-decode` is parsed by the shared distributed CLI parser and is
  accepted only on worker processes that own `N:output`.
- `--local-decode` implies full worker residency and is advertised in
  HELLO/route metadata.
- The coordinator pushes its local KV shard to the output-owning worker
  after distributed prefill using stream-based `DSVL` helpers; the
  temp-file detour was removed from this handoff path.
- Reusable coordinator sessions catch up their local slice from the
  generated token ids returned by worker local generation.
- One-shot CLI coordinator runs now attempt the worker-owned handoff path
  directly for both greedy and sampled generation.
- The coordinator now preserves the worker's final logits after handoff
  even when the caller does not request a full logits trace.
- The normal distributed `ds4_session_eval(token)` path now lazily
  activates worker local decode after sync and delegates subsequent token
  eval through forced one-step worker `LOCAL_GENERATE` calls.
- `ds4_server` inherits that path automatically because it already samples
  on the coordinator and advances through `ds4_session_eval()`.

Measured on 2026-06-05 with the plain `ds4` CLI:

- `Metal -> CUDA`, full `README.md`, `ctx=16384`:
  - distributed prefill: `603.15 tok/s`
  - KV handoff: `105,725,160` bytes in `0.345 s` (`292.47 MiB/s`)
  - worker local generation: `13.08 tok/s`
- `CUDA -> Metal`, fresh Metal worker, full `README.md`, `ctx=16384`:
  - distributed prefill: `589.93 tok/s`
  - KV handoff: `107,097,320` bytes in `0.350 s` (`291.42 MiB/s`)
  - worker local generation: `30.33 tok/s`
- Pure local Mac baseline, full `README.md`, `ctx=16384`:
  - local prefill: `413.44 tok/s`
  - local generation: `34.78 tok/s`

Current limitation:

- Reusable-session catch-up no longer looks broadly broken, but the next
  follow-up sync on top of a caught-up reused state can still drift from a
  fresh full-transcript session before turn two begins.
- The severity of that remaining reusable-session drift differs by route:
  on the tested sampled normal-eval seed, `Metal -> CUDA` still diverged in
  second-turn tokens while `CUDA -> Metal` preserved the same second-turn
  token sequence despite frontier drift.

## Source documents

- `PLAN.md` Phase 5 section defines the goal, candidate options, and decision criteria.
- `NAVIGATION.md` documents the public API, distributed orchestration, and validation surface.
- `artifacts/issue-304/decision-log.md` records the upstream decisions:
  - Phase 4 final-worker handoff is viable in both backend directions.
  - `CUDA -> Metal` strict parity requires a fresh Metal worker process.
  - Whole-payload `DSV4` handoff remains a debug/fallback, not the primary path.
- `artifacts/issue-304/runbook.md` captures the current closeout rules and the
  Phase 4 operator workflow.
- `artifacts/issue-304/engine-residency.md` confirms the preferred residency
  design (full-resident final worker, sliced distributed execution during prefill).
- `artifacts/issue-304/failure-cases.md` lists the rejection cases the new
  workflow must continue to reject.

## Current State After Phase 4

- A working same-session final-worker handoff exists and is reachable from C.
- The current public API is:
  - `ds4_session_distributed_route_ready()` in `ds4.h`
  - `ds4_session_distributed_route_summary()` in `ds4.h`
  - `ds4_session_distributed_handoff_argmax()` in `ds4.h`
  - `ds4_session_distributed_handoff_argmax_trace()` in `ds4.h`
  - `ds4_dist_session_handoff_argmax()` / `..._trace()` in `ds4_distributed.h`
- The implementation is in `ds4_distributed.c`:
  - `dist_session_handoff_argmax_trace()` validates route, validates shard layout,
    saves the coordinator's local slice to a temp file via
    `ds4_session_save_layer_payload()`, hands the bytes to the final worker via
    `dist_load_remote_shard_from_payload()`, copies the current logits, and
    dispatches `DS4_DIST_MSG_LOCAL_GENERATE_REQ` to the worker.
  - The worker decodes locally and returns tokens through the new
    `DS4_DIST_MSG_LOCAL_GENERATE_RES` message family.
- The current path is **coordinator-initiated**: the coordinator session
  calls `ds4_dist_session_handoff_argmax()` and orchestrates the transfer.
  Phase 5 inverts this: the **worker** initiates the local-decode transition
  by asking the coordinator to ship its KV cache back.
- The harness in `tests/issue304_phase4_handoff.c` and
  `tests/issue304_phase4_diagnose.c` is the only existing user of the API.
- No CLI flag, no server wiring, and no README documentation exists yet.
- `Makefile` already builds both helpers as standalone binaries.
- `ds4_dist_parse_cli_arg()` is the single shared CLI parser for
  distributed options. It is already called by `ds4_cli.c`,
  `ds4_server.c`, `ds4_bench.c`, `ds4_agent.c`, and `ds4_eval.c`. New
  distributed flags added here are automatically recognized by all five
  frontends.

## Phase 5 Goal

> Turn the Phase 4 in-process handoff into a real, user-facing, benchmarkable
> workflow by having the worker advertise its local-decode intent and full
> residency during the normal HELLO handshake, then letting the coordinator
> push its KV cache back to the worker in memory (no temp file) using the
> existing snapshot framing. No new decode or catch-up protocol is required.

### Architectural direction

The Phase 4 path is coordinator-initiated: the coordinator calls
`ds4_dist_session_handoff_argmax()`, saves its local shard, ships it to the
worker, and dispatches the local-generate request. Phase 5 inverts the
direction while reusing more of the existing machinery:

- The **worker** owns the `--local-decode` flag and the local-decode
  decision.
- During the existing HELLO handshake, the worker advertises
  `wants_local_decode = true` to the coordinator. The coordinator records
  this on its `ds4_dist_worker_entry` for the worker.
- After distributed prefill completes, the coordinator pushes its local
  KV shard back to the output-owning worker using the **existing**
  `DS4_DIST_MSG_SNAPSHOT_*` family (`BEGIN`/`CHUNK`/`DONE` and the
  corresponding `SNAPSHOT_LOAD_BEGIN` on the worker).
- The worker's existing `dist_worker_handle_snapshot_load()` receives the
  bytes, validates them, and calls `ds4_session_load_layer_payload()` on
  its own session.
- The worker then owns local generation using the existing local
  `ds4_session_eval()` path and returns generated token ids through the
  existing local-generate response.
- Reusable coordinator sessions use those returned token ids to catch up
  their coordinator-owned KV before accepting a subsequent prompt. One-off
  coordinator runs can ignore catch-up because they exit after printing the
  worker output.

Why this is simpler than a dedicated pull protocol:

- The coordinator already serializes worker-owned shards and the worker
  already loads them. The new direction is just "coordinator serializes
  its own shard, worker loads it." The framing and chunking code paths
  are reused as-is.
- The decision to push is communicated once, in HELLO, not as a
  separate request after prefill. This avoids a new request-response
  roundtrip and a new message family.
- The same push channel can later stream KV chunks during prefill instead
  of after, which is the natural Phase 6 (pipelined KV return) extension.
  No protocol rework is needed to add KV pipelining; only the timing of
  the push changes.

Implications for the coordinator:

- A new boolean field on `ds4_dist_worker_entry` records
  `wants_local_decode`. Set during HELLO parsing.
- A new post-prefill step in the coordinator checks whether the final
  route entry has `wants_local_decode = true`, and if so, triggers the
  existing snapshot-push path to that worker.
- The coordinator's startup code path does not change. The new field is
  learned through HELLO, which is the normal registration event.
- Reusable coordinator sessions must track whether local KV is current
  after worker-owned generation. If not current, they forced-evaluate the
  exact returned worker tokens through the coordinator-owned slice before
  processing the next prompt.

Implications for the worker:

- A new field on `ds4_dist_options` (e.g. `bool local_decode`) is parsed
  by the shared CLI parser.
- The worker startup path validates:
  - the worker's layer range is `N:output` (or otherwise owns the output
    head),
  - and exits with a clear error if not.
- `--local-decode` implies full worker residency, replacing the operator
  need to set `DS4_DIST_WORKER_FULL_RESIDENT` for the user-facing path.
- During HELLO, the worker sends `wants_local_decode = opt->local_decode`
  to the coordinator.
- The existing `dist_worker_handle_snapshot_load()` handler does not
  need to change. It already calls `ds4_session_load_layer_payload()`
  and validates token hash, model id, and layer range. The handler
  only needs to know it is being called for the local-decode handoff
  case so it can also accept the coordinator's local-start range
  (currently the snapshot path expects the worker's own range).

### Single-flag parsing

`ds4_dist_parse_cli_arg()` in `ds4_distributed.c` is the only place that
parses distributed CLI flags today. It is already invoked by `ds4_cli.c`,
`ds4_server.c`, `ds4_bench.c`, `ds4_agent.c`, and `ds4_eval.c`. Adding
`--local-decode` here makes it recognized by all five frontends at once.
The worker is the only consumer of the parsed value; the coordinator
simply ignores it.

### Narrow current support

The Phase 4 implementation only supports local decode on a worker that owns
the output head (`--layers N:output`). Until Phase 9 (topology rewiring)
lands, `--local-decode` must succeed only in the narrow configuration and
must fail loudly everywhere else.

| Configuration | `--local-decode` behavior |
| --- | --- |
| worker role + `--layers N:output` (or any range that owns the output head) | succeed: worker advertises local-decode/full-resident capability in HELLO, coordinator pushes KV after prefill, worker owns local generation |
| worker role + `--layers N:42` (output on coordinator) | fail at startup: "local-decode requires worker-owned output head; use --layers N:output" |
| worker role + multi-worker route where this worker is not the final hop | fail at startup: "local-decode is only valid on the worker that owns the final layer range" |
| coordinator role | ignored (with a stderr note that it only applies to workers) |
| none role (not distributed) | fail at startup: "local-decode requires --role worker" |
| any frontend in any config | flag is accepted and forwarded; the startup check above still applies |

Note: the "no coordinator connection yet" case is intentionally absent.
The worker has not connected to the coordinator at startup, so the
check would always trigger. The connection is a transient state that
resolves at HELLO time. The runtime guarantee is that
`wants_local_decode` only takes effect once the worker successfully
registers; the coordinator will not push a KV shard to a worker that
never advertised the intent.

The validation belongs in the worker-side `ds4_distributed.c` startup
path, not in each frontend's `parse_options`. This keeps the frontends
dumb: they accept the flag, write it to the shared `ds4_dist_options`
field, and let the worker session code apply or reject it.

The phase 5 exit gate, copied from `PLAN.md`:

> A user-visible path can run distributed prefill and then local-only decode with
> measured speedup and correctness artifacts.

## API Shape Decision

Phase 5 must pick one of these before adding any public wiring:

| Option | How the KV gets to the worker | Reuses existing machinery? | Verdict |
| --- | --- | --- | --- |
| (a) **Coordinator pushes KV, advertised in HELLO, streaming with fixed staging buffer** | Worker advertises `wants_local_decode` and full residency in HELLO; coordinator pushes via the existing snapshot framing using a compile-time staging size after prefill | Yes — same `DS4_DIST_MSG_SNAPSHOT_*` family and `dist_worker_handle_snapshot_load()`; `ds4_session_*_layer_payload_stream` replaces the temp-file detour | **Preferred.** Smallest protocol delta. Bounded per-handoff memory. No disk IO on the handoff path. Naturally extends to pipelined push in Phase 6. |
| (b) Coordinator-initiated handoff (current Phase 4 path) | Coordinator calls `ds4_dist_session_handoff_argmax()` directly | N/A | Keep as a low-level primitive and debug/diagnostic surface. Do not expose through `--local-decode`. |
| (c) Worker-pulled KV with a dedicated protocol | Worker sends a new `DS4_DIST_MSG_KV_PULL_REQ` after prefill | No — new request/response family | Reject. Adds a new roundtrip and message family for no benefit over (a). |
| (d) Whole-payload `DSV4` handoff (Phase 2 path) | Coordinator stages `DSV4` to disk, worker loads | Partially | Keep as debug/fallback only. Do not expose as primary. |
| (e) In-memory payload handoff (without the HELLO advertisement change) | In-memory snapshot transfer | No | Subsumed by (a) — (a) already does the in-memory path. |

### Decision

**Adopt option (a): coordinator pushes KV, advertised in HELLO,
streaming with a fixed staging buffer.** The new user-facing flag
`--local-decode` lives on the worker. The worker advertises its
local-decode intent and full-resident capability during the normal
HELLO handshake. The coordinator records the intent on the worker
entry, then pushes its local KV shard back to the worker after prefill
using the existing snapshot framing, but with a streaming payload path
(`ds4_session_save_layer_payload_stream` /
`ds4_session_load_layer_payload_stream`) that uses a compile-time
staging size and eliminates the temp-file detour. No new decode or
catch-up protocol is required, no disk IO is involved in the handoff,
and the design naturally extends to pipelined KV return in Phase 6.

The existing `ds4_session_distributed_handoff_argmax()` /
`..._trace()` API remains available as a coordinator-side low-level
primitive and as a debug/diagnostic surface. It is no longer wired to
the user-facing flag.

The new API surface for option (a):

- `ds4_dist_options` gains one field:
  - `bool local_decode` — parsed from `--local-decode`.
- `ds4_dist_worker_entry` gains a `bool wants_local_decode` field.
  The coordinator sets it from the worker's HELLO message.
- A new post-prefill step in the coordinator (in
  `dist_coordinator_*` somewhere after the final prefill chunk ACK):
  - checks whether the final route entry has `wants_local_decode = true`,
  - and if so, calls the streaming snapshot-push path
    (`ds4_session_save_layer_payload_stream`) targeted at the final
    worker. The `write_chunk` callback sends
    `DS4_DIST_MSG_SNAPSHOT_CHUNK` frames using the fixed staging size.
- The worker's existing `dist_worker_handle_snapshot_load()` handler
  does not need a new dispatch path. It already receives wire
  chunks; the change is that the chunks are streamed into
  `ds4_session_load_layer_payload_stream` via a `read_chunk`
  callback instead of being accumulated in a temp file. The handler
  must accept the coordinator's local layer range (currently the
  snapshot path expects the worker's own range).

The API must satisfy these contract rules:

- `--local-decode` on the worker must require the worker to own the
  output head (`N:output`). Reject `N:42` at startup because the
  worker would not have the output head to produce logits locally.
- `--local-decode` on the worker must require this worker to be the
  final hop in the route. Reject intermediate workers at startup.
- The KV push must validate token hash, model id, ctx, prefill_cap,
  and layer range exactly as the existing `dist_kv_route_validate()`
  does for save/load.
- The push must preserve the token timeline, logits, raw SWA rows in
  logical order, compressed KV rows, compressor frontier state, and
  indexer state. The existing `DSVL` shard load path already covers
  these.
- After the worker loads the coordinator's KV shard, the worker
  owns generation using the existing local session API and returns
  generated token ids via the existing local-generate response.
- Reusable coordinators must forced-evaluate returned worker token ids
  through their local slice before processing a subsequent prompt.
  One-off coordinators may skip this catch-up.
- The flag must surface timing for KV push and local decode so callers
  can attribute end-to-end cost.

## Phase 5 Work Items

The items are ordered so each one produces a small, testable artifact before
the next one is started.

### 1. Extend the HELLO protocol with the local-decode advertisement

Goal: let the worker tell the coordinator it wants local decode during the
existing registration handshake.

Work:

- In `ds4_distributed.c` and `ds4_distributed.h`:
  - Add a `bool local_decode` field to `ds4_dist_options` (parsed from CLI
    in work item 2).
  - Add a `bool wants_local_decode` field to `ds4_dist_worker_entry`.
  - Extend the HELLO message wire format to include a local-decode/full-resident
    flag. The current HELLO already advertises layer range and output-head
    ownership; this is a small extension in the same record.
  - Update HELLO parsing on the coordinator side to read the new byte and
    set `entry->wants_local_decode` accordingly.
  - Update HELLO serialization on the worker side to write the new byte
    from `opt->local_decode`.
- No new message types are introduced. HELLO is already the registration
  message; the new field is an extension of its existing payload.

Exit evidence: a focused unit test in `tests/ds4_test.c` that round-trips
a HELLO with `wants_local_decode = true` through the wire format and
verifies the coordinator reads it back correctly.

### 2. Parse `--local-decode` in the shared distributed CLI parser

Goal: make the flag recognized by all five frontends at the CLI level
without per-frontend changes.

Work:

- In `ds4_distributed.c`:
  - The `bool local_decode` field on `ds4_dist_options` was added in
    work item 1.
  - Add a `--local-decode` case to `ds4_dist_parse_cli_arg()` that sets
    `opt->local_decode = true`. No argument.
  - Add the new flag to `ds4_dist_usage()` with the
    "worker-only" caveat.
- No frontend-level changes are required:
  - `ds4_cli.c`, `ds4_server.c`, `ds4_bench.c`, `ds4_agent.c`, and
    `ds4_eval.c` all call `ds4_dist_parse_cli_arg()` already. They pick
    up the new flag automatically.
  - The worker reads `opt->local_decode` and advertises it in HELLO; the
    coordinator ignores it.

Exit evidence: `ds4 --help` and `ds4-bench --help` both list
`--local-decode`; running the binary with the flag in an unsupported
configuration produces the documented error message and a non-zero exit.

### 3. Wire the worker-side local-decode startup check

Goal: the worker rejects the configuration at startup if it cannot do
local decode, before any prefill work, and loads enough model state to
own decode after handoff.

Work:

- In `ds4_distributed.c`, the worker startup path (the part that runs
  when a `--role worker` process opens its engine and prepares to
  connect to the coordinator) must:
  - check `opt->local_decode`,
  - verify that the worker's layer range is `N:output` (or otherwise
    owns the output head),
  - verify that this worker is the final hop in the route
    (i.e., it is the only worker, or the last in a multi-worker
    chain),
  - make the worker full-resident for model weights/output when
    `--local-decode` is set,
  - and reject the configuration with the documented error message if
    any check fails.
- The check must run before any distributed prefill or decode work
  begins so the failure cost is zero.
- The "coordinator connection" state is intentionally not checked here:
  the worker has not yet connected, and the check would always
  trigger. The runtime guarantee is that the KV push only happens when
  the worker has successfully registered via HELLO with
  `wants_local_decode = true`.
- Frontend `parse_options` code must not duplicate this validation;
  they only forward the boolean.

Exit evidence: a focused unit test in `tests/ds4_test.c` that calls the
startup check on each unsupported configuration and verifies the right
error code and message.

### 4. Add the coordinator-side KV push after prefill

Goal: when the final worker has advertised local decode, the coordinator
pushes its own KV shard back to that worker using the existing snapshot
framing.

Work:

- In `ds4_distributed.c`, the coordinator's prefill completion path
  (after the final prefill chunk is acknowledged) must:
  - check whether the final route entry has `wants_local_decode = true`,
  - and if so, call the snapshot-push path targeted at the final
    worker. The path now uses the streaming payload API (work item
    4b), not a temp file.
- Extend `dist_worker_handle_snapshot_load()` to accept the
  coordinator's local layer range (the worker's own range is the
  current default). Token hash, model id, and layer range validation
  already exist and should be reused.
- Capture timing for the push so the helper and CLI can report it.
- The existing coordinator-initiated
  `ds4_dist_session_handoff_argmax()` / `..._trace()` stays as a
  low-level primitive. It is not exposed through the user-facing flag.

Exit evidence: a focused unit test in `tests/ds4_test.c` that exercises
the coordinator's post-prefill KV push against a fake worker and
verifies the worker receives the bytes and can resume local decode.

### 4b. Add streaming payload path with a fixed staging buffer (eliminates the temp file)

Goal: avoid the temp-file detour on both coordinator and worker sides
of the handoff. Per `in-memory-payload-investigation.md`, the current
snapshot framing makes the KV shard hit `/tmp` four times per
handoff. The streaming path replaces the temp file with a small
fixed staging buffer that is reused for every chunk. The staging size
is a compile-time constant for Phase 5; do not add CLI tuning until
measurements show it is needed.

Work:

- In `ds4.c`:
  - Add `ds4_session_save_layer_payload_stream(session,
    write_chunk, ctx, layer_start, layer_end,
    bytes_out, err, errlen)`. Walks the layer range, reads from GPU
    in fixed-size staging chunks, and calls `write_chunk` for each
    chunk. The total payload size is returned in `*bytes_out` so the
    caller can populate the `DS4_DIST_MSG_SNAPSHOT_BEGIN` frame.
  - Add `ds4_session_load_layer_payload_stream(session, read_chunk,
    ctx, payload_bytes, tokens, n_tokens,
    layer_start, layer_end, err, errlen)`. Walks the layer range,
    calls `read_chunk` to fill the staging buffer, and writes to
    GPU. Validation (token hash, model id, layer range, ctx,
    prefill_cap) all happens before the first GPU write.
  - Refactor `payload_write_tensor_span` and `payload_read_tensor_span`
    into chunked GPU read / GPU write helpers that take a staging
    buffer and a chunk callback. Keep the `FILE *`-based path as a
    thin wrapper that provides a `FILE *`-based chunk callback so
    the existing local save/load tests continue to work.
  - The header / per-layer metadata read / write also moves to the
    chunk callback.
- In `ds4_distributed.c`:
  - Coordinator: replace `dist_tmpfile_or_err(...)` +
    `ds4_session_save_layer_payload(fp, ...)` with the fixed staging
    buffer and `ds4_session_save_layer_payload_stream(...)`. The
    `write_chunk` callback sends a `DS4_DIST_MSG_SNAPSHOT_CHUNK`
    per staging-buffer fill.
  - Worker: replace the chunked `fwrite` to a temp file and
    `ds4_session_load_layer_payload(tmp, ...)` with the streaming
    load function. The `read_chunk` callback receives the next
    wire chunk and copies bytes into the staging buffer.
  - Wire framing (`DS4_DIST_MSG_SNAPSHOT_*`) is unchanged. The wire
    chunk size (`DS4_DIST_SNAPSHOT_CHUNK_BYTES`, 8 MB) is
    the natural default for the fixed staging size unless code-level
    implementation constraints require a smaller internal buffer.

Exit evidence:

- The streaming path produces the same token sequences as the
  existing temp-file path on the authoritative DGX/Mac topology.
- Per-handoff memory growth is bounded by the fixed staging buffer
  plus the existing wire chunk buffer.
  No temp file is created.
- No `/tmp/ds4-dist-*` files appear during a successful handoff
  (verifiable with `ls /tmp/ds4-dist-*` before/after).
- Timing is at least no worse than the temp-file path for small
  payloads and measurably faster for payloads above ~50 MB.
- No staging-size CLI or public API is added in Phase 5.

### 4c. Add coordinator catch-up for reusable sessions

Goal: keep coordinator-owned KV current after worker-owned local
generation so subsequent prompts can safely reuse the distributed
session.

Work:

- Use the generated token ids already returned by
  `DS4_DIST_MSG_LOCAL_GENERATE_RES`; do not add a new catch-up
  protocol.
- In reusable coordinator/session paths, after worker local generation
  returns:
  - append the generated token ids to the coordinator transcript,
  - forced-evaluate those exact tokens through the coordinator's local
    layer slice,
  - mark coordinator local KV current only after replay reaches the
    worker decode frontier.
- In one-off coordinator paths, skip coordinator catch-up after
  receiving and printing the worker output.
- If catch-up fails in a reusable session, reject the next prompt
  rather than continuing with stale coordinator KV.

Exit evidence: a reusable-session integration run can perform
distributed prefill -> worker local generation -> coordinator catch-up
-> subsequent prompt prefill -> second worker local generation without
stale-token or layer-layout failures.

### 5. Regression test in `tests/ds4_test.c`

Goal: make the new flow executable by the standard regression target.

Work:

- Add a `--local-decode-push` test entry in `ds4_test.c` that:
  - constructs a `ds4_dist_options` with
    `role = DS4_DISTRIBUTED_WORKER`,
    `layers = {start, has_output = true}`,
    `local_decode = true`,
  - requires a worker process to be reachable on `127.0.0.1:1234` (or
    skips if no worker is present),
  - runs a small distributed prefill,
  - waits for the coordinator's post-prefill KV push to complete,
  - receives generated token ids from the worker local-generate path,
  - catches up the coordinator local KV when the test is running in
    reusable-session mode,
  - validates timing fields are populated,
  - and emits a one-line summary.
- Gate the test behind an environment variable (e.g.
  `DS4_RUN_DISTRIBUTED_TEST=1`) so default `make test` still passes
  without a second process.

Exit evidence:
`DS4_RUN_DISTRIBUTED_TEST=1 ./ds4_test --local-decode-push`
prints a `pass` line and `make test` remains green without the variable
set.

### 6. Runbook update

Goal: an operator can run the new workflow from a copy-pasteable command.

Work:

- Update `artifacts/issue-304/runbook.md` with:
  - the new worker-side CLI form (`--role worker --layers N:output --local-decode`),
  - the fact that `--local-decode` implies full worker residency,
  - the same startup memory guard rules as Phase 4,
  - the `CUDA -> Metal` fresh-worker caveat,
  - the one-off vs reusable-session catch-up rule,
  - and a "what good output looks like" example.
- Add a Phase 5 section above the existing Phase 4 closeout rules so the
  workflow steps are obvious to the next engineer.

Exit evidence: the runbook has a "Phase 5 worker-initiated workflow" section
that matches the flag set actually implemented in `ds4_distributed.c`.

### 7. README update (minimal, deferred-until-stable)

Goal: do not over-document while the workflow is still moving.

Work:

- Add a short paragraph in the distributed-inference section noting that
  distributed prefill can hand off to local-only decode on the worker.
- Point at the runbook for the exact commands.
- Do not promise strict same-token parity across repeated runs; preserve the
  Phase 3.5 caveat language about backend variance.

Exit evidence: the README has at least one sentence about the handoff workflow
that is consistent with what the CLI actually does.

### 8. Failure case documentation

Goal: keep the negative cases recorded so the next implementer does not
regress them.

Work:

- Append to `artifacts/issue-304/failure-cases.md`:
  - `--local-decode` on a worker that does not own `N:output`.
  - `--local-decode` on an intermediate worker in a multi-worker route.
  - `--local-decode` on a coordinator process (silently ignored with a
    stderr note).
  - `--local-decode` on a non-distributed (none-role) process.
  - Stale coordinator/worker session (token hash mismatch on the
    pushed shard).
  - Worker prefill-cap mismatch (already known from Phase 3).
  - `--local-decode` accepted (not rejected as unknown) by `ds4`,
    `ds4-server`, `ds4-bench`, `ds4-agent`, and `ds4-eval` `--help`,
    but the worker startup check fails the runtime configuration.
  - Note that "no coordinator connection" is intentionally not a
    failure case: the connection is a transient state and the KV push
    only happens once HELLO has been processed.
  - In-memory staging buffer allocation failure. The handoff should
    fail with a clear "out of memory" diagnostic before any GPU
    write happens.
  - Reusable coordinator catch-up failure after worker local
    generation. The next prompt must fail clearly rather than using
    stale coordinator KV.

Exit evidence: `failure-cases.md` has a "Phase 5" section that lists each
case and the diagnostic the user sees.

### 9. Performance and correctness refresh

Goal: re-collect the post-implementation numbers in the existing artifacts.

Work:

- Run the new worker-side CLI form on the DGX/Mac topology for both
  `Metal -> CUDA` and `CUDA -> Metal` (with the fresh-Metal-worker
  rule for the latter).
- Update `artifacts/issue-304/perf-breakdown.md` with:
  - prefill sec, in-memory KV push sec, local decode sec, local tok/s,
  - reusable-session coordinator catch-up sec and any catch-up tail
    latency,
  - the existing distributed decode sec/tok/s for comparison,
  - the issue-reference distributed prefill and pure local generation
    timing for comparison,
  - and a side-by-side row comparing the old temp-file KV push
    timing to the new in-memory KV push timing on the same prompt
    (verifies work item 4b delivered the expected speedup).
- Update `artifacts/issue-304/logit-comparisons.md` with:
  - handoff logits parity (expect `top1` match, `rms=0`, `max_abs=0`),
  - greedy continuation for the sampled window,
  - and the standard same-backend fresh-local comparison.
- Update `artifacts/issue-304/compatibility-matrix.md` with a Phase 5 row
  per backend direction.

Exit evidence: a `## Phase 5` section in each of the three artifacts that
shows the new numbers and routes them to the correct cells.

### 10. Decision log update

Goal: leave a clear record of what was decided and what was rejected.

Work:

- Append a `2026-06-XX` entry to `artifacts/issue-304/decision-log.md`
  recording:
  - the selected API shape (coordinator-pushed KV, advertised in HELLO),
  - the rejected alternatives with one-line reasons (coordinator-initiated
    is kept only as a low-level primitive; worker-pulled with a dedicated
    protocol was rejected in favor of reusing the snapshot framing; whole-
    payload and explicit shard merge are deferred),
  - the inclusion of the streaming payload path with a compile-time
    fixed staging buffer (work item 4b) and the side-by-side timing
    comparison vs. the previous temp-file path,
  - the reusable-session coordinator catch-up requirement and the
    one-off coordinator exemption,
  - the inversion of the Phase 4 design: the coordinator no longer
    needs to know about `--local-decode` at startup,
  - the operator-facing rules the workflow assumes (full-decode worker
    weights, fresh Metal worker for strict `CUDA -> Metal` parity,
    matching `ctx` and `prefill_cap`).

Exit evidence: the decision log has a Phase 5 entry dated within the same
session that the code lands.

## Validation Plan

After implementation, run these in order and stop on the first failure:

1. `make`
2. `make test` (must pass without `DS4_RUN_DISTRIBUTED_TEST`)
3. `DS4_RUN_DISTRIBUTED_TEST=1 ./ds4_test --local-decode-push`
   (must pass with the second-process setup if available, otherwise
   skip)
4. The new worker-side CLI form on the DGX/Mac topology in both
   directions:
   - `Metal -> CUDA` on a reused CUDA worker
   - `CUDA -> Metal` on a fresh Metal worker
5. `./ds4_test --logprob-vectors` (must still pass)
6. `./ds4_test --local-golden-vectors` (must still pass)
7. `./ds4_test --local-payload-resume` (must still pass)
8. Spot-check that `--local-decode` is accepted (and not rejected as
   unknown) by `ds4 --help`, `ds4-server --help`, `ds4-bench --help`,
   `ds4-agent --help`, and `ds4-eval --help`.
9. Negative cases: invoke `--local-decode` in each unsupported
   configuration (none role, `N:42` worker layers, intermediate
   worker in a multi-worker route) and verify the failure messages
   match the "Narrow current support" table.
10. Streaming path verification: run the authoritative DGX/Mac
    handoff with the new streaming path. Confirm that
    `ls /tmp/ds4-dist-*` is empty during a successful handoff (no
    temp files are created). Confirm the streaming KV push time is
    at least no worse than the previous temp-file KV push time on
    the same prompt, and ideally measurably faster for payloads
    above 50 MB. Confirm the generated token sequence is identical
    to the previous Phase 4 helper output.
11. Reusable-session verification: run distributed prefill -> worker
    local generation -> coordinator catch-up -> next prompt prefill ->
    second worker local generation, and confirm no stale-KV or token-hash
    failure occurs.

If any step regresses, fix or document the regression in
`artifacts/issue-304/failure-cases.md` before declaring Phase 5
complete.

## Phase 5 Exit Gate

Phase 5 is complete when:

- A single boolean flag (`--local-decode`) is parsed by
  `ds4_dist_parse_cli_arg()` and recognized by `ds4`, `ds4-server`,
  `ds4-bench`, `ds4-agent`, and `ds4-eval` `--help`.
- The flag is read only by the worker. The coordinator's startup path
  is unchanged.
- The flag succeeds in the narrow current configuration (worker role
  with `--layers N:output`, on the final hop of a single-worker
  route) and fails with the documented message in every other
  configuration, regardless of which frontend passed it in.
- `--local-decode` implies full worker residency and advertises that
  capability in HELLO.
- The coordinator pushes its KV shard after distributed prefill, and
  the worker owns local generation through the existing local-generate
  request/response path.
- Reusable coordinator sessions catch up their local KV from the
  generated token ids returned by the worker before accepting another
  prompt. One-off coordinator sessions may skip catch-up.
- The CLI invocation produces the same kind of timing data the
  Phase 4 helper produces.
- Both `Metal -> CUDA` and `CUDA -> Metal` pass at least once on the
  authoritative DGX/Mac topology.
- The regression test in `tests/ds4_test.c` runs (or cleanly skips)
  under `make test`.
- The decision log, runbook, perf breakdown, logit comparisons, and
  compatibility matrix all have a Phase 5 entry.
- The README has at least one paragraph about the handoff workflow.
- The failure cases doc lists the user-visible negative cases the new
  flag can produce.

## Out Of Scope For Phase 5

These are explicitly deferred to later phases and should not block Phase 5:

- Pipelined / incremental KV return (Phase 6). The in-memory buffer
  path from work item 4b is a prerequisite for this: it removes the
  disk IO so the remaining cost is wire time plus memory copies.
  Pipelining then re-orders the chunked transfer relative to
  prefill.
- Topology decoupling (Phase 9).
- Multi-worker route handoff (deferred to topology work).
- Strict same-token parity on a reused Metal worker (carried forward as a
  Phase 3.5-class caveat).
- New official/local-golden vector cases (revisit after Phase 6 if needed).
- Wire-to-GPU streaming (defer; the in-memory buffer path is enough
  for current payload sizes).
