# DS4 Codebase Navigation

This document is a handoff map for working in this repository, with extra detail around issue #304: distributed prefill with local generation. It is intentionally explicit. Prefer updating this file when new control-flow facts, constraints, or validation entry points are discovered.

## Quick Orientation

- `ds4.c`
  - Core model loading, weights, engine/session lifecycle, local sync/eval, graph execution, CPU fallback, full `DSV4` payloads, per-layer `DSVL` payloads.
- `ds4.h`
  - Public engine/session API, distributed options, payload constants, low-level graph-slice APIs used by distributed inference.
- `ds4_distributed.c`
  - Distributed transport, coordinator route planning, pipelined prefill, distributed decode, worker runtime, route failure recovery, remote KV shard save/load.
- `ds4_distributed.h`
  - Distributed backend API used by `ds4.c`.
- `ds4_gpu.h`
  - Backend-neutral tensor/kernels API implemented by Metal and CUDA.
- `ds4_metal.m`
  - Metal backend implementation and graph tensor operations.
- `ds4_cuda.cu`
  - CUDA backend implementation of the same `ds4_gpu.h` surface.
- `ds4_kvstore.c` / `ds4_kvstore.h`
  - Disk KV cache envelope and store/load integration around engine payloads.
- `ds4_cli.c`
  - CLI entry point and user-visible command/flag wiring.
- `ds4_server.c`
  - Server runtime, session workflow, persistent KV behavior.
- `README.md`
  - User-facing architecture and distributed inference documentation.
- `tests/ds4_test.c`
  - Main integration/regression test harness.
- `tests/cuda_long_context_smoke.c`
  - CUDA long-context kernel smoke tests.
- `Makefile`
  - Primary build and test commands.

## Top-Level Entry Points

- CLI binary:
  - `ds4_cli.c`
  - drives normal chat/generation, distributed coordinator mode, and worker mode through shared engine/session APIs.
- Server binary:
  - `ds4_server.c`
  - uses normal engine/session APIs and `ds4_kvstore` for persistent session/KV behavior.
- Bench/eval/agent binaries:
  - `ds4_bench.c`, `ds4_eval.c`, `ds4_agent.c`
  - useful for workflows, but not the core issue #304 implementation surface.
- Distributed standalone path:
  - `ds4_distributed.c`
  - `ds4_dist_run()` runs coordinator/worker role logic after CLI parsing.

## Public API Boundary

Relevant file:

- `ds4.h`

Important public types:

- `ds4_engine`
  - Opaque loaded model/backend object.
- `ds4_session`
  - Opaque mutable inference state: checkpoint timeline, logits, KV state, optional distributed coordinator state.
- `ds4_engine_options`
  - Model/backend configuration.
  - Contains explicit load-slice fields: `load_layer_start`, `load_layer_end`.
  - Contains `ds4_distributed_options distributed`.
- `ds4_distributed_options`
  - Distributed role, route layer range, listen/connect addresses, prefill chunk/window, activation bits, debug/replay options.
- `ds4_distributed_layers`
  - Parsed inclusive layer range plus `has_output`.
- `ds4_session_payload_file`
  - Temp-file staged payload handle.
- `ds4_session_snapshot`
  - In-memory payload buffer; currently not supported for distributed sessions.

Important public functions:

- `ds4_engine_open()`
  - Opens model, initializes backend, applies layer residency decisions.
- `ds4_engine_close()`
  - Releases model/backend resources.
- `ds4_session_create()`
  - Allocates mutable inference state.
  - Attaches a distributed coordinator session when engine role is coordinator.
- `ds4_session_sync()`
  - Synchronizes live session to a prompt token prefix.
  - Distributed coordinators delegate to `ds4_dist_session_sync()`.
- `ds4_session_eval()`
  - Evaluates one generated token.
  - Distributed coordinators delegate to `ds4_dist_session_eval()`.
- `ds4_session_eval_layer_slice()`
  - Low-level graph-slice execution primitive used by distributed inference.
- `ds4_session_eval_output_head_from_hc()`
  - Computes logits from final hidden state when the output head is local.
- `ds4_session_save_payload()` / `ds4_session_load_payload()`
  - Full topology-neutral `DSV4` session save/load.
- `ds4_session_save_layer_payload()` / `ds4_session_load_layer_payload()`
  - Per-layer-range `DSVL` shard save/load used by distributed mode.
- `ds4_session_stage_payload()` / `ds4_session_write_staged_payload()`
  - File-backed staging around `ds4_session_save_payload()`.
- `ds4_session_save_snapshot()` / `ds4_session_load_snapshot()`
  - In-memory wrappers around payload save/load, but explicitly reject distributed sessions today.

## Core Engine And Session Flow

Relevant files:

- `ds4.c`
- `ds4.h`

Key internal structs:

- `struct ds4_engine` in `ds4.c`
  - Owns model map, weights, vocab, backend, distributed options, MTP state, quality/power settings.
- `struct ds4_session` in `ds4.c`
  - Owns engine pointer, optional `ds4_dist_session *distributed`, graph/CPU KV state, checkpoint tokens, logits, prefill cap, context size, checkpoint validity.

Engine open flow:

- `ds4_engine_open()`
  - Loads GGUF/model metadata and tokenizer.
  - Initializes `ds4_weights`.
  - Applies distributed layer loading before backend mapping:
    - if `opt->distributed.role != DS4_DISTRIBUTED_NONE` and `opt->distributed.layers.set`, `load_layer_start/load_layer_end` are derived from `opt->distributed.layers`.
    - `load_output` follows `layers.has_output`.
    - coordinator output-head loading may be optional via `load_output_optional`.
  - For graph backend:
    - calls `ds4_gpu_init()`.
    - configures model map with `ds4_gpu_set_model_map_range()` for full mapping or `ds4_gpu_set_model_map_spans()` for sliced mapping.
    - slice spans are computed by `weights_model_map_spans()`.
    - optional cache/preload work goes through `ds4_engine_preload_pro_q4_expert_tables()` and `accelerator_cache_model_tensors()`.

Session create flow:

- `ds4_session_create()`
  - CPU sessions are direct, but distributed coordinator sessions require graph backend.
  - Graph sessions allocate `ds4_gpu_graph` with `metal_graph_alloc_raw_cap()`.
  - `weights_first_bound_layer()` determines available layer shape from loaded weights.
  - Allocates logits and optional MTP logits.
  - Coordinator sessions attach `ds4_dist_session_create()`.

Local sync/eval flow:

- `ds4_session_sync()`
  - Distributed coordinator: delegates to `ds4_dist_session_sync()`.
  - Local session: if current checkpoint is prompt prefix, extends suffix; otherwise rebuilds from scratch.
- `ds4_session_eval_internal()` / `ds4_session_eval()`
  - Distributed coordinator: delegates to `ds4_dist_session_eval()`.
  - Local graph: runs full local decode path.
  - CPU path exists but does not support distributed graph slices.

Layer-slice execution:

- `ds4_session_eval_layer_slice()`
  - Requires graph backend.
  - Requires requested layers are loaded: `weights_layers_bound()`.
  - Requires nonzero layer input hidden state unless `layer_start == 0`.
  - Can output logits only when the requested slice ends at final transformer layer.
  - This is the key primitive that lets a full-resident engine still execute only a distributed route-owned slice.

## Model Layer Residency

Relevant files:

- `ds4.c`
- `ds4.h`
- `ds4_cli.c`
- `README.md`
- `ds4_gpu.h`
- `ds4_metal.m`
- `ds4_cuda.cu`

Current behavior:

- Distributed `--layers` currently affects both:
  - distributed route ownership, and
  - which model tensor spans are mapped/loaded.
- This coupling is implemented around `ds4_engine_open()`:
  - distributed layers become `load_layer_start`, `load_layer_end`, and `load_output`.
  - `weights_model_map_spans()` creates tensor spans for that range.
  - `ds4_gpu_set_model_map_spans()` restricts backend-visible model mapping.
- `README.md` documents this behavior: `--layers` controls which tensors are mapped.

Issue #304 consequence:

- Distributed prefill works well with sliced model residency.
- Local generation after prefill needs full model weights and full KV state on the decode machine.
- Reloading the engine or running two full engines is likely impractical.
- The implementation probably needs to decouple route ownership from weight residency.

Primary code questions:

- Can the coordinator map full model weights while advertising/executing only its route-owned prefill slice?
- Does `ds4_session_create()` allocate graph/KV state for all layers when the engine has full weights?
- Does `ds4_session_eval_layer_slice()` already allow sliced execution over a full-resident engine?
- How does output head residency interact with `load_output_optional`?
- Can backend model-map spans be expanded after engine open, or must the decision be made at startup?

Likely starting points:

- `ds4_engine_open()`
- `weights_model_map_spans()`
- `weights_layers_bound()`
- `ds4_gpu_set_model_map_spans()`
- `ds4_gpu_set_model_map_range()`
- `ds4_engine_preload_pro_q4_expert_tables()`
- `accelerator_cache_model_tensors()`
- `dist_coordinator_build_route_plan()`
- `ds4_session_eval_layer_slice()`

## Payloads And KV Persistence

Relevant files:

- `ds4.h`
- `ds4.c`
- `ds4_kvstore.h`
- `ds4_kvstore.c`
- `ds4_distributed.c`

Whole-session payload:

- Magic/version:
  - `DS4_SESSION_PAYLOAD_MAGIC` is `DSV4`.
  - `DS4_SESSION_PAYLOAD_VERSION` is `2`.
  - `DS4_SESSION_PAYLOAD_U32_FIELDS` is `13`.
- Body includes:
  - shape/layout header,
  - exact checkpoint tokens,
  - last logits,
  - per-layer `n_comp`,
  - per-layer `n_index_comp`,
  - raw SWA rows in logical order,
  - compressed attention rows,
  - ratio-4 indexer rows,
  - compressor/indexer frontier state.

Layer-shard payload:

- Magic/version:
  - `DS4_SESSION_LAYER_PAYLOAD_MAGIC` is `DSVL`.
  - `DS4_SESSION_LAYER_PAYLOAD_VERSION` is `1`.
  - `DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS` is `14`.
- Used for a contiguous `[layer_start, layer_end]` shard.
- Graph-only.
- Requires the caller to supply exact tokens and token count on load.

Important functions:

- `ds4_session_payload_bytes()`
  - Exact local payload size.
  - Returns `0` for distributed sessions despite distributed save being supported.
- `ds4_session_stage_payload()`
  - Writes temp file through `ds4_session_save_payload()` and records actual byte count.
  - Useful for distributed payload sizing.
- `ds4_session_save_payload()`
  - Local CPU/GPU full payload save.
  - Distributed coordinator delegates to `ds4_dist_session_save_payload()`.
- `ds4_session_load_payload()`
  - Local CPU/GPU full payload restore.
  - Distributed coordinator delegates to `ds4_dist_session_load_payload()`.
- `ds4_session_save_snapshot()` / `ds4_session_load_snapshot()`
  - Memory wrappers that reject distributed sessions today.
- `ds4_session_save_layer_payload()` / `ds4_session_load_layer_payload()`
  - Per-layer `DSVL` save/load.
  - Used by workers and distributed snapshot paths.

KV store integration:

- `ds4_kvstore.c` owns the outer `KVC` envelope.
- Engine/session payload bytes inside the KVC file remain normal `DSV4`.
- `ds4_kvstore_store_live_prefix_text()`
  - validates live tokens,
  - stages payload,
  - writes header + text key + payload + optional trailer.
- `ds4_kvstore_try_load_text()`
  - finds best text-prefix cache match,
  - validates model/quant/context/hash/text prefix,
  - calls `ds4_session_load_payload()`.
- Existing KV store can persist distributed coordinator payloads because staging calls `ds4_session_save_payload()`.

Issue #304 implications:

- The lowest-risk correctness proof is distributed prefill -> coordinator staged `DSV4` -> full local non-distributed session load -> local decode.
- If memory-only handoff is required, add or adapt a distributed-capable memory payload API; current snapshot helpers reject distributed sessions.
- If explicit shard merge is required, build it around existing `DSVL` semantics and still restore full checkpoint/logits metadata.

## Distributed Orchestration

Relevant files:

- `ds4_distributed.h`
  - Public distributed backend API used by `ds4.c`.
- `ds4_distributed.c`
  - Transport, route planning, coordinator orchestration, worker runtime, pipelined prefill, decode dispatch, KV shard save/load.
- `README.md`
  - Distributed behavior, layer split examples, pipeline notes, route failure recovery, snapshot save/load semantics.

Key structs:

- `ds4_dist_session`
  - Coordinator-owned session wrapper.
  - Holds coordinator state, listener, active route plan, session id, request id.
- `ds4_dist_coordinator_state`
  - Coordinator registry and runtime config.
  - Tracks local layer range, model id, worker list, prefill chunk/window, activation transport bits.
- `ds4_dist_worker_entry`
  - Registered worker metadata from `HELLO`.
  - Includes peer/data endpoint, model id, quant bits, layer range, output-head flag.
- `ds4_dist_route_plan`
  - Resolved worker chain.
  - Contains executable route entries and serialized route blob sent in `WORK`.
- `ds4_dist_route_entry`
  - One remote hop: endpoint, layer range, flags, first-hop fd.
- `ds4_dist_worker_state`
  - Worker runtime: local slice metadata and per-session KV state map.
- `ds4_dist_worker_session`
  - Worker-side session/KV state keyed by coordinator `session_id`.
  - Tracks rolling token hash.
- `ds4_dist_prefill_sender`, `ds4_dist_prefill_send_slot`, `ds4_dist_prefill_result_reader`
  - Pipelined prefill sender/result-reader state.
- `ds4_dist_kv_layout`, `ds4_dist_kv_shard_file`
  - Shard metadata used to validate/merge/split distributed KV payloads.

Public distributed API:

- `ds4_dist_session_create()`
  - Creates coordinator distributed session, listener, accept loop.
- `ds4_dist_session_route_ready()`
  - Checks if a complete route is available.
- `ds4_dist_session_sync()`
  - Distributed prefill/sync entry point.
- `ds4_dist_session_eval()`
  - Distributed one-token decode entry point.
- `ds4_dist_session_save_payload()`
  - Gathers local and remote shards into one normal `DSV4` payload.
- `ds4_dist_session_load_payload()`
  - Splits one normal `DSV4` payload over the current route.
- `ds4_dist_run()`
  - Standalone coordinator/worker role runner.

Coordinator route and prefill functions:

- `dist_coordinator_build_route_plan()`
  - Builds contiguous route from coordinator local layer `0` through workers to final layer/output.
- `dist_route_search_workers()`
  - Recursive compatible worker route search.
- `dist_coordinator_ensure_route()`
  - Ensures current route is usable.
- `dist_coordinator_prefill_chunk_cap()`
  - Chooses prefill chunk size.
- `dist_coordinator_prefill_window()`
  - Chooses max in-flight prefill chunks.
- `dist_coordinator_prefill_prompt()`
  - Prefill dispatcher.
- `dist_coordinator_prefill_prompt_pipelined()`
  - Overlaps coordinator local slice for chunk N+1 with remote processing of chunk N.
- `dist_coordinator_eval_span()`
  - Common coordinator evaluator for prefill chunks and decode tokens.
- `dist_coordinator_send_remote_work_on_fd()`
  - Builds/sends `WORK` frames with tokens, hashes, hidden state, route blob.
- `dist_coordinator_eval_remote_on_fd()`
  - Sends work, receives result, validates hash, copies logits or runs local output head.
- `dist_recv_result_alloc()`
  - Reads `RESULT`, telemetry, payload, and decodes compressed activation transport.
- `dist_coordinator_rebuild_from_transcript()`
  - Replays token history after route failures/reconnects.

Coordinator prefill flow:

- `ds4_dist_session_sync()` ensures route.
- If current checkpoint is a prefix, only suffix is evaluated; otherwise full prompt is rebuilt.
- Pipelined prefill is used when multiple chunks and remote workers are available.
- Coordinator evaluates local slice with `ds4_session_eval_layer_slice()`.
- Sender thread sends hidden state chunks to first worker.
- Reader thread receives ACK/logits/hidden state results.
- Intermediate chunks can be ACK-only.
- Final chunk returns logits, or hidden state if coordinator runs local output head with `ds4_session_eval_output_head_from_hc()`.
- Serial fallback loops chunks through `dist_coordinator_eval_span()`.

Coordinator decode flow:

- `ds4_dist_session_eval()` handles one generated token.
- It calls `dist_coordinator_eval_span()` with `n_tokens = 1`.
- Decode is strictly autoregressive: next token cannot start until logits return and sampling chooses the token.
- Distributed decode is expected to be slower because every token traverses the route.

Distributed save/load flow:

- Save:
  - `ds4_dist_session_save_payload()`
  - `dist_kv_route_validate()`
  - local shard via `ds4_session_save_layer_payload()`
  - remote shards via `dist_save_remote_shard_to_file()`
  - parse shards with `dist_kv_parse_layer_payload()`
  - write one merged topology-neutral `DSV4`.
- Load:
  - `ds4_dist_session_load_payload()`
  - read normal `DSV4`
  - split by route with `dist_prepare_shard_from_session_payload()`
  - local restore via `ds4_session_load_layer_payload()`
  - remote restore via `dist_load_remote_shard_from_payload()`
  - restore logits with `ds4_session_set_logits()`.

Worker control flow:

- `dist_run_worker()`
  - Starts data listener, initializes `ds4_dist_worker_state`, connects/reconnects to coordinator, sends `HELLO`.
- `dist_worker_data_listener_main()`
  - Accepts coordinator or upstream-worker data connections.
- `dist_worker_read_loop()` / `dist_worker_read_loop_prefetch()`
  - Handles frames, optionally with worker prefetch queue.
- `dist_worker_handle_work()`
  - Reads `WORK` frame payload.
- `dist_worker_process_work_payload()`
  - Validates model, layer range, route blob, token ids, token hash, hidden-state size, output-head legality, context bounds.
  - Executes assigned layer range using `ds4_session_eval_layer_slice()`.
  - Updates worker session token hash after successful eval.
  - Forwards to next worker or sends final `RESULT`.
- `dist_forward_work_to_next()`
  - Middle-worker forwarding path.
- `dist_worker_handle_snapshot_save()`
  - Serializes worker-owned layer payload.
- `dist_worker_handle_snapshot_load()`
  - Restores worker-owned layer payload.
- On coordinator disconnect:
  - worker clears sessions and reconnects.

Protocol concepts:

- Frame header:
  - `DS4_DIST_MAGIC`, message type, byte count.
- Message types:
  - `DS4_DIST_MSG_HELLO`
  - `DS4_DIST_MSG_ERROR`
  - `DS4_DIST_MSG_WORK`
  - `DS4_DIST_MSG_RESULT`
  - `DS4_DIST_MSG_SNAPSHOT_SAVE_REQ`
  - `DS4_DIST_MSG_SNAPSHOT_BEGIN`
  - `DS4_DIST_MSG_SNAPSHOT_CHUNK`
  - `DS4_DIST_MSG_SNAPSHOT_DONE`
  - `DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN`
- Work flags:
  - `DS4_DIST_WORK_F_INPUT_HC`
  - `DS4_DIST_WORK_F_OUTPUT_LOGITS`
  - `DS4_DIST_WORK_F_RESET_SESSION`
  - `DS4_DIST_WORK_F_ACK_ONLY`
- Result kinds:
  - `DS4_DIST_RESULT_ACK`
  - `DS4_DIST_RESULT_HIDDEN_STATE`
  - `DS4_DIST_RESULT_LOGITS`
- Route blob:
  - Serialized route entries plus return target.
  - Allows middle workers to forward without coordinator relay.
- Telemetry:
  - Layer range, route index, token span, eval time, downstream wait, forward-send time, input bytes, output bytes.

Current distributed assumptions:

- Coordinator route starts at layer `0`.
- Worker slices are contiguous and cover later layers through final layer.
- Final route either owns output head or returns hidden state for coordinator output-head evaluation.
- Each worker owns its slice of KV cache.
- Coordinator/workers must match model id, layer count, context compatibility, quant profile, and route metadata.
- Rolling token hash guards against stale worker KV.
- Prefill can pipeline; decode cannot.
- Persistent distributed KV save/load is topology-neutral `DSV4`, but APIs are file-oriented.
- Protocol has no authentication/encryption and is not release-stable.

Issue #304 starting points:

- Start at `ds4_dist_session_sync()`, because distributed prefill completes there.
- Use `ds4_dist_session_save_payload()` as the reference implementation for gathering remote KV into one normal `DSV4`.
- Use `dist_worker_handle_snapshot_save()` as the current worker-side "return my KV shard" primitive.
- Use `dist_kv_parse_layer_payload()` and `dist_kv_layout` fields to validate shard compatibility.
- Validate handoff by comparing:
  - distributed prefill + distributed decode,
  - distributed prefill + merged `DSV4` load into local session,
  - single-node local prefill/decode when feasible.

## Backend Portability, Graph/Tensor API, And Validation

Relevant files:

- `ds4_gpu.h`
  - Backend-neutral GPU tensor and kernel API.
- `ds4_metal.m`
  - Metal implementation.
- `ds4_cuda.cu`
  - CUDA implementation.
- `ds4.c`
  - Graph/session owner and payload serializer.
- `tests/ds4_test.c`
  - Main integration/regression harness.
- `tests/cuda_long_context_smoke.c`
  - CUDA-only smoke tests.
- `Makefile`
  - Build/test entry points.

Key tensor APIs:

- `ds4_gpu_init()` / `ds4_gpu_cleanup()`
  - Backend lifecycle.
- `ds4_gpu_set_model_map_range()`
  - Full contiguous model tensor mapping.
- `ds4_gpu_set_model_map_spans()`
  - Sliced/non-contiguous model tensor mapping.
- `ds4_gpu_cache_model_range()`
  - Optional model tensor cache.
- `ds4_gpu_preload_q4_expert_tables()`
  - PRO Q4 expert-table preload.
- `ds4_gpu_tensor_alloc()` / `ds4_gpu_tensor_alloc_managed()`
  - Backend tensor allocation.
- `ds4_gpu_tensor_view()`
  - Subviews of backend tensors.
- `ds4_gpu_tensor_read()` / `ds4_gpu_tensor_write()`
  - Primary backend-neutral serialization boundary.
- `ds4_gpu_tensor_copy()`
  - Backend-local tensor copy.
- `ds4_gpu_synchronize()`
  - Required before snapshot reads and after restores.
- `ds4_gpu_store_raw_kv_tensor()` / `ds4_gpu_store_raw_kv_batch_tensor()`
  - Raw sliding-window KV writes.
- `ds4_gpu_kv_fp8_store_raw_tensor()`
  - Decode fused KV finalizer and raw cache write.
- `ds4_gpu_compressor_store_batch_tensor()`
  - Rolling compressor frontier update.
- `ds4_gpu_compressor_prefill_tensor()`
  - Compressed KV rows/frontier for prefill spans.
- `ds4_gpu_compressor_prefill_ratio4_replay_tensor()`
  - Ratio-4 replay path.
- `ds4_gpu_compressor_prefill_state_ratio4_tensor()`
  - Ratio-4 frontier reconstruction.
- Attention consumers:
  - `ds4_gpu_attention_decode_heads_tensor()`
  - `ds4_gpu_attention_decode_raw_batch_heads_tensor()`
  - `ds4_gpu_attention_decode_mixed_batch_heads_tensor()`
  - `ds4_gpu_attention_indexed_mixed_batch_heads_tensor()`
  - `ds4_gpu_attention_prefill_raw_heads_tensor()`
  - `ds4_gpu_attention_prefill_static_mixed_heads_tensor()`
  - `ds4_gpu_attention_prefill_masked_mixed_heads_tensor()`

Metal/CUDA responsibilities:

- Metal:
  - `ds4_metal.m`
  - `ds4_gpu_tensor` backed by `MTLBuffer` plus offsets/views.
  - `ds4_gpu_tensor_read()` / `ds4_gpu_tensor_write()` use host `memcpy` through shared buffer contents.
- CUDA:
  - `ds4_cuda.cu`
  - `ds4_gpu_tensor` backed by CUDA device pointers.
  - `ds4_gpu_tensor_read()` / `ds4_gpu_tensor_write()` use `cudaMemcpy`.
- Shared assumption:
  - graph/session code should interact only through `ds4_gpu.h`, not backend-native handles.

Serialization portability risks:

- Full payloads and layer payloads stream tensor bytes through `ds4_gpu_tensor_read()` / `ds4_gpu_tensor_write()`.
- Metal can store compressed attention KV as F16 internally when `DS4_GPU_ATTN_COMP_CACHE_F16` is enabled.
- CUDA compressed attention KV is F32 from the serializer's point of view.
- Cross-backend checks should be semantic, not byte-exact.
- Expected drift sources:
  - F16 compressed-cache storage,
  - FP8 KV round-trip,
  - RoPE,
  - compressor pooling,
  - backend math differences.
- Raw KV rows are serialized in logical token order and restored into destination physical ring using `pos % raw_cap`.
- Ratio-4 compressed layers require frontier/indexer state, not just completed compressed rows:
  - `index_comp_cache`,
  - `index_state_kv`,
  - `index_state_score`,
  - attention compressor state.
- Incremental KV return must preserve `ds4_gpu_synchronize()` safety currently provided by snapshot export/restore paths.

Validation commands:

- `make test`
- `make cuda-regression`
- `./ds4_test --metal-short-prefill`
- `./ds4_test --metal-kernels`
- `./ds4_test --metal-tensor-equivalence`
- `./ds4_test --logprob-vectors`
- `./ds4_test --local-golden-vectors`
- `./ds4_test --long-context`

Issue #304 validation additions:

- Add local full-payload save/load resume tests.
- Add representative `DSVL` layer-shard save/load tests:
  - uncompressed layers,
  - compressed layers,
  - ratio-4 layers,
  - final/output-adjacent layers.
- Add distributed gather/load smoke:
  - coordinator + worker prefill,
  - save merged `DSV4`,
  - load into non-distributed full local session,
  - compare logits and greedy tokens.
- Add Phase 3 official/local parity checks:
  - run official-vector prompts through fully local inference on the decode backend,
  - run the same prompts through distributed prefill -> merged `DSV4` -> local decode on that backend,
  - compare distributed handoff against fully local before treating cross-engine drift as a defect,
  - compare selected tokens/top-logprobs against `tests/test-vectors/official.vec`,
  - compare local full-logit behavior against `tests/test-vectors/local-golden.vec` where available.
- Add cross-backend manual smoke where hardware allows:
  - CUDA worker/shard -> Metal local load/decode,
  - Metal shard -> CUDA local load/decode.
- Compare:
  - top1 match,
  - top5/top20 overlap,
  - max absolute logit drift,
  - RMS logit drift,
  - first 16 greedy tokens.
- Treat bit-exact cross-engine logits as diagnostic evidence, not a Phase 3 requirement, unless they expose a same-backend handoff or official/local-vector failure.

## User-Facing Workflow And Documentation

Relevant files:

- `README.md`
  - Distributed setup, current `--layers` meaning, snapshot persistence format, limitations.
- `ds4_cli.c`
  - CLI command/flag integration.
- `ds4_distributed.c`
  - Distributed CLI parsing and help for role/layers/listen/coordinator/prefill options.
- `ds4_server.c`
  - Server sessions and persistent KV behavior.
- `ds4_web.c` / `ds4_web.h`
  - Web/server support utilities.
- `ds4_help.c` / `ds4_help.h`
  - Help text utilities.

Current distributed CLI concepts:

- `--role coordinator`
  - Coordinator owns prompt/session/sampling/control plane and currently owns initial layer range.
- `--role worker`
  - Worker connects to coordinator and owns a later layer slice.
- `--layers A:B`
  - Inclusive distributed layer slice.
  - Currently also affects which tensors are mapped.
- `--layers N:output`
  - Worker owns final transformer layers plus output head.
- `--layers N:42`
  - Worker owns later transformer layers only; coordinator computes output head/logits.
- `--listen HOST PORT`
  - Coordinator control listener.
- `--coordinator HOST PORT`
  - Worker connects to coordinator.
- `--dist-prefill-chunk`
  - Coordinator prefill chunk size.
- `--dist-prefill-window`
  - Coordinator in-flight prefill chunks.
- `--dist-activation-bits`
  - Activation transport compression width.

Documentation areas likely affected by issue #304:

- `README.md` distributed section:
  - explain local generation after distributed prefill,
  - document any new residency/loading option,
  - document whether `--layers` still maps tensors or only describes route ownership,
  - document output-head/logit ownership,
  - document KV return cost and limitations.
- CLI help:
  - new flags or changed `--layers` semantics.
- Server docs:
  - persistent sessions and local-decode handoff behavior if server uses it.

## Tests, Build, And Validation Map

Primary commands:

- `make`
  - default build.
- `make test`
  - main regression suite.
- `make cuda-regression`
  - CUDA long-context smoke.

Useful test binary modes:

- `./ds4_test --metal-short-prefill`
- `./ds4_test --metal-kernels`
- `./ds4_test --metal-tensor-equivalence`
- `./ds4_test --logprob-vectors`
- `./ds4_test --local-golden-vectors`
- `./ds4_test --long-context`

Relevant test files:

- `tests/ds4_test.c`
  - Local/Metal/session regression surface.
  - Best place for focused payload save/load and layer-shard tests.
- `tests/test-vectors/official.vec`
  - Official API top-logprob vector fixture used as the Phase 3 selected-token/top-logprob acceptance surface.
- `tests/test-vectors/local-golden.vec`
  - Local top-k/logit fixture used to catch substantial local backend drift.
- `tests/cuda_long_context_smoke.c`
  - CUDA-specific kernel smoke surface.
- `tests/long_context_story_prompt.txt`
  - Long-context prompt fixture.
- `tests/long_context_security_prompt.txt`
  - Long-context security prompt fixture.
- `tests/test_q4k_dot.c`
  - Quant/kernel test surface.

Issue #304 expected validation artifacts:

- `artifacts/issue-304/compatibility-matrix.md`
- `artifacts/issue-304/logit-comparisons.md`
- `artifacts/issue-304/shard-metadata.md`
- `artifacts/issue-304/perf-breakdown.md`
- `artifacts/issue-304/decision-log.md`
- `artifacts/issue-304/runbook.md`
- `artifacts/issue-304/failure-cases.md`
- `artifacts/issue-304/engine-residency.md`
- `artifacts/issue-304/topology-decoupling.md`

## Issue #304 Starting Guide

Immediate code reading order:

- `PLAN.md`
  - Current research/implementation plan and phase gates.
- `ds4.h`
  - API and payload boundaries.
- `ds4.c`
  - Engine residency, session lifecycle, payload formats, layer-slice execution.
- `ds4_distributed.h`
  - Distributed backend API.
- `ds4_distributed.c`
  - Distributed prefill/decode/save/load orchestration.
- `ds4_gpu.h`
  - Tensor serialization/backend abstraction.
- `ds4_metal.m` / `ds4_cuda.cu`
  - Inspect only when backend portability or mapping behavior is in question.
- `tests/ds4_test.c`
  - Add focused correctness tests here first where possible.

Minimal correctness proof:

- Run distributed prefill with current coordinator/worker topology.
- Stage or save distributed coordinator payload:
  - `ds4_session_stage_payload()`, or
  - `ds4_session_save_payload()`.
- Load merged `DSV4` into a full local non-distributed graph session:
  - `ds4_session_load_payload()`.
- Decode locally for 1-3 tokens.
- Compare against current distributed decode and single-node local decode where feasible.
- For Phase 3, run the official-vector prompt set through the same sequence and require same-backend distributed handoff parity with fully local inference before classifying cross-engine forced-logit drift as acceptable backend variance.

Known blockers to analyze before user-facing implementation:

- Distributed coordinator currently often maps only its `--layers` slice.
- Local decode needs full model weights plus full KV state.
- `ds4_session_save_snapshot()` rejects distributed sessions.
- `ds4_session_payload_bytes()` returns `0` for distributed sessions.
- Existing distributed save/load is file-oriented.
- Current route model assumes coordinator-first layer execution.
- Output/logit production is coupled to `N:output` versus `N:42`.

Primary implementation decision points:

- Whole `DSV4` handoff versus in-memory payload versus explicit `DSVL` shard merge.
- One full-resident engine with sliced distributed execution versus dynamic residency expansion.
- Whether route ownership and weight residency must be decoupled before the first production implementation.
- Whether topology decoupling is required now or can be deferred after the first safe local-generation handoff.

Failure cases to preserve:

- Token hash mismatch.
- Layer range mismatch.
- Context/raw-window/compression-cap mismatch.
- Model id or quant profile mismatch.
- Missing output head when logits are requested.
- Worker reconnect before shard fetch.
- Stale worker session after route rebuild.
- Cross-backend payload load drift that changes same-backend handoff behavior or fails official/local-vector tolerance.

## Search Tips

- Find public API declarations:
  - `rg -n "ds4_session_|ds4_engine_|ds4_dist_" ds4.h ds4_distributed.h`
- Find distributed protocol messages:
  - `rg -n "DS4_DIST_MSG|DS4_DIST_WORK|DS4_DIST_RESULT" ds4_distributed.c`
- Find distributed prefill path:
  - `rg -n "prefill_prompt|prefill_sender|prefill_result" ds4_distributed.c`
- Find distributed save/load path:
  - `rg -n "save_payload|load_payload|snapshot|dist_kv" ds4_distributed.c ds4.c`
- Find model residency path:
  - `rg -n "load_layer|model_map_spans|set_model_map|weights_layers_bound" ds4.c ds4_gpu.h ds4_metal.m ds4_cuda.cu`
- Find tensor serialization boundary:
  - `rg -n "payload_write_tensor_span|payload_read_tensor_span|tensor_read|tensor_write" ds4.c ds4_gpu.h ds4_metal.m ds4_cuda.cu`
- Find CLI distributed parsing/help:
  - `rg -n "role|layers|dist-prefill|coordinator|listen" ds4_cli.c ds4_distributed.c`
