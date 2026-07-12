# DS4 env-gated instrumentation and variation inventory

Date: 2026-07-12. Scope: runtime environment variables read by `ds4.c` in the
main DS4 implementation. This inventory is meant to track two kinds of switches:

1. **instrumentation / diagnostics** — profiling, tracing, dumps, debug replay,
   logging, and memory reporting; and
2. **env-gated execution variations** — alternative speculative paths, batching,
   streaming/paging policy, fusion toggles, and backend behavior changes.

Out of scope for the main table:

- test-only `DS4_TEST_*` variables from `tests/ds4_test.c`;
- harness-local env handling that only records whether a runtime flag was set,
  such as `ds4_spec_bench.c` reading `DS4_DSPARK_TIMING`; and
- compile-time macros that are not read from the environment.

This file should be updated whenever a new env gate is added, removed, or its
meaning materially changes.

## General runtime and CPU/prefill shaping

| Variable | Class | Effect |
|---|---|---|
| `DS4_THREADS` | variation | Overrides the CPU worker-thread count. |
| `DS4_PREFILL_BATCH` | variation | Sets CPU prefill batch size. |
| `DS4_BATCHED_FFN` | variation | Enables batched FFN handling in CPU prefill. |
| `DS4_PARALLEL_FFN` | variation | Enables parallel FFN execution in CPU prefill. |
| `DS4_NO_SHARED_BATCH_FFN` | variation | Disables shared-batch FFN path. |
| `DS4_NO_BATCHED_ATTN` | variation | Disables batched attention path. |
| `DS4_PARALLEL_ATTN_ROWS` | variation | Forces parallel attention-row handling. |
| `DS4_NO_PARALLEL_ATTN_ROWS` | variation | Disables automatic parallel attention rows. |
| `DS4_BATCHED_ROPE_MAX` | variation | Caps the row count where batched RoPE is used. |
| `DS4_NO_BATCHED_ROPE` | variation | Disables batched RoPE path. |
| `DS4_ROUTED_TOKEN_PARALLEL` | variation | Forces routed-token parallel execution. |
| `DS4_NO_ROUTED_TOKEN_PARALLEL` | variation | Disables routed-token parallel execution. |
| `DS4_METAL_PREFILL_CHUNK` | variation | Overrides Metal prompt/prefill chunk size. |
| `DS4_METAL_RESUME_PREFILL_MIN` | variation | Sets the minimum prompt remainder before resuming prefill work. |
| `DS4_METAL_GPU_BATCH_EMBED_MIN` | variation | Sets the token-count threshold for GPU batch embedding. |
| `DS4_METAL_GRAPH_RAW_CAP` | variation | Overrides raw-cache capture capacity in Metal graph diagnostics. |
| `DS4_METAL_GRAPH_PROMPT_TOKENS` | diagnostic | Overrides prompt length used by the Metal prompt-graph diagnostic path. |
| `DS4_METAL_NO_PREFILL_KERNEL_WARMUP` | variation | Skips the prefill-kernel warmup path. |
| `DS4_METAL_STREAMING_DECODE_PREFILL_MAX` | variation | Caps streamed decode-prefill reuse. |
| `DS4_LOCK_FILE` | variation | Overrides the DS4 lock-file path used for single-engine memory coordination. |

## Generic runtime instrumentation

| Variable | Class | Effect |
|---|---|---|
| `DS4_TRACE_TOP` | diagnostic | Logs top-token information during generation. |
| `DS4_TOKEN_TIMING` | diagnostic | Emits per-token timing information during generation. |
| `DS4_DECODE_PROFILE_DETAIL` | diagnostic | Enables detailed CPU decode-stage profiling. |
| `DS4_PREFILL_PROFILE_DETAIL` | diagnostic | Enables detailed CPU prefill profiling. |
| `DS4_PREFILL_PROFILE_TOKEN` | diagnostic | Enables token-level prefill profiling output. |
| `DS4_CPU_DUMP_LOGITS` | diagnostic | Dumps CPU logits in the Metal prompt-graph comparison path. |
| `DS4_CPU_DUMP_PREFILL_LOGITS` | diagnostic | Dumps CPU prefill logits from the CPU runtime path. |
| `DS4_ORACLE_LOGITS` | diagnostic | Loads external oracle logits for prompt-graph comparison. |
| `DS4_EXPERT_PROFILE` | diagnostic | Loads expert profile data for routing/streaming decisions. |
| `DS4_EXPERT_HOTLIST` | variation | Loads an expert hotlist used by runtime preload/streaming heuristics. |
| `DS4_MOE_REPLAY_SELECTED_IDS` | diagnostic | Replays routed expert ids instead of using live router selection. |

## Metal graph dump, trace, and prompt-graph diagnostics

| Variable | Class | Effect |
|---|---|---|
| `DS4_METAL_GRAPH_DUMP_PREFIX` | diagnostic | Enables tensor dumps under the given output prefix. |
| `DS4_METAL_GRAPH_DUMP_NAME` | diagnostic | Filters graph dumps by tensor/debug name. |
| `DS4_METAL_GRAPH_DUMP_LAYER` | diagnostic | Filters graph dumps by layer. |
| `DS4_METAL_GRAPH_DUMP_POS` | diagnostic | Filters graph dumps by token position. |
| `DS4_METAL_GRAPH_DUMP_LOGITS` | diagnostic | Dumps Metal prompt-graph logits for comparison/debug. |
| `DS4_METAL_GRAPH_TRACE_LAYERS` | diagnostic | Enables full-layer trace output in the Metal full-graph diagnostic. |
| `DS4_METAL_GRAPH_TRACE_STAGE_LAYER` | diagnostic | Limits stage tracing to a chosen layer. |
| `DS4_METAL_GRAPH_TEACHER_FORCE` | diagnostic | Uses teacher forcing inside the Metal full-graph trace path. |
| `DS4_METAL_GRAPH_TRACE_CACHE` | diagnostic | Dumps raw-cache trace diagnostics in prompt-graph comparison. |
| `DS4_METAL_GRAPH_TRACE_COMP` | diagnostic | Dumps compressed-cache trace diagnostics in prompt-graph comparison. |
| `DS4_METAL_GRAPH_TOKEN_PROFILE` | diagnostic | Enables single-token Metal graph timing/profile output. |
| `DS4_METAL_GRAPH_PREFILL_PROFILE` | diagnostic | Enables prompt/prefill Metal graph timing output. |
| `DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE` | diagnostic | Enables split-prefill profiling output. |
| `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS` | variation | Splits token graph execution at selected layers for profiling/experiments. |
| `DS4_METAL_GRAPH_OUTPUT_ROW` | diagnostic | Selects which row is materialized in prompt/prefill graph output debugging. |
| `DS4_METAL_MEMORY_REPORT` | diagnostic | Emits Metal memory-map / allocation reporting. |

## Metal stage-level profiling

| Variable | Class | Effect |
|---|---|---|
| `DS4_METAL_LAYER_STAGE_PROFILE` | diagnostic | Enables per-layer stage timing in Metal decode/prefill. |
| `DS4_METAL_DECODE_STAGE_PROFILE` | diagnostic | Enables decode-stage timing in Metal decode. |
| `DS4_METAL_INDEXER_STAGE_PROFILE` | diagnostic | Enables indexer-stage timing. |
| `DS4_METAL_Q_STAGE_PROFILE` | diagnostic | Enables Q-path stage timing. |
| `DS4_METAL_PRO_Q4_CPU_ROUTER_PROFILE` | diagnostic | Profiles the Pro-Q4 CPU-router fallback path. |
| `DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE` | diagnostic | Profiles the streaming IQ2 CPU-router path. |
| `DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE` | diagnostic | Profiles selected-expert readahead behavior. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_PROFILE` | diagnostic | Profiles prefill selected-expert preparation. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE` | diagnostic | Profiles selected-expert page-in behavior. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE` | diagnostic | Profiles selected-expert madvise behavior. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_PROFILE` | diagnostic | Profiles selected-expert prefill readahead behavior. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE` | diagnostic | Profiles per-layer prefill page-in behavior. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE` | diagnostic | Profiles per-layer prefill pread behavior. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_MADVISE_PROFILE` | diagnostic | Profiles per-layer prefill madvise behavior. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE` | diagnostic | Profiles per-layer prefill readahead behavior. |
| `DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE` | diagnostic | Profiles cache-seed preparation during streamed prefill. |
| `DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE` | diagnostic | Profiles expert-hotlist generation/use. |

## Metal streaming, paging, and address-table variations

| Variable | Class | Effect |
|---|---|---|
| `DS4_METAL_ENABLE_STREAMING_READAHEAD` | variation | Enables decode-time file readahead for streamed weights. |
| `DS4_METAL_DISABLE_STREAMING_READAHEAD` | variation | Disables decode-time file readahead. |
| `DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED` | variation | Enables decode-time `madvise(...WILLNEED)` for streamed weights. |
| `DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED` | variation | Disables decode-time `madvise(...WILLNEED)`. |
| `DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP` | variation | Disables the static streamed decode map. |
| `DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE` | variation | Disables cached state for static streamed maps. |
| `DS4_METAL_DISABLE_STREAMING_LAYER_BATCH` | variation | Disables batched layer submission in the streamed decode path. |
| `DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE` | variation | Forces use of the full expert address-table path. |
| `DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE` | variation | Disables the full expert address-table path. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN` | variation | Enables selected-expert page-in during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN` | variation | Disables selected-expert page-in during prefill. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE` | variation | Enables selected-expert madvise during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE` | variation | Disables selected-expert madvise during prefill. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN` | variation | Enables per-layer page-in during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN` | variation | Disables per-layer page-in during prefill. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD` | variation | Enables per-layer readahead during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD` | variation | Disables per-layer readahead during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD` | variation | Disables explicit pread preparation during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE` | variation | Disables per-layer prefill prepare work. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE` | variation | Disables per-layer prefill madvise. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR` | variation | Forces the batched selected-expert address path in prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR` | variation | Disables the batched selected-expert address path in prefill. |
| `DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR` | variation | Disables the CUDA analogue of batched selected-expert addressing. |
| `DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN` | variation | Lower token bound for using batched selected-expert addressing. |
| `DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX` | variation | Upper token bound for using batched selected-expert addressing. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD` | variation | Enables selected-expert readahead during prefill. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD` | variation | Disables selected-expert readahead during prefill. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED` | variation | Enables shared-file selected-expert prefill readahead. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED` | variation | Disables shared-file selected-expert prefill readahead. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_GAP` | variation | Sets the selected-expert prefill readahead gap. |
| `DS4_METAL_ENABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY` | variation | Enables delayed shared selected-expert readahead. |
| `DS4_METAL_DISABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY` | variation | Disables delayed shared selected-expert readahead. |
| `DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP` | variation | Disables overlap between selected-expert and shared loads. |
| `DS4_CUDA_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP` | variation | Disables CUDA selected/shared overlap path. |
| `DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD` | variation | Disables asynchronous selected-expert loading. |
| `DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_EARLY_COMMIT` | variation | Disables early commit of async selected-expert loads. |
| `DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL` | variation | Disables decode-time prefill reuse. |
| `DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL` | variation | Disables cold-start decode-prefill reuse. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS` | variation | Thread count for layer-prepare prefill helpers. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_THREADS` | variation | Thread count for layer page-in helpers. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_THREADS` | variation | Thread count for selected-expert prepare helpers. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_THREADS` | variation | Thread count for selected-expert madvise helpers. |
| `DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_GAP` | variation | Gap between selected-expert prepare work and use. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_NO_OVERLAP` | variation | Disables overlap for layer-prepare prefill work. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_NO_OVERLAP` | variation | Disables overlap for layer page-in prefill work. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE_OVERLAP` | variation | Disables the explicit prefill layer-prepare overlap path. |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN_OVERLAP` | variation | Disables the explicit prefill layer page-in overlap path. |
| `DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_AHEAD` | variation | How far ahead to prepare per-layer prefill resources. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED` | variation | Enables cache seeding for streamed prefill. |
| `DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K` | variation | Sets the cache-seed depth/count. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST` | variation | Disables runtime-generated expert hotlists. |
| `DS4_METAL_STREAMING_EXPERT_HOTLIST` | variation | Loads a persisted expert hotlist path. |
| `DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP` | variation | Caps auto-preloaded experts in streamed execution. |
| `DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD` | variation | Disables CUDA prefill batched selected-weight loads. |
| `DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE` | diagnostic | Profiles CUDA prefill batched selected-weight loads. |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE` | diagnostic | Profiles CUDA streamed expert caching. |
| `DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB` | variation | Sets CUDA direct-model preload span. |
| `DS4_CUDA_DIRECT_MODEL` | variation | Enables CUDA direct-model mapping path. |

## Metal fusion, routing, and expert-table variations

| Variable | Class | Effect |
|---|---|---|
| `DS4_METAL_Q4_PRO_MAP_GROUPS` | variation | Changes how Pro-Q4 model views are grouped/mapped. |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` | variation | Sets sparsity threshold for decode indexer path selection. |
| `DS4_METAL_HC_NORM_FUSION_CHECK_TOL` | diagnostic | Tolerance used by HC-norm fusion validation checks. |
| `DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER` | variation | Enables CPU-router fallback for streamed IQ2 routing. |
| `DS4_METAL_DISABLE_STREAMING_IQ2_CPU_ROUTER` | variation | Disables CPU-router fallback for streamed IQ2 routing. |
| `DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS` | variation | Enables Q4 selected-expert views. |
| `DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS` | variation | Enables Pro-Q4 selected-expert views. |
| `DS4_METAL_ENABLE_Q4_EXPERT_TABLE` | variation | Enables Q4 expert-table path. |
| `DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE` | variation | Enables Q4 expert-address-table path. |
| `DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO` | variation | Enables auto-selected Pro-Q4 expert-table path. |
| `DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO` | variation | Enables auto-selected Pro-Q4 expert-address path. |
| `DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO` | variation | Disables auto-selected Pro-Q4 expert-table path. |
| `DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_PRELOAD` | variation | Disables Pro-Q4 expert-table preload path. |
| `DS4_METAL_DISABLE_Q4_EXPERT_TABLE` | variation | Disables Q4 expert-table path. |
| `DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS` | variation | Disables Q4 selected-expert views. |
| `DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS` | variation | Disables IQ2 selected-expert views. |
| `DS4_METAL_DISABLE_IQ2_SELECTED_SHARED_OVERLAP` | variation | Disables IQ2 selected/shared overlap. |
| `DS4_METAL_MOE_WRITE_CLAMPED_ACT` | variation | Changes routed-activation handling to clamped-write behavior. |
| `DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION` | variation | Disables routed-pair SwiGLU fusion. |
| `DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION` | variation | Disables shared gate/up SwiGLU fusion. |

## Metal dump paths tied to generation outputs

| Variable | Class | Effect |
|---|---|---|
| `DS4_METAL_DUMP_PREFILL_LOGITS` | diagnostic | Dumps Metal prefill logits from the runtime generation path. |

## MTP speculative runtime

| Variable | Class | Effect |
|---|---|---|
| `DS4_MTP_ANCHOR_REUSE` | variation | Enables the experimental anchor-reuse verifier path. |
| `DS4_MTP_BATCH_VERIFY` | variation | Enables the batched verify path where the strict K=2 shortcut would otherwise stay sequential. |
| `DS4_MTP_STRICT` | variation | Forces strict MTP gating behavior even outside quality mode. |
| `DS4_MTP_MIN_MARGIN` | variation | Sets the minimum confidence margin for draft acceptance decisions. |
| `DS4_MTP_PROBE` | diagnostic | Logs MTP draft/probe behavior without full speculative logging. |
| `DS4_MTP_SPEC_LOG` | diagnostic | Enables detailed MTP speculative-cycle logging. |
| `DS4_MTP_TIMING` | diagnostic | Enables MTP timing output. |
| `DS4_MTP_CONF_LOG` | diagnostic | Logs MTP confidence values. |
| `DS4_MTP_FULL_LOGITS` | diagnostic | Retains/uses full MTP draft logits for debug and confidence reporting. |
| `DS4_MTP_VERIFY_PROFILE` | diagnostic | Profiles MTP verifier execution. |
| `DS4_MTP_VERIFY_EXPERT_PROFILE` | diagnostic | Profiles expert-level MTP verifier behavior. |
| `DS4_MTP_CAPTURE_PREFIX1` | diagnostic | Forces prefix-1 capture in exact replay/debug paths. |
| `DS4_MTP_EXACT_REPLAY` | diagnostic | Enables exact replay debugging for MTP verifier work. |
| `DS4_MTP_FORCE_SNAPSHOT` | diagnostic | Forces snapshot-based state handling in replay/debug flows. |

## DSpark speculative runtime

| Variable | Class | Effect |
|---|---|---|
| `DS4_DSPARK_VERIFY_K` | variation | Forces a fixed verify length and disables adaptive scheduling when set to a non-negative value. |
| `DS4_DSPARK_CONF_SCHEDULE` | variation | Enables or disables confidence-scheduled verification. |
| `DS4_DSPARK_CONF_THRESHOLD` | variation | Sets the survival threshold used by DSpark scheduled verification. |
| `DS4_DSPARK_SCHEDULE_BATCHED` | variation | Enables the experimental batched scheduled-drafter path. |
| `DS4_DSPARK_SCHEDULE_BATCH_N` | variation | Caps how many draft rows the batched scheduler computes. |
| `DS4_DSPARK_SCHEDULE_GPU_HEAD` | variation | Enables the experimental GPU-assisted scheduled output-head path. |
| `DS4_DSPARK_SPEC_LOG` | diagnostic | Enables DSpark speculative-cycle logging. |
| `DS4_DSPARK_TIMING` | diagnostic | Enables per-cycle DSpark timing capture. |
| `DS4_DSPARK_VERIFY_DIST_PROBE` | diagnostic | Measurement 1 (option A): non-committing probe that runs the sublinear batched verifier (`verify_suffix_tops`) alongside the real sequential DSpark verify and records per-position argmax-flip / max-abs-logit / TV / KL(seq‖batched) into `dspark_last_cycle.verify_dist`, emitted per-cycle as `verify_dist` objects in the ds4-spec-bench JSONL. Requires the graph to allocate `spec_logits` + spec-frontier tensors for DSpark (now done). Non-mutating to the real decode. |

## Inventory notes for current research

- The milestone-2 DSpark migration work currently depends most directly on the
  `DS4_DSPARK_*` family, `DS4_MTP_ANCHOR_REUSE`, and the `DS4_METAL_*`
  graph/streaming diagnostics.
- `DS4_DSPARK_SCHEDULE_BATCHED`, `DS4_DSPARK_SCHEDULE_BATCH_N`, and
  `DS4_DSPARK_SCHEDULE_GPU_HEAD` are still experimental execution-shape knobs
  rather than validated production defaults.
- The largest instrumentation surface in `ds4.c` today is the Metal streaming
  and graph-debug stack. That surface should be treated as part of the research
  harness, not only as production runtime configuration.
