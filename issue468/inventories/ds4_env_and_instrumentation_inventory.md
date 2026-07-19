# DS4 env-gated instrumentation and variation inventory

Date: 2026-07-14 (updated 2026-07-19 for the M3 default-on ungate + the missing levers 3–5). Scope: runtime environment variables read by `ds4.c` in the
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
| `DS4_DSPARK_VERIFY_BATCHED` | variation | **M3 lever 1.** Committing sublinear batched verify: the partial-accept commit uses the batched `verify_suffix_tops` rows + a captured-hidden push instead of the exact sequential replay. Divergent-but-score-neutral (NOT byte-exact; temp>0 TV~0.0104, net +4 on 92Q). **M3 default-on (2026-07-19)** whenever `--dspark` is loaded; opt-out via `DS4_DSPARK_VERIFY_BATCHED=0\|off`. |
| `DS4_DSPARK_ANCHOR_REUSE` | variation | **M3 lever 2.** When `first_token` is the greedy argmax of the current logits + batched verify is on, skip the standalone anchor decode + fold `first_token` into the batched verify as `drafts[0]` (continuation drafted from the stale DSpark window). **M3 default-on**; opt-out via `=0\|off`. Requires `DS4_DSPARK_VERIFY_BATCHED`. |
| `DS4_DSPARK_OUTPUT_BATCHED` | variation | Replaces the drafter's `draft_n` separate output-head `matvec_q8_0` calls with one `matmul_q8_0_batch` (reads the ~917 MB target output weights once). Minor (~+1.3%); the output-head is only ~3-5 ms (the 3-stage forward dominates). **M3 default-on**; opt-out via `=0\|off`. |
| `DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT` | variation | **M3 verifier-improvements.** Generalizes the prefix-1 capture to `block_size-1` slots, so a partial-accept commit restores the layer attention state from a captured slot instead of a sequential replay. A `spec_prefix_capture_valid` flag gates it (disabled for aligned ratio-4 cycles that skip capture -> falls to the replay, no stale KV). **M3 default-on**; opt-out via `=0\|off`. Required for `DS4_DSPARK_ANCHOR_REUSE` to be viable (else partial-replay kills it). |
| `DS4_DSPARK_DRAFT_METAL` | variation | **M3 lever 3.** Ports the PR #502 `metal_graph_dspark_*` GPU Metal drafter (the 6 functions + noncausal batch attention + the batch capture + the `refresh_verified_rows` commit lifecycle + `dspark_n_real` advance). Replaces the CPU drafter in `ds4_session_eval_speculative_argmax`; draft 45→7.6 ms (3.9× faster). Requires `DS4_DSPARK_VERIFY_BATCHED` (accepted drafts persist via the batched commit). **M3 default-on**; opt-out via `=0\|off`. |
| `DS4_DSPARK_DRAFT_METAL_STS` | variation | **M3 lever 3 STS composition.** The Metal drafter computes the learned confidence (`conf_logits` via the `conf_proj` head) and the STS adapts the batch `verify_n` (instead of fixed `verify_n = draft_n`). The threshold function (`dspark_schedule_threshold`) auto-selects 0.15 for Metal+STS vs 0.08 for the CPU path. Requires `DS4_DSPARK_DRAFT_METAL`. **M3 default-on**; opt-out via `=0\|off`. |
| `DS4_DSPARK_CONF_SCHEDULE` | variation | Enables or disables confidence-scheduled verification. |
| `DS4_DSPARK_CONF_THRESHOLD` | variation | Sets the survival threshold used by DSpark scheduled verification. |
| `DS4_DSPARK_SCHEDULE_BATCHED` | variation | Enables the experimental batched scheduled-drafter path. |
| `DS4_DSPARK_SCHEDULE_BATCH_N` | variation | Caps how many draft rows the batched scheduler computes. |
| `DS4_DSPARK_SPEC_LOG` | diagnostic | Enables DSpark speculative-cycle logging. |
| `DS4_DSPARK_DRAFT_PARITY` | diagnostic | Logs Metal-drafter vs target draft-token parity per cycle (`base_real`, `draft0`, `target_next`, match/no-match, the first 8 drafts). Used during lever-3 parity debugging. |
| `DS4_DSPARK_TIMING` | diagnostic | Enables per-cycle DSpark timing capture. |
| `DS4_DSPARK_VERIFY_DIST_PROBE` | diagnostic | Measurement 1 (option A): non-committing probe that runs the sublinear batched verifier (`verify_suffix_tops`) alongside the real sequential DSpark verify and records per-position argmax-flip / max-abs-logit / TV / KL(seq‖batched) into `dspark_last_cycle.verify_dist`, emitted per-cycle as `verify_dist` objects in the ds4-spec-bench JSONL. Requires the graph to allocate `spec_logits` + spec-frontier tensors for DSpark (now done). Non-mutating to the real decode. |
| `DS4_DSPARK_DUMP_HIDDEN` | diagnostic | Live greedy-spine hidden dump (milestone 2 live-vs-oracle trace). When set to a path, forces a host refresh of the DSpark `main_hidden` (layer-40/41/42 mean) at every committed token and appends a (pos, token, hidden[3*N_EMBD]) record. Because greedy speculative preserves the greedy spine, these per-committed-token hiddens ARE the clean capture the offline oracle drafter consumes. Set per-prompt by `ds4-spec-bench --dump-hidden-dir`. |

## Inventory notes for current research

- **M3 is the default for `--dspark` (2026-07-19 ungate, commit `c2a3df0`).** All
  six M3 levers — `DS4_DSPARK_VERIFY_BATCHED` (lever 1), `DS4_DSPARK_ANCHOR_REUSE`
  (lever 2), `DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT` (verifier-improvements),
  `DS4_DSPARK_DRAFT_METAL` (lever 3), `DS4_DSPARK_DRAFT_METAL_STS` (lever 3 STS),
  and `DS4_DSPARK_OUTPUT_BATCHED` — are now **default-on whenever `--dspark` is
  loaded**, opt-out via `=0`/`=off` per lever. The full stack engages across every
  frontend that reaches `ds4_session_eval_speculative_argmax` (`ds4`, `ds4-server`,
  `ds4-eval`, `ds4-spec-bench`); `ds4-bench` does not support `--dspark`.
- **User-facing contract change (accepted 2026-07-19):** `--dspark` is no longer
  byte-exact. Greedy (temp=0) is score-neutral (the batched verify flips ~0.64%
  of argmax positions; 56.6% committed-token divergence on the benchmark; net +4
  on 92Q at 90.2% same-verdict). Sampling (temp>0) is distribution-close, not
  exact (TV ~0.0104). The exact sequential verify remains reachable by disabling
  the levers via env (`DS4_DSPARK_VERIFY_BATCHED=0`, etc.).
- **Server-side speculation gate** (`server_should_speculate` in `ds4_server.c`,
  out of ds4.c scope but load-bearing here): lifted so `--dspark` speculates at
  any temperature; `--mtp` stays temp≤0 only. `DS4_MTP_SPEC_DISABLE` (also in
  `ds4_server.c`) still forces plain decode for both paths.
- **Measured (M5 Max, IQ2XXS, warm, full M3 stack):** **+4.9% over plain**
  (40.04 vs 38.16 t/s on the 176-entry corpus; +4.8% on the long-context
  baseline_corpus), the first config to beat plain locally. The Metal drafter
  (lever 3) drove draft 45→7.6 ms (3.9×). The verify still dominates (~80% of the
  cycle, ~16 ms/token); +20% needs Lead 08 (a fused verify kernel). Full record:
  `summaries/dspark_runtime_milestone_3_progress.md`.
- `DS4_DSPARK_SCHEDULE_BATCHED` and `DS4_DSPARK_SCHEDULE_BATCH_N` remain
  experimental execution-shape knobs for the scheduled drafter (NOT promoted by
  the M3 ungate; still default-off). `DS4_DSPARK_SCHEDULE_GPU_HEAD` was **removed
  from `ds4.c`** (it was the prior GPU-head cut line that failed schedule-parity;
  replaced by `DS4_DSPARK_DRAFT_METAL`, the PR #502 Metal drafter port).
- The largest instrumentation surface in `ds4.c` today is the Metal streaming
  and graph-debug stack. That surface should be treated as part of the research
  harness, not only as production runtime configuration.

## Retained milestone-2 analysis tooling

- **ds4-spec-bench** (`ds4_spec_bench.c`): the one-load bulk speculative bench.
  Milestone-2 additions: per-cycle `anchor_id` + `draft_ids[5]` in the DSpark
  metrics (live-vs-oracle trace); a per-run `think` config field (`high`/`max`/
  `none`) so the bench can reproduce `ds4`'s default `DS4_THINK_HIGH` chat
  template; `--dump-hidden-dir` to drive `DS4_DSPARK_DUMP_HIDDEN` per prompt; the
  `verify_dist` JSON block (incl. `batched_verify_ms`) per cycle.
- **`dspark_oracle/measure_rejection_acceptance.py`** — measurement 2: projects
  rejection-sampling acceptance (1-TV / min(1,p/q)) from retained draft + target
  distributions; emits per-position + E[a|K] (greedy vs rs-greedy-draft vs
  rs-sampled).
- **`dspark_oracle/measure_acceptance_bundle.py --emit-draft-dist`** — extended
  to emit the per-position drafter distribution (top-K ids+probs) sidecar
  (`draft_dist.json`) consumed by the RS + live-vs-oracle analyses.
- **`dspark_oracle/run_rejection_acceptance_all.py`** — one-load driver that runs
  the oracle drafter over many bundles and the RS measurement together.
- **`dspark_oracle/convert_dumps_to_bundles.py`** — converts live hidden dumps
  (`DS4_DSPARK_DUMP_HIDDEN`) into the oracle-bundle format so the offline oracle
  drafter runs on the LIVE greedy-spine hiddens (self-aligned).
- **`dspark_oracle/compare_live_vs_oracle.py`** — aligns the live runtime trace
  to the oracle drafts on the same greedy spine (robust sequential anchor match)
  and reports p1 / mean-prefix / draft-agreement. Decisive for the
  drafter-faithfulness question (n=946 result).
- **Environment:** `issue468/.venv` + `issue468/requirements.txt` (numpy, torch
  MPS, safetensors, pyarrow, tqdm, modal, huggingface-hub). Single consolidated
  analysis venv for the dossier; recreate from requirements.txt.

## Retained milestone-3 analysis tooling

- **ds4-eval engagement fix** (`ds4_eval.c`): `--dspark` now routes generation
  through `ds4_session_eval_speculative_argmax` (previously `--dspark` only loaded
  the drafter; generation used plain `ds4_session_eval`). Gated; the plain path is
  unchanged. Required for the 20-question no-regression gate (the DSpark path must
  engage).
- **ds4-spec-bench token dump** (`DS4_BENCH_DUMP_TOKENS`, `ds4_spec_bench.c`):
  when set to a directory, writes `<id>.ids` (token ids) + `<id>.txt` (decoded
  text) per run, for committed-output divergence diffing.
- **ds4-spec-bench large-corpus bench**: the bulk-config JSONL harness runs the
  300-prompt corpus (Stage 2 + Lead 03 manifests) via ds4-spec-bench (NOT a `ds4`
  loop) with per-cycle timing + bootstrap-CI aggregation, for spec_speedup_model.md-
  comparable fidelity.
- **dist-probe** (`DS4_DSPARK_VERIFY_DIST_PROBE`, above): the batched-vs-exact
  distribution-divergence measurement (TV / argmax-flip / max-abs / KL).
- **PR #502 worktree** at `/Users/lobanov/Projects/ds4-pr502` (branch `pr-502`,
  fetched as `pull/502/head`): the Metal DSpark drafter port source for lever 3
  (`metal_graph_dspark_*` + `ds4_metal.m` kernels + `metal_graph_dspark_refresh_verified_rows`).
