# Issue #304 Engine Residency

This artifact records the model residency decision for turning the Phase 2 proof path into a practical local-generation handoff workflow.

## Current Status

- Phase 2 proved distributed prefill can hand off through a merged `DSV4` payload and continue local decode after the distributed engine is released.
- That proof path opens a fresh full local engine/session for decode. It is acceptable as a research harness, but not yet a user-facing workflow.
- Phase 3 and Phase 3.5 completed variance classification well enough to document risk, but not well enough to prove product harm.
- The next phase is a fused residency-plus-practical-workflow phase: the residency decision now exists to support a runnable, benchmarkable handoff workflow rather than as a separate precondition.

## Known Constraints

- Distributed `--layers` currently influences both route ownership and local model layer residency.
- Local decode needs full model weights, output/logit capability, and full KV state.
- A coordinator that maps only its route slice cannot directly perform full local decode without changing residency or opening another full engine.
- The existing validated handoff boundary is the merged `DSV4` payload.

## Candidate Directions For Phase 4

- Preferred first experiment: one full-resident final-worker engine with sliced distributed execution.
  - `--layers` should still describe each participant's route-owned prefill slice.
  - The final worker should map full local-decode weights and output head when the probe mode is enabled.
  - Distributed prefill should continue to call `ds4_session_eval_layer_slice()` only for the worker's later-layer route slice.
  - Local decode should run from a separate local session over the same full-resident final-worker engine after loading or merging the handoff state.
- Route/residency decoupling:
  - Decouple final-worker loaded spans from distributed route ownership only for the Phase 4 probe.
  - Keep the coordinator-first topology and coordinator early-layer prefill unchanged.
  - Defer public CLI naming until Phase 5 chooses the API shape.
- Dual session over one engine:
  - Use the final worker's distributed-prefill session and a local non-distributed session for decode.
  - Measure whether this duplicates only KV/session graph memory or creates unexpected model-weight duplication.
- Lazy or staged residency expansion:
  - Keep as a fallback if full upfront final-worker residency is too expensive.
  - Only pursue it if backend model-map spans, cached tensors, and preloaded expert tables can be expanded safely and cheaply.
- Two engines or engine reload:
  - Keep as the Phase 2/3 correctness harness and fallback diagnostic.
  - Do not choose it as the practical workflow unless every one-engine option fails and the measured compromise is explicitly accepted.

## Phase 4 Work Plan

1. Record the current loading contract.
   - Trace `ds4_engine_open()` from `opt->distributed.layers` to `load_layer_start`, `load_layer_end`, `load_output`, and `load_output_optional`.
   - Record how `weights_model_map_spans()` and `ds4_gpu_set_model_map_spans()` restrict model mapping.
   - Record how output-head ownership differs between `N:output` and `N:42`.

2. Build a full-resident-final-worker probe.
   - Let the final worker keep full local decode residency while advertising/executing only its distributed later-layer route slice.
   - Verify `weights_layers_bound()` and `ds4_session_eval_layer_slice()` still accept sliced execution against a full-resident worker engine.
   - Keep the first probe internal or harness-only.

3. Measure memory and timing.
   - Compare route-slice final worker, full-resident final worker, dual-session-over-one-engine, and the current two-engine proof harness.
   - Capture peak process memory, backend allocation behavior, staged payload bytes, stage/load time, and local decode throughput.
   - Keep `ctx` and `prefill_cap` aligned across coordinator and worker before interpreting failures.

4. Prototype the practical handoff workflow.
   - Use staged merged `DSV4` as the first correctness boundary.
   - Run distributed prefill in the distributed session.
   - Transfer or merge the coordinator-owned early-layer KV state into the final worker's full local decode session.
   - Prefer the existing merged `DSV4` boundary first if it is the lowest-risk way to prove the workflow.
   - Load the handoff state into the local decode session on the same full-resident final-worker engine.
   - Continue local decode and report clear diagnostics for missing full weights, missing output head, layout mismatch, stale worker state, or memory-budget failure.

5. Decide the Phase 5 API shape.
   - If staged `DSV4` is cheap enough and operator flow is acceptable, carry whole-payload handoff forward.
   - If temp-file staging is the main cost or workflow problem, consider an in-memory payload helper.
   - Defer explicit `DSVL` shard merge and incremental KV return unless measurements show whole-payload handoff is not viable.

## Acceptance Direction

The Phase 4 residency design should preserve the existing guardrails without turning them into a false proof of equivalence:

- do not regress the current official/local vector surfaces,
- keep the known resumed-route variance documented,
- and produce a practical workflow that can later be benchmarked realistically against direct-generation baselines.

Phase 4 exits successfully when one full-resident final-worker workflow can run distributed prefill and local-only decode without closing/reopening the model at handoff time, or when the measured reason that this is not viable is recorded with the smallest next experiment.
Completed implementation validation must cover both `CUDA -> Metal` and `Metal -> CUDA` routes before the workflow is treated as complete.
