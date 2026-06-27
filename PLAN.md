# DSpark Feasibility Research Plan For `ds4`

## Goal

Prove or disprove that a DSpark-style speculative path can deliver a **material local decode speedup** on the `ds4` engine architecture.

This is a research plan, not a product plan. The first objective is not full DSpark parity with the paper. The first objective is to determine whether `ds4` can realize a real local speedup from:

- a parallel draft pass,
- a lightweight sequential correction head,
- and eventually a scheduled verification prefix.

The output of this work should be a decision:

- proceed with DSpark integration,
- narrow the scope,
- or stop because `ds4`'s architecture does not admit a worthwhile speedup.

## Success Criteria

Primary gate:

- demonstrate `>= 20%` greedy decode throughput improvement versus baseline `ds4` target-only decoding on at least one realistic local machine/backend setup, with exact greedy output preservation.

Secondary gates:

- beat or clearly match the current `--mtp` speculative path on the same workload.
- preserve exact greedy output stream relative to target-only decode.
- avoid pathological memory overhead or replay overhead.
- show a plausible path to server-side gains after local single-request gains are proven.

## Non-Goals For The First Research Cycle

- full DSpark training pipeline.
- full serving scheduler integrated into `ds4-server`.
- RNN head support.
- support for every backend from day one.
- polished CLI or stable public flags.

## Relevant Existing Code

The current speculative path is MTP-specific:

- `ds4_engine_options` currently exposes `mtp_path`, `mtp_draft_tokens`, and `mtp_margin` in [ds4.h](/Users/lobanov/Projects/ds4-dspark/ds4.h:94).
- CLI greedy speculative decode enters through `ds4_session_eval_speculative_argmax()` in [ds4_cli.c](/Users/lobanov/Projects/ds4-dspark/ds4_cli.c:476).
- The engine loads a separate MTP GGUF and binds fixed `mtp.0.*` tensors in [ds4.c](/Users/lobanov/Projects/ds4-dspark/ds4.c:4436) and [ds4.c](/Users/lobanov/Projects/ds4-dspark/ds4.c:25682).
- The current verifier already supports multi-token draft verification through `metal_graph_verify_suffix_tops()` in [ds4.c](/Users/lobanov/Projects/ds4-dspark/ds4.c:27466).

The GGUF tooling is Flash-specific and template-driven:

- quantization and GGUF regeneration live under [gguf-tools/README.md](/Users/lobanov/Projects/ds4-dspark/gguf-tools/README.md:1).

## Working Hypothesis

DSpark can only beat baseline locally if:

`(target step to get anchor + dspark draft pass + sequential head + verification) / accepted_tokens`

is materially lower than:

`plain target decode cost per emitted token`

for at least one meaningful prompt class on at least one intended local machine.

The main risk is that `ds4`'s verifier and state-management overhead erase the benefit of longer accepted prefixes.

## Phase 0: Baseline Measurement And Instrumentation

### Objective

Build the measurement harness needed to reason about speedup before adding DSpark.

### Code Touchpoints

- `ds4_cli.c`
- `ds4_server.c`
- `ds4.c`
- `ds4_bench.c`
- possibly a new small benchmark helper under `tests/` or `speed-bench/`

### Work

1. Add opt-in timing instrumentation for decode cycles.
2. Split cycle timing into:
   - target sample/top-token selection
   - target decode step
   - speculative draft time
   - verifier time
   - replay/commit time
3. Record speculative quality counters:
   - draft length
   - committed length
   - acceptance by draft position
   - full accept count
   - partial accept count
4. Add a machine-readable output mode for benchmarking.

### Deliverables

- repeatable benchmark command set for local Metal first.
- CSV or TSV output for later analysis.
- baseline numbers for:
  - target-only greedy decode
  - current `--mtp` path

### Stop/Go Gate

Do not implement DSpark execution until baseline timing clearly identifies where `ds4` spends time in speculative decode.

## Phase 1: Verifier Cost Curve Study

### Objective

Measure the actual cost of verification on `ds4`, because DSpark viability depends on verifier economics.

### Code Touchpoints

- `ds4.c`, especially the existing speculative verifier path around [ds4.c](/Users/lobanov/Projects/ds4-dspark/ds4.c:27195) and [ds4.c](/Users/lobanov/Projects/ds4-dspark/ds4.c:27466)
- `ds4_bench.c` or a new benchmark entrypoint

### Work

1. Add a microbenchmark that exercises `metal_graph_verify_suffix_tops()` for suffix lengths `1..N`.
2. Measure verifier wall time as a function of:
   - suffix length
   - context size
   - prompt class
3. For local research, derive a simple `SPS(B)` or equivalent throughput/cost curve from actual measured verifier behavior.

### Deliverables

- verifier cost curve for local target setups.
- a short note explaining whether verifier cost is close to linear, sublinear, or cliffy.

### Stop/Go Gate

If verifier cost grows too aggressively with suffix length, DSpark may still be useful, but only with strong prefix pruning. That changes the implementation priority order.

## Phase 2: Offline Feasibility Simulator

### Objective

Test whether DSpark can plausibly win on `ds4` timings before implementing kernels or loaders.

### Code Touchpoints

- new script under `speed-bench/` or `tools/`
- benchmark outputs from Phases 0 and 1

### Work

1. Build a simulator that consumes:
   - baseline target decode cost
   - verifier cost curve
   - assumed draft latency
   - assumed accepted-prefix probabilities by position
2. Simulate:
   - fixed-length verification
   - confidence-truncated verification
   - comparison to plain target decode
   - comparison to current `--mtp`
3. Use actual DSpark release parameters where available, especially the released block size rather than paper-general assumptions.

### Deliverables

- required accepted-prefix profile for break-even.
- required draft latency budget for break-even.
- a ranking of the most likely win conditions:
  - code
  - chat
  - long context
  - short context

### Stop/Go Gate

If the simulator says DSpark needs unrealistic draft latency or unrealistic acceptance to beat baseline, stop before runtime integration.

## Phase 3: Loader And Format Reconnaissance

### Objective

Determine the minimum work needed to load DSpark auxiliary tensors into `ds4`.

### Code Touchpoints

- `ds4.c` loader and tensor binding code
- `gguf-tools/deepseek4-quantize.c`
- optional inspection tools

### Work

1. Inspect the DSpark checkpoint tensor names and metadata.
2. Compare them to the current hardcoded MTP binding path in [ds4.c](/Users/lobanov/Projects/ds4-dspark/ds4.c:4436).
3. Decide the first format strategy:
   - direct load from original checkpoint layout, or
   - convert into a ds4-native auxiliary GGUF layout.
4. Prefer a format that supports fast iteration and avoids rewriting the whole main model pipeline.

### Recommended First Strategy

Use a **separate DSpark auxiliary GGUF**, parallel to `--mtp`, instead of modifying the main target GGUF first.

Reasons:

- it matches the current engine structure.
- it minimizes risk to the main model path.
- it keeps research iterations localized to speculative code.

### Deliverables

- documented proposed tensor schema for a ds4-native DSpark auxiliary GGUF.
- list of required metadata fields such as:
  - block size
  - target layer ids
  - markov rank
  - hidden dimensions

### Stop/Go Gate

If the DSpark checkpoint cannot be mapped into a tractable auxiliary format without large engine surgery, reassess whether to prototype with an oracle or simplified synthetic module first.

## Phase 4: Draft-Only DSpark Prototype

### Objective

Run DSpark drafting in isolation before attempting speculative decode integration.

### Scope

Start with:

- greedy only
- one backend first: Metal
- Markov head only
- fixed released block size
- no confidence scheduler yet

### Code Touchpoints

- `ds4.c`
- `metal/` kernels if needed
- possible new structs beside current MTP structures

### Work

1. Add new engine/session structures for DSpark auxiliary state.
2. Extract target hidden-state features from selected target layers.
3. Run the DSpark parallel block once per cycle.
4. Run the Markov head left-to-right on the drafted block.
5. Emit:
   - draft tokens
   - base logits
   - optional per-position confidence scores

### Measurements

- draft pass latency
- Markov head latency
- total draft latency per cycle
- conditional acceptance by draft position versus target greedy stream
- expected accepted prefix length under exact greedy verification

### Deliverables

- a research-only command or env flag to run draft-only analysis.
- draft-quality reports for representative prompts.

### Stop/Go Gate

If draft-only latency is already too high, or conditional acceptance decays too quickly, stop before verifier integration.

## Phase 5: Fixed-Length DSpark Speculative Decode

### Objective

Answer the core question: can DSpark beat local baseline on `ds4` before any scheduler sophistication?

### Code Touchpoints

- `ds4.c`
- `ds4_cli.c`
- `tests/ds4_test.c`

### Work

1. Add a new speculative path beside MTP, not inside MTP.
2. Reuse existing exact acceptance logic and verifier plumbing where possible.
3. Start with a fixed verification prefix length.
4. Preserve exact greedy output stream.
5. Collect full timing and acceptance counters.

### Recommended API Shape

Add DSpark-specific engine options rather than overloading MTP fields. For example:

- `dspark_path`
- `dspark_block_size`
- `dspark_verify_len`
- `dspark_enable_confidence`

This should remain research-only until viability is proven.

### Deliverables

- end-to-end greedy speculative DSpark decode on local `ds4`.
- benchmark comparison against:
  - target-only baseline
  - current `--mtp`

### Stop/Go Gate

If fixed-length DSpark does not show a material local win, do not proceed to scheduler work.

## Phase 6: Confidence Head And Scheduled Prefix Research

### Objective

Test whether confidence-based prefix truncation improves local or server-style performance enough to justify complexity.

### Scope

This phase comes only after Phase 5 shows a real win.

### Code Touchpoints

- `ds4.c`
- `ds4_server.c`
- benchmark harness

### Work

1. Surface confidence scores from the DSpark draft path.
2. Implement the smallest scheduler that chooses a prefix length from:
   - cumulative survival estimates
   - measured verifier throughput curve
3. Start with single-request local mode for validation.
4. Then evaluate whether the same mechanism is worth lifting into `ds4-server`.

### Important Note

The paper's hardware-aware scheduler is most naturally a **server concurrency** optimization. It may provide little benefit in single-request local CLI use. That is acceptable. The local proof goal is still valid if most of the gain comes from better drafting rather than from scheduling.

### Deliverables

- measured delta between:
  - fixed-length DSpark
  - confidence-truncated DSpark
- explicit conclusion about whether scheduler work belongs in local CLI, `ds4-server`, or both.

## Phase 7: Stress And Robustness Study

### Objective

Verify that the speedup is real and not a narrow benchmark artifact.

### Work

Test across:

- short code prompts
- long code prompts
- general chat
- different context sizes
- memory pressure scenarios
- target quant variants

Record:

- speedup distribution
- acceptance distribution
- replay frequency
- memory impact

### Deliverables

- final feasibility memo:
  - where DSpark wins
  - where it loses
  - why

## Proposed File-Level Execution Order

### Step 1: Instrumentation

- `ds4.c`
- `ds4_cli.c`
- `ds4_bench.c`
- `tests/ds4_test.c`

### Step 2: Verifier Microbench

- `ds4.c`
- `ds4_bench.c`
- optionally `speed-bench/`

### Step 3: DSpark Auxiliary Loader

- `ds4.h`
- `ds4.c`
- `gguf-tools/deepseek4-quantize.c` or a new DSpark conversion helper

### Step 4: Draft-Only Execution

- `ds4.c`
- possibly `metal/` kernels

### Step 5: Full Fixed-Length Speculative Decode

- `ds4.c`
- `ds4_cli.c`
- tests

### Step 6: Confidence Scheduler

- `ds4.c`
- `ds4_server.c`

## Testing Strategy

### Unit-Level

- auxiliary tensor metadata parsing
- DSpark option parsing
- Markov head math on small synthetic inputs
- confidence-to-prefix-length logic

### Regression-Level

- exact greedy output equality against baseline target-only decode
- no session corruption after partial accepts
- no checkpoint corruption on replay paths

### Benchmark-Level

- tokens/sec
- ms/cycle
- accepted tokens/cycle
- conditional acceptance by position

## Risks To Watch Early

- hidden-state extraction from target layers may cost too much.
- the Markov head may be cheap in FLOPs but expensive in host-device orchestration.
- verifier replay/commit overhead may erase accepted-prefix gains.
- DSpark auxiliary GGUF loading may require broader format work than expected.
- local single-request workloads may benefit less from scheduling than the paper suggests.

## Recommended First Concrete Tasks

1. Add speculative timing counters and structured benchmark output.
2. Add a verifier cost microbench for suffix lengths `1..5`.
3. Build the offline feasibility simulator from those measured timings.
4. Inspect DSpark checkpoint metadata and draft a ds4-native auxiliary GGUF schema.
5. Implement a draft-only Markov-head DSpark runner on Metal.

## Decision Rule

Proceed to full integration only if all three are true:

- measured verifier costs leave room for speculative gain,
- draft-only DSpark quality and latency look plausible,
- fixed-length DSpark beats baseline materially in end-to-end greedy local decode.
