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

> **Phase 0 note (exactness is the crux).** On local Metal the only verifier
> that preserves exact greedy output (`--quality`) is a −28% regression, while
> the fast `--mtp` path that beats baseline does so by perturbing near-tied
> logits. The "exact greedy output preservation" requirement is therefore the
> binding constraint of this whole effort. Two deferred branches (see Decision
> Rule) address it: (A) make the exact verifier cheap, or (B) relax exactness
> and validate via real-world task benchmarks. Phase 1 collects data to choose
> between them.

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

> **Working hypothesis status after Phase 0: CONFIRMED.** The risk materialized
> exactly as written — verifier cost dominates and, under the exactness
> constraint, erases the gain. See Phase 0 Outcome below.

## Phase 0 Outcome (EXECUTED — full data in `issue468/`)

Machine: Apple M5 Max / 128 GB / Metal, git `c7ef1bf`. Backend: Metal.

**Where a speculative cycle is spent.** Draft ≈ 2 ms; snapshot/prefix/replay
≈ sub-millisecond (±0.5 ms across all replay strategies); **the verifier owns
the cycle**: ~33 ms/cycle (fast batch) to ~67 ms/cycle (exact), against a
~28 ms baseline target decode step. State-management overhead is *not* the
problem; verifier kernel cost is.

**Current `--mtp` vs baseline (depth 2, greedy, `-n 256 -c 8192`):**

| path | gen t/s | vs target-only | exact greedy? |
|---|---|---|---|
| target-only (chat / code) | 35.60 / 35.78 | — | yes |
| `--mtp` default fast (chat / code) | 37.76 / 38.21 | **+6% / +7%** | **no** |
| `--mtp --quality` exact (chat) | 25.65 | **−28%** | yes |

Default depth-2 MTP is far under the 20% primary gate, and the only exact path
is a net loss. So on local Metal, MTP at depth 2 is **not** a real win under the
success criteria.

**Verifier economics at L=2 (the Phase 1 input):**

| verify | cost | vs 2× sequential (~56 ms) |
|---|---|---|
| 1 token (margin-skip) | 26 ms | — |
| 2 tokens (batch, layer-fused) | 40 ms | 0.71× (sub-linear; saves only on full accept) |
| 2 tokens (exact, decode-kernel ×2) | 58–67 ms | 1.04–1.20× (**slower than baseline**) |

**Draft quality.** Default MTP position-2 conditional acceptance is only
**0.41–0.42** (paper DFlash ≈ 0.63–0.72). Even a free verifier caps gains at
current acceptance.

**Critical structural fact for Phase 1.** The exact fused verifier
(`metal_graph_verify_decode2_exact`, `ds4.c:21218`) exists **only for N=2**.
For L>2 there is no exact-fused kernel: the code falls back to the batch
verifier + snapshot/replay, or to the linear sequential path (one decode kernel
per token, ~28 ms × L). So the exact verifier is *both* super-linear at L=2 and
*non-existent* as a fused kernel for L>2.

**Instrumentation.** No engine change was needed. The existing `DS4_MTP_TIMING`
surface plus a small parser (`issue468/parse_spec_log.py`) produced the full
breakdown. The aggregate-counter delta proposed in `issue468/02_gap_and_spec.md`
is **deferred to Phase 4**. Two documentation corrections: `ds4-bench` cannot
drive the spec path (biggest harness gap), and `DS4_DECODE_PROFILE_DETAIL` is
CPU-only on this build.

**Phase 0 decision: PROCEED to Phase 1 — narrowed to the exact-verifier
question.** See the updated Phase 1 below and the Decision Rule for the two
deferred branches.

## Phase 0: Baseline Measurement And Instrumentation  — STATUS: EXECUTED

### Objective

Build the measurement harness needed to reason about speedup before adding DSpark.

> **Outcome:** see "Phase 0 Outcome" above. The stop/go gate was met — baseline
> timing clearly identifies the verifier as the dominant cost. Work items 1, 2,
> and 3 were largely satisfied by the pre-existing `DS4_MTP_TIMING` surface;
> work item 4 (machine-readable spec output) is deferred to Phase 4.

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

## Phase 1: Verifier Cost Curve Study  — NOW THE MAKE-OR-BREAK PHASE

### Objective

Quantify `verify(L)` for L=1..5 on **both** exactness regimes, because Phase 0
showed the exact verifier is the binding constraint. This phase decides whether
the effort proceeds, narrows (Branch A), or relaxes exactness (Branch B).

### Why this changed (from Phase 0)

- The fast batch verifier (`metal_graph_verify_suffix_tops`, `ds4.c:21117`) is
  sub-linear at L=2 (40 ms ≈ 0.71× of 2× sequential) but **not bit-exact**.
- The exact fused verifier (`metal_graph_verify_decode2_exact`, `ds4.c:21218`)
  is super-linear at L=2 (58–67 ms ≈ 1.04–1.20× of 2× sequential) and
  **exists only for N=2**.
- The only exact option for L>2 today is the linear sequential path
  (`metal_graph_eval_token_raw_swa` per token, ~28 ms × L).

### Code Touchpoints

- `ds4.c` — the three verifier kernels above and the path selection in
  `ds4_session_eval_speculative_argmax()` (`ds4.c:27167`, decision at
  `ds4.c:27346`).
- a microbenchmark entrypoint. **Recommended: a new research-only function in
  `ds4.c` (mirroring the `ds4_engine_metal_graph_*_test` family), invoked by a
  CLI/test flag — not a `ds4-bench` overhaul.** The latter is the Phase 0
  harness gap but is larger and riskier; defer unless the microbench proves
  insufficient.

### Work

1. Build a microbenchmark that calls each verifier kernel directly on
   synthetic suffixes of length L=1..5, isolated from drafter/accept logic:
   - batch (`metal_graph_verify_suffix_tops`)
   - exact fused N=2 (`metal_graph_verify_decode2_exact`) — L=2 only
   - sequential exact (`metal_graph_eval_token_raw_swa` × L) — the exact
     baseline at any L
2. Sweep L=1..5 × context sizes (2k/4k/8k) × prompt classes (code, chat).
3. **Internal phase breakdown of the exact verifier at L=2** (Branch A data):
   separate the cost of the 2× single-token layer dispatches, the per-layer
   prefix-1 capture, and the 2× output-head + 2× full-vocab readbacks. This is
   where the headroom candidates live.
4. Derive a `verify(L)` curve per kernel and an `SPS(B)`-equivalent from the
   batch kernel for the simulator.

### Deliverables

- `verify(L)` curves for **three kernels** (batch / exact-fused-N2 /
  sequential-exact), L=1..5, for code and chat.
- the internal cost breakdown of the exact verifier at L=2.
- a one-paragraph verdict: is the exact curve sub-linear, linear, or cliffy,
  and where is the headroom.

### Stop/Go Gate

Phase 1 no longer just "changes priority order." It selects a branch:

- If an exact `verify(L)` curve can be made sub-linear (Branch A viable) →
  proceed to Phase 2 with exact costs.
- If only the batch curve is sub-linear and exact stays linear/super-linear →
  Branch B becomes the path: Phase 2 must add a benchmark-quality methodology,
  and exact-greedy preservation is dropped from the primary gate (see Decision
  Rule).
- If even the batch curve is cliffy at low L → stop local work; reconsider
  server-side only.

## Phase 2: Offline Feasibility Simulator

### Objective

Test whether DSpark can plausibly win on `ds4` timings before implementing kernels or loaders.

> **Phase 0/1 dependency.** The simulator must use the cost curve from the
> branch selected in Phase 1: exact costs if Branch A, batch costs if Branch B.
> If Branch B is chosen, this phase also stands up the real-world task
> benchmark methodology (HumanEval/MBPP/MT-Bench/Arena-Hard-style evals) needed
> to prove the batch verifier's logit drift is quality-neutral — promoted here
> from Phase 7, because it becomes load-bearing rather than a robustness check.

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

### Step 1: Instrumentation  — DONE (Phase 0)

- `ds4.c` / `ds4_cli.c` already carry `DS4_MTP_TIMING` + `DS4_MTP_*` diagnostics.
- Aggregate counters / `ds4-bench --spec` deferred to Phase 4 (see
  `issue468/02_gap_and_spec.md`); not needed for Phase 1.

### Step 2: Verifier Microbench  — NEXT (Phase 1, make-or-break)

- `ds4.c` (three verifier kernels + path selection)
- a new focused microbench entrypoint (not a `ds4-bench` overhaul)

### Step 3: DSpark Auxiliary Loader  — CAN RUN IN PARALLEL with Step 2

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

## Risks To Watch Early  (status after Phase 0)

- hidden-state extraction from target layers may cost too much. — *open; first tested in Phase 4.*
- the Markov head may be cheap in FLOPs but expensive in host-device orchestration. — *open; Phase 4.*
- verifier replay/commit overhead may erase accepted-prefix gains. — **DISPROVEN by Phase 0**: replay/snapshot is sub-ms; the verifier *kernel*, not state management, is the cost.
- DSpark auxiliary GGUF loading may require broader format work than expected. — *open; Phase 3 (can run in parallel with Phase 1).*
- local single-request workloads may benefit less from scheduling than the paper suggests. — *accepted in advance; the paper's scheduler is a server-concurrency win (Phase 6 note).*
- **NEW (Phase 0):** exact verification is super-linear at L=2 and non-existent as a fused kernel for L>2. This is now the dominant risk and the subject of Phase 1.
- **NEW (Phase 0):** MTP suffix acceptance is weak (pos-2 ≈ 0.41–0.42 vs paper DFlash ≈ 0.63–0.72). Even a free verifier caps gains until drafting improves.

## Recommended First Concrete Tasks  (status after Phase 0)

1. ~~Add speculative timing counters and structured benchmark output.~~ — **DONE** (pre-existing `DS4_MTP_TIMING` + `issue468/parse_spec_log.py`; aggregate counters deferred to Phase 4).
2. Add a verifier cost microbench for suffix lengths `1..5`, **on both exact and batch kernels, plus an internal breakdown of the exact verifier at L=2.** — **NEXT (Phase 1).**
3. Build the offline feasibility simulator from those measured timings. — Phase 2 (use the curve from the Phase 1 branch decision).
4. Inspect DSpark checkpoint metadata and draft a ds4-native auxiliary GGUF schema. — Phase 3 (**can start in parallel with task 2**).
5. Implement a draft-only Markov-head DSpark runner on Metal. — Phase 4 (gated on Phase 1).

## Decision Rule

Proceed to full integration only if all three are true:

- measured verifier costs leave room for speculative gain,
- draft-only DSpark quality and latency look plausible,
- fixed-length DSpark beats baseline materially in end-to-end greedy local decode.

### Deferred decision branches (decide after Phase 1, not now)

Phase 0 made "exact greedy output preservation" the binding constraint. Two
branches address it; Phase 1 collects the data to choose:

- **Branch A — keep exactness, find exact-verifier headroom.** Attack the exact
  verifier's cost (2× single-token layer dispatches, per-layer prefix-1 capture,
  2× output-head + 2× full-vocab readbacks; possibly bit-stable batched
  reductions or a single exact-fused-N kernel). Keeps the primary gate as
  written. Chosen if an exact `verify(L)` curve can be made sub-linear.
- **Branch B — relax exactness, validate via real-world task benchmarks.** Accept
  the fast batch verifier's near-tied logit drift and prove it is quality-neutral
  on HumanEval/MBPP/MT-Bench/Arena-Hard-style evals. The batch curve (already
  sub-linear at L=2) becomes load-bearing; benchmark-quality methodology is
  promoted to Phase 2. Chosen if only the batch curve is sub-linear and the
  drift is shown to be quality-neutral. Risk to retire: default `--mtp` is
  currently *token-different* from target-only, so Branch B must show the drift
  is harmless, not merely small.

If neither branch yields a sub-linear verifier curve at realistic L, stop local
work and reconsider server-side only (where the paper's gains actually live).
