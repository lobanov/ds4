# DSpark runtime path — current benchmark

Date: 2026-07-12. Status: **partial / provisional**. This note records the
current end-to-end `ds4 --dspark` runtime after five retained DSpark runtime
changes:

1. session-lifetime DSpark scratch replaces per-cycle allocation in the CPU
   draft path;
2. confidence scheduling is folded into the draft evaluator itself, so the
   scheduled path can stop drafting at its own STS frontier;
3. DSpark hidden capture / stage-KV push is GPU-first, avoiding full captured
   HC readback on each committed target token;
4. verifier-accepted DSpark tokens now advance the DSpark support-state window
   instead of only advancing the target checkpoint; and
5. the GPU hidden-push `main_proj` path now consumes the concatenated
   `3 * 4096` layer means, matching the CPU path, rather than incorrectly
   projecting one layer mean at a time.

The retained profiling scripts were also extended to parse a second DSpark
timing-detail line, so the current profile now separates verifier target-decode
time from DSpark state-push time and final logits readback.

The last two fixes are semantic, not cosmetic. They preserved the retained
greedy and conservative temp-parity gates, but they also weakened the prior
runtime headline. That means some of the earlier faster DSpark numbers were at
least partly measuring a distorted support-model state rather than a clean
implementation of the intended DSpark path.

Artifacts:

- greedy exactness: `issue468/artifacts/dspark_exactness_compare/summary.json`
- temp>0 logit/distribution parity:
  `issue468/artifacts/dspark_temp_distribution_compare/summary.json`
- short powered-corpus sample:
  `issue468/artifacts/dspark_corpus_bench/summary.json`
- long-prompt cycle profile:
  `issue468/artifacts/dspark_phaseA_profile/summary.json`
- adversarial optimization review:
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_dspark_opt_review.md`
- reusable bulk speculative bench binary:
  `ds4-spec-bench` from `ds4_spec_bench.c`

## Measurement methodology

This note treats DSpark runtime work as a **two-axis benchmark problem**:
correctness of the final generated stream, and preservation of speculative
economics. The second axis matters because a runtime optimization can preserve
final tokens and sampled distributions while still degrading draft quality,
accepted-prefix length, or full-accept frequency enough to erase any speed
benefit.

The benchmarking contract therefore has four layers.

1. **Final-output correctness gates remain mandatory.**
   Greedy output must remain byte-identical to baseline, and temp>0 parity must
   hold at the target logits / sampling-distribution level rather than only at
   emitted tokens. These are the minimum gates for any speculative runtime
   change.
2. **Draft-side quality must be checked separately from final-output parity.**
   Verifier exactness can hide a degraded drafter: the generated stream still
   matches because rejected drafts are corrected, but acceptance falls and the
   path gets slower. For DSpark, runtime-valid optimizations therefore need
   draft-token parity, confidence-logit parity, scheduled verify-length parity,
   and DSpark window-state parity against the trusted CPU/oracle reference, not
   just final text/logit agreement.
3. **Acceptance metrics are first-class outputs.**
   Each retained bench should continue to report at least drafted length,
   verify length, accepted/verified drafts per cycle, and where relevant the
   full-accept rate. This is especially important for scheduled DSpark because
   its policy intentionally drafts more than it verifies, and because the
   current policy shape is meant to reduce full-block accepts by drafting one
   token beyond the confident prefix.
4. **Cycle-cost attribution must stay split by component.**
   Aggregate tokens/s is not enough to identify the next lever. The current
   profile keeps `decode_ms`, `draft_ms`, `verify_ms`, `verify_decode_ms`,
   DSpark state-push time, and logits readback separate so that verifier cost,
   drafter cost, and state-update cost do not get conflated.

These rules imply a practical benchmarking stance for future DSpark work:

- Any optimization that changes DSpark execution order, placement, or precision
  should be measured on the same retained exactness, powered-corpus, and
  long-context corpora already used here.
- `ds4_spec_bench.c` / `ds4-spec-bench` should be the default DSpark benchmark
  substrate and extended as needed rather than replaced with many narrow
  one-off drivers. The important retained properties are:
  batch submission from one bulk config, low-file-count output (for example
  JSONL summaries rather than one artifact tree per prompt), and reuse of a
  single loaded engine/session so DSpark measurements reflect steady-state
  runtime rather than repeated model-load overhead.
- A change is not validated by greedy exactness plus temp>0 parity alone if it
  moves draft acceptance, confidence values, or scheduled verify lengths enough
  to change speculative economics.
- Throughput comparisons should continue to include both a fixed low-K
  reference (`verify_k=1`) and the scheduled path, because the scheduled path
  can lose for two different reasons: verifier cost or drafter-execution cost.

## What was measured

- **Runtime path:** `ds4 --dspark /Users/lobanov/Projects/ds4/gguf/dspark.gguf`
- **Greedy exactness gate:** retained 10-prompt exactness corpus, `n=32`
- **Temp>0 parity gate:** retained exactness corpus at `temp=0.5` and `1.0`,
  32 sampled steps/prompt via `ds4_test --dspark-temp-logit-parity`
- **Throughput sample:** first 6 prompts from the powered Stage-2 corpus,
  `n=32`, comparing baseline vs DSpark default scheduling vs fixed `verify_k=1`
- **Cycle profile:** retained 8k long-prompt corpus (`code_8k`,
  `synthesis_8k`, `grounded_8k`), `n=32`, DSpark default scheduling and
  fixed `verify_k=1`

## Correctness result

**Greedy exactness: PASS on the retained exactness corpus sample.**

- 10/10 prompts matched baseline byte-for-byte.
- Mean generation throughput on those prompts:
  - baseline: **39.98 t/s**
  - DSpark: **21.94 t/s**

**Temp>0 logit/distribution parity: PASS under the current conservative parity mode.**

- 640 total sampled steps, 574 eligible DSpark steps.
- `max_abs = 0.0`, `rms = 0.0`, `sampled_lp_diff = 0.0`

Important caveat: the current temp-parity gate still forces
`DS4_DSPARK_VERIFY_K=0`, so it proves speculative entry/logit parity while
preventing extra speculative commits beyond the sampled target token. That is
still the right conservative gate for the current implementation, but it is
narrower than a full multi-token speculative parity proof.

## Throughput result

**The corrected DSpark runtime remains below baseline, and the current best
local operating point is still fixed `verify_k=1`.**

On the refreshed 6-prompt Stage-2 sample (`n=32`):

| impl | mean gen t/s | mean total ms/cycle | mean draft ms/cycle | mean verify ms/cycle | mean verified tokens/cycle |
|---|---:|---:|---:|---:|---:|
| baseline | **39.95** | — | — | — | 0.000 |
| DSpark default scheduling | **24.65** | **151.28** | **49.93** | **74.39** | **2.753** |
| DSpark fixed `verify_k=1` | **30.20** | **62.64** | **11.67** | **24.11** | **0.890** |

Interpretation:

- default scheduled DSpark is about **0.62x baseline**
- fixed `verify_k=1` is about **0.76x baseline**
- the path still benefits from speculative acceptance, but the cleaned-up state
  semantics do **not** support the earlier “nearly 31 t/s + improving toward a
  GPU drafter win” framing as strongly as before

## Cycle-cost profile

The retained 8k profile still shows the same basic runtime shape, but with the
semantic fixes in place. The new detail buckets show that **verifier target
decode dominates verifier cost; DSpark state-push overhead is small; and final
logits readback is negligible**.

- `code_8k`, scheduled:
  - gen **23.56 t/s**
  - mean draft **44.78 ms**
  - mean verify **62.02 ms**
  - mean verify target-decode **60.86 ms**
  - mean DSpark verify-state push **1.54 ms**
  - mean logits readback **0.006 ms**
  - mean cycle total **135.75 ms**
  - mean verified **2.20**
  - median verified **2**
- `synthesis_8k`, scheduled:
  - gen **23.01 t/s**
  - mean draft **43.66 ms**
  - mean verify **54.14 ms**
  - mean verify target-decode **53.30 ms**
  - mean DSpark verify-state push **1.33 ms**
  - mean logits readback **0.007 ms**
  - mean cycle total **126.34 ms**
  - mean verified **1.91**
  - median verified **2**
- `grounded_8k`, scheduled:
  - gen **22.09 t/s**
  - mean draft **48.86 ms**
  - mean verify **53.98 ms**
  - mean verify target-decode **52.61 ms**
  - mean DSpark verify-state push **1.31 ms**
  - mean logits readback **0.006 ms**
  - mean cycle total **131.62 ms**
  - mean verified **1.91**
  - median verified **2**

- `code_8k`, `verify_k=1`:
  - gen **28.91 t/s**
  - mean draft **11.45 ms**
  - mean verify **29.40 ms**
  - mean verify target-decode **28.30 ms**
  - mean DSpark verify-state push **0.70 ms**
  - mean cycle total **69.12 ms**
  - mean verified **1.00**
- `synthesis_8k`, `verify_k=1`:
  - gen **28.09 t/s**
  - mean draft **13.67 ms**
  - mean verify **28.53 ms**
  - mean verify target-decode **27.44 ms**
  - mean DSpark verify-state push **0.70 ms**
  - mean cycle total **71.13 ms**
  - mean verified **1.00**
- `grounded_8k`, `verify_k=1`:
  - gen **28.76 t/s**
  - mean draft **10.37 ms**
  - mean verify **22.80 ms**
  - mean verify target-decode **21.89 ms**
  - mean DSpark verify-state push **0.56 ms**
  - mean cycle total **61.74 ms**
  - mean verified **0.778**

## Current interpretation

Five points now look solid.

1. **The current DSpark path is correctness-gated and benchmarkable.** Greedy
   exactness and the conservative temp>0 distribution gate still pass after the
   state-advance and `main_proj` semantic fixes.
2. **The current runtime headline is weaker than the previous provisional one.**
   Once DSpark’s support-state window is advanced correctly and the GPU
   `main_proj` semantics match the CPU path, the measured DSpark throughput
   falls back toward **~22–30 t/s** instead of the earlier more optimistic band.
3. **Verifier target decode is now the clearest ceiling term.** The new detail
   profile shows that most of `verify_ms` is the serial target decode itself;
   DSpark support-state pushes are only about **~0.6–1.5 ms/cycle**, and final
   logits readback is effectively noise at this scale.
4. **Fixed `verify_k=1` remains the best local DSpark operating point.** It is
   the cleanest reference path and still the least-bad throughput point, but it
   is not close enough to baseline to justify a positive speedup claim.
5. **Further optimization work is still justified, but the next step changed.**
   The current evidence does **not** support “full DSpark GPU drafter body/head
   first, keep the verifier unchanged” as the clearest next lever.

## Next steps

**Proceed with DSpark optimization work, but do not headline a positive local
runtime result yet and do not treat a full GPU drafter body/head port as the
first obvious next move.**

This section should be updated after each optimization iteration so it remains
the current workload and recommendation snapshot.

Current workload:

1. keep the new `ds4-spec-bench` path and retained corpus/profile gates as the
   measurement substrate;
2. parity-gate DSpark window-state semantics explicitly; and
3. investigate verifier/state-update economics before committing to a full
   DSpark GPU body/head port.

What is established now:

- DSpark is correctness-gated on the retained greedy and conservative temp>0
  parity checks.
- The path is benchmarkable on the same corpora as the MTP/verifier work.
- The current implementation includes the accepted CPU draft cleanups and two
  additional semantic fixes required for a defensible runtime picture.
- The updated bottleneck picture is clearer: target decode plus exact verifier
  cost are now load-bearing enough that drafter-side speedups alone may still
  fail to clear baseline.
- The most plausible scheduled-DSpark runtime redesign is now “draft a short
  full block efficiently, then use confidence to truncate verify length,” not
  “keep token-serial early-stop drafting as the core optimization.”

What is not established:

- a baseline-beating local DSpark path
- a scheduled DSpark path with economics good enough to justify a positive
  runtime headline
- that “GPU drafter body/head first” is the best next engineering move under
  the current verifier

The live optimization space is now narrower.

1. **Keep verifier work as the primary ceiling.** The timing-detail split
   shows that most of `verify_ms` is serial target decode inside the exact
   verifier, not DSpark support-state push or logits readback. This keeps Lead
   08 Phase B fused low-K verifier work live.
2. **Treat variable verify span as more important than token-serial early-stop
   drafting.** The likely better scheduled runtime design is to batch a short
   full block (`4` or `5` tokens), produce draft ids plus confidence logits in
   one pass, and let confidence choose how much of that block to verify. That
   keeps the scheduling benefit while avoiding much of the current token-serial
   draft overhead.
3. **Preserve the existing “confident prefix + 1 drafted token” policy logic
   even if drafting becomes batched.** The verify-length chooser should keep
   the current asymmetry so the system does not drift back toward paying more
   standalone anchor decodes than necessary.
4. **Move more of DSpark execution onto the GPU only with confidence and state
   carried along.** The remaining drafter-side path points to persistent device
   DSpark KV/window state plus GPU body, head, markov, confidence, and argmax,
   while host policy can remain on CPU because it only consumes a few scalars.
5. **Do not spend more time on drafter-weight precision or scheduler-only
   tuning.** Earlier lead work already closed these as primary levers. The live
   question is execution cost, not whether the confidence signal exists.

The current DSpark path is real and measurable, but the cleaned-up evidence now
points to **verifier and state-update economics as the next load-bearing issue**,
not just “move the drafter to GPU and the rest will probably work out.”

## Worklog

### 2026-07-12 — scheduled-drafter optimization cycle 1

Tried the most obvious drafter-cost optimization suggested by the current
profile: keep confidence scheduling, but make the scheduler's drafting side
**batched** instead of token-serial. Concretely, the experimental path uses a
new opt-in env, `DS4_DSPARK_SCHEDULE_BATCHED=1`, to draft the full short block
with the existing batched CPU drafter and then choose `verify_n` from the
resulting confidence logits, instead of stopping draft compute token-by-token
at the STS frontier.

Important status:

- the default runtime was **not** changed; the retained token-serial scheduled
  path remains the default
- the batched path is **experimental / opt-in only**
- a new lock-safe smoke regression was added to `ds4_test`:
  `--dspark-schedule-parity`

Smoke outcome before any benchmark:

- `./ds4_test --dspark-temp-logit-parity` still passes for the current DSpark
  path
- `./ds4_test --dspark-schedule-parity` currently **fails** for the
  experimental batched path: committed chunking differs from the default serial
  scheduler on the retained exactness prompts

Interpretation:

- the current batched-scheduler implementation is **not yet benchmark-ready**
  under this note's methodology
- even with final-output parity protected by exact verification, the draft-side
  economics are changing enough to trip the new smoke gate
- next step is adversarial review of the divergence before any benchmark claims
  are recorded

Pre-benchmark adversarial review:

- retained at
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batched_schedule_prebench_review.md`
- verdict: the current full-block batched path is **not** serial-equivalent by
  construction, because the batched drafter lets early rows see future/noise
  block rows and therefore changes proposals / confidence / accepted chunking
- recommendation: do **not** benchmark or summarize this path as a valid drop-in
  optimization of the current scheduler; either rework it toward causal
  prefix-limited intra-block attention, or treat it as a distinct speculative
  policy with a different benchmark contract

### 2026-07-12 — scheduled-drafter optimization cycle 2

Reworked the experimental batched scheduler into a **prefix-limited** batched
path rather than the earlier full-block path. The retained default runtime is
still unchanged; the experimental path remains opt-in behind
`DS4_DSPARK_SCHEDULE_BATCHED=1`. The key semantic change is that intra-block
attention for row `t` is limited to the real prefix plus `t+1` visible rows, so
the batched path no longer lets early scheduled rows see future/noise rows.

The pre-benchmark adversarial review for this current implementation is retained
at:

- `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_prefix_limited_batched_schedule_prebench_review.md`

That review judged the current prefix-limited path a plausible semantic match to
the retained serial scheduler, but also correctly pointed out that the old smoke
coverage was still too weak: committed-token / chunk parity alone could miss
draft-id, confidence-logit, or scheduled-`verify_n` drift.

In response, the retained smoke gate was strengthened:

- added `ds4_session_dspark_schedule_probe(...)` so tests can inspect the real
  scheduled DSpark cycle outputs without exposing DSpark internals broadly
- upgraded `./ds4_test --dspark-schedule-parity` to compare, cycle-by-cycle:
  draft ids, confidence logits, computed `verify_n`, committed accepted tokens,
  and accepted chunking

Lock-safe smoke outcome for the current implementation:

- `./ds4_test --dspark-temp-logit-parity`: **PASS**
- strengthened `./ds4_test --dspark-schedule-parity`: **PASS**

Interpretation:

- the current prefix-limited batched scheduled path has now cleared a much
  stronger pre-benchmark parity gate than the earlier rejected full-block path
- it is therefore benchmark-eligible under this note's methodology
- the next step is actual corpus/profile measurement, followed by the required
  post-benchmark adversarial review before any benchmark summary is propagated

First retained benchmark slice after that stronger gate:

- short retained `ds4-spec-bench` sample:
  `issue468/artifacts/dspark_corpus_bench/spec_sched_compare_sample_summary.json`
- retained long-profile reruns:
  - `issue468/artifacts/dspark_phaseA_profile/summary__serial_now.json`
  - `issue468/artifacts/dspark_phaseA_profile/summary__batched_now.json`

Measured result on this first slice was **mixed**:

- short 6-prompt retained sample:
  - serial scheduled mean `tokens_per_second`: **23.303**
  - batched scheduled mean `tokens_per_second`: **22.743**
  - reported chunk-level `accepted_mean`: unchanged at **2.004449**
- retained 8k sched-only profile:
  - `code_8k`: gen **24.17 -> 25.24 t/s**, draft mean **41.832 -> 35.855 ms**
  - `synthesis_8k`: gen **23.20 -> 23.88 t/s**, draft mean **42.748 -> 39.190 ms**
  - `grounded_8k`: gen **22.15 -> 23.70 t/s**, draft mean **48.399 -> 40.168 ms**
  - measured `verified` means were unchanged across those three retained prompts

Post-benchmark adversarial review:

- retained at
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_prefix_limited_batched_schedule_postbench_review.md`
- verdict: this is a **follow-up-worthy mixed signal**, not a validated runtime
  win
- safe framing:
  - the prefix-limited path appears semantically sound enough to measure
  - short retained prompts showed slightly worse throughput with unchanged
    chunk-level acceptance
  - long retained prompts showed lower draft cost and slightly better `gen_tps`
- unsafe framing:
  - claiming the optimization is already proven faster
  - claiming full acceptance economics are identical from the current short
    sample alone

Immediate next step from that review:

- extend `ds4-spec-bench` so the retained DSpark benchmark substrate itself
  records schedule mode plus per-cycle draft/verify/timing fields, instead of
  relying on a split between JSONL throughput runs and separate CLI timing
  parses

### 2026-07-12 — scheduled-drafter optimization cycle 3

Extended `ds4_spec_bench.c` / `ds4-spec-bench` so the retained JSONL output now
captures DSpark run mode and per-cycle telemetry directly, rather than requiring
separate CLI timing parses. The current JSONL records, per run:

- whether DSpark metrics were present
- whether scheduled verification was used at all during the run
- whether batched scheduling was enabled
- DSpark cycle count
- mean drafted / verify / verified / accepted counts per cycle
- mean DSpark timing splits per cycle (`decode_ms`, `draft_ms`, `verify_ms`,
  `total_ms`, `verify_decode_ms`, push timing, logits readback)
- a retained `dspark_cycles` array with the per-cycle raw fields

One schema bug was caught immediately by the first rerun: top-level
`scheduled_verify` had been reflecting only the **last** speculative cycle
rather than summarizing the whole run. That was corrected before the retained
batched rerun, so the v2 artifacts below are directly comparable.

Retained v2 short-sample artifacts:

- serial:
  `issue468/artifacts/dspark_corpus_bench/spec_sched_serial_sample_v2.jsonl`
- batched:
  `issue468/artifacts/dspark_corpus_bench/spec_sched_batched_sample_v2.jsonl`

Measured result on the same 6-prompt retained sample:

- serial scheduled mean `tokens_per_second`: **24.463**
- batched scheduled mean `tokens_per_second`: **22.484**
- run-level `accepted_mean`: unchanged at **2.004449**
- cycle-level accepted mean: unchanged at **1.985386**
- mean verifier time stayed effectively flat:
  **26.677 -> 26.699 ms**
- mean draft time worsened materially:
  **27.875 -> 34.914 ms**
- mean drafted rows per cycle rose:
  **2.940 -> 4.614**
- mean `verify_n` stayed unchanged:
  **2.119 -> 2.119**

Interpretation:

- the integrated `ds4-spec-bench` substrate is now sufficient to replace the
  earlier split throughput-plus-CLI-timing workflow for future DSpark cycles
- on this retained short sample, the current prefix-limited batched scheduler
  is slower for a now-clear reason: it drafts substantially more rows per cycle
  without reducing verifier work
- this does **not** invalidate the earlier positive long-context profile slice,
  but it does narrow the current follow-up question: find cases where batching
  reduces draft cost enough to offset extra drafted rows, or change the
  scheduled policy so batched drafting does not overshoot the current serial
  `verify_n` economics

### 2026-07-12 — scheduled-drafter optimization cycle 4

The next retained cycle pursued the follow-up question directly by making the
experimental batched scheduled span **tunable** rather than hard-wired to the
full short block. The new opt-in env is:

- `DS4_DSPARK_SCHEDULE_BATCH_N=<1..5>`

It only applies when the experimental batched scheduled path is active
(`DS4_DSPARK_SCHEDULE_BATCHED=1`) and scheduled verification is still in use.
The goal is simple: keep the “batch then let confidence choose verify length”
shape from the previous cycle, but reduce the draft-side overcompute that made
cap-5 batching slower on short prompts.

Before benchmarking, this cycle also fixed a real retained-harness confound
found by adversarial review:

- `ds4-spec-bench` no longer reuses one mutable session timeline across runs
- it now reuses **one loaded engine** but creates a **fresh session per run**
- it no longer forces `DS4_DSPARK_TIMING=1`; throughput and attribution passes
  are now intentionally separate
- DSpark side state is explicitly reset on snapshot/payload restore and GPU
  sync-prefill rebuild paths

Those changes were not cosmetic. A retained hygiene smoke had shown that two
identical capped runs through the old reused-session harness could produce the
same high-level counts but different per-cycle confidence traces. After the
fresh-session change, identical repeated runs matched on all non-timing JSON
fields.

Retained pre-benchmark review for this cleaned-up state:

- `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batchcap_prebench_review_v2.md`

That review judged the current harness and cap path benchmark-ready for a short
cap sweep, with two key constraints:

- throughput comparison should run with `DS4_DSPARK_TIMING` unset
- each batched row should be rejected if the JSONL does not show the intended
  scheduler flags (`scheduled_verify=true`, `schedule_batched=true`,
  `schedule_batch_limit=<cap>`)

Lock-safe smoke outcome before benchmarking:

- `./ds4_test --dspark-temp-logit-parity`: **PASS**
- `./ds4_test --dspark-schedule-parity`: **PASS**
- one-prompt cap smoke via `ds4-spec-bench`: **PASS**
- repeated-run fresh-session hygiene smoke: **PASS**

Retained benchmark artifacts for the short 6-prompt Stage-2 sample:

- throughput pass, timing **off**:
  - serial:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_serial_sample_v3.jsonl`
  - cap 5:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_batched_cap5_sample_v3.jsonl`
  - cap 4:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_batched_cap4_sample_v3.jsonl`
  - cap 3:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_batched_cap3_sample_v3.jsonl`
- diagnostic attribution pass, timing **on**:
  - serial:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_serial_sample_v3_timing.jsonl`
  - cap 4:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_batched_cap4_sample_v3_timing.jsonl`
  - cap 3:
    `issue468/artifacts/dspark_corpus_bench/spec_sched_batched_cap3_sample_v3_timing.jsonl`
- retained roll-up:
  `issue468/artifacts/dspark_corpus_bench/spec_sched_batchcap_v3_summary.json`

Measured throughput-pass means on this retained sample:

- serial: **24.712 t/s**
- cap 5: **24.058 t/s**
- cap 4: **25.545 t/s**
- cap 3: **26.991 t/s**

Measured economic deltas on the same pass:

- serial:
  - `accepted_mean`: **2.093338**
  - `rows_computed_mean`: **3.170583**
  - `verify_n_mean`: **2.359150**
- cap 5:
  - `accepted_mean`: **2.093338**
  - `rows_computed_mean`: **4.602689**
  - `verify_n_mean`: **2.359150**
- cap 4:
  - `accepted_mean`: **2.093338**
  - `rows_computed_mean`: **3.754008**
  - `verify_n_mean`: **2.282353**
- cap 3:
  - `accepted_mean`: **2.071116**
  - `rows_computed_mean`: **2.877641**
  - `verify_n_mean`: **2.097870**

Measured attribution pass means:

- serial:
  - draft **27.266 ms**
  - verify **29.123 ms**
  - total **82.348 ms**
- cap 4:
  - draft **25.839 ms**
  - verify **29.071 ms**
  - total **81.122 ms**
- cap 3:
  - draft **20.181 ms**
  - verify **28.577 ms**
  - total **74.928 ms**

Retained post-benchmark adversarial review:

- `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batchcap_postbench_review.md`

That review's judgement is the right narrow framing for this cycle:

- `cap5` is a local miss on this sample; it raises rows computed without buying
  back enough acceptance or verifier reduction
- `cap4` is the conservative positive: modestly faster than serial on this
  sample while preserving reported `accepted_mean`
- `cap3` is the stronger local candidate: fastest on this sample, with the
  gain coming mainly from lower draft-side cost rather than lower verifier cost
- `cap3` is **not** yet free to promote as a default, because one prompt
  (`dolly_0080`) showed a small acceptance/verified regression relative to
  serial and `cap4`

Interpretation:

- the new evidence does **not** support “batched scheduling is faster” in the
  abstract
- it does support a narrower claim: **shorter batched scheduled caps can beat
  both serial scheduling and cap-5 batching on the retained short sample**
- `cap4` currently looks like the safer local default candidate
- `cap3` looks like the higher-leverage candidate that now deserves a larger
  interleaved paired validation rather than another broad cap sweep
