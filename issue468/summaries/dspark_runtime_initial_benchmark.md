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

## Known optimization avenues

The current measurements and implementation review narrow the live optimization
space to a shorter list.

1. **Keep verifier work as the primary ceiling.** The new timing-detail split
   makes the current bottleneck picture more precise: most of `verify_ms` is
   serial target decode inside the exact verifier, not DSpark support-state
   push or logits readback. This keeps Lead 08 Phase B fused low-K verifier
   work live.
2. **Move more of DSpark execution onto the GPU, but with confidence and state
   carried along.** The live drafter still pays for CPU body/head execution and
   host-side DSpark window handling. The remaining drafter-side path therefore
   points to persistent device DSpark KV/window state plus GPU body, head,
   markov, confidence, and argmax. Host policy can remain on CPU because the
   scheduling decision itself is only over a few scalars.
3. **Treat variable verify span as the more important scheduled-DSpark lever
   than token-serial early-stop drafting.** The current scheduled path drafts
   token-by-token so it can stop as soon as cumulative survival drops below the
   STS threshold. That preserves the intended policy shape, but it makes the
   drafter expensive. The likely better next runtime design is to batch a full
   short block (`4` or `5` tokens), produce draft ids plus confidence logits in
   one pass, and then let confidence choose how much of that block to verify.
   That keeps the main scheduling benefit while avoiding much of the current
   token-serial draft overhead.
4. **Preserve the existing “confident prefix + 1 drafted token” policy logic
   even if drafting becomes batched.** The current scheduled implementation is
   deliberately asymmetric: it can draft one token beyond the confident prefix
   while verifying only the confident prefix, which is meant to reduce the
   frequency of full-block accepts and therefore the need to pay the standalone
   anchor decode on the next cycle. If scheduled drafting is reworked around
   batched full-block drafting, the verify-length chooser should preserve this
   economics rather than collapsing into plain fixed-K.
5. **Do not spend more time on drafter-weight precision or scheduler-only
   tuning.** Earlier lead work already closed these as primary levers. The
   vendored Q4_K drafter is already effectively at its source-weight ceiling,
   and confidence scheduling by itself was only marginal secondary material
   under cheaper verifier assumptions. The live question is execution cost, not
   whether the confidence signal exists.

## Current recommendation

**Proceed with DSpark optimization work, but do not headline a positive local
runtime result and do not treat a full GPU drafter body/head port as the first
obvious next step.**

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

The next engineering step should therefore be:

1. keep the new `ds4-spec-bench` path and retained corpus/profile gates as the
   measurement substrate;
2. parity-gate DSpark window-state semantics explicitly; and
3. investigate verifier/state-update economics before committing to a full
   DSpark GPU body/head port.

The current DSpark path is real and measurable, but the cleaned-up evidence now
points to **verifier and state-update economics as the next load-bearing issue**,
not just “move the drafter to GPU and the rest will probably work out.”
