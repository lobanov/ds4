# DSpark runtime milestone 2 — progress

Date: 2026-07-12. Status: **active / in progress**.

Milestone 2 reframes DSpark runtime work around a narrower question: **how far can
the drafter be moved toward a GPU-resident execution shape before further CPU-local
tuning stops being strategically useful?** The retained CPU DSpark path remains
necessary, but mainly as a semantic oracle, policy reference, and measurement
substrate. The working goal is to preserve scheduled-verification semantics and
draft-quality economics while reducing host-side drafter work enough to make a
benchmarkable GPU-targeted path plausible.

## Goal

Milestone 2 should answer four practical questions.

1. What is the best **GPU cut line** for the current DSpark drafter under
   `ds4 --dspark <drafter.gguf>`?
2. Which parts of the present scheduled path are true policy requirements, and
   which are only artifacts of the current CPU reference implementation?
3. Can host-side drafter cost be reduced without degrading draft ids, confidence
   values, scheduled `verify_n`, accepted chunking, or final outputs?
4. If not yet, what exact parity-preserving migration step should be taken next,
   and how much of the remaining runtime problem still belongs to the verifier?

This is intentionally narrower than "make DSpark fast on CPU." The primary
strategic objective is to find and validate a migration path for the drafter, not
to spend many cycles polishing a placement that is unlikely to be final.

## Measurement methodology

This milestone still treats DSpark runtime work as a **two-axis benchmark
problem**: correctness of the final generated stream, and preservation of
speculative economics. The second axis matters because a runtime optimization can
preserve final tokens and sampled distributions while still degrading draft
quality, accepted-prefix length, or full-accept frequency enough to erase any
speed benefit.

The retained benchmarking contract has four layers.

1. **Final-output correctness gates remain mandatory.**
   Greedy output must remain byte-identical to baseline, and temp>0 parity must
   hold at the target logits / sampling-distribution level rather than only at
   emitted tokens.
2. **Draft-side quality must be checked separately from final-output parity.**
   Verifier exactness can hide a degraded drafter. For DSpark, runtime-valid
   optimizations therefore need draft-token parity, confidence-logit parity,
   scheduled `verify_n` parity, accepted-chunk parity, and DSpark window-state
   parity against the trusted reference, not just final text/logit agreement.
3. **Acceptance metrics are first-class outputs.**
   Each retained bench should continue to report at least drafted length, verify
   length, accepted/verified drafts per cycle, and where relevant the full-accept
   rate.
4. **Cycle-cost attribution must stay split by component.**
   Aggregate tokens/s is not enough to identify the next lever. The retained
   profile keeps `decode_ms`, `draft_ms`, `verify_ms`, `verify_decode_ms`, DSpark
   state-push time, and logits readback separate so verifier cost, drafter cost,
   and state-update cost do not get conflated.

Operational rules carried into this milestone:

- Use the same retained exactness, powered-corpus, and long-context corpora that
  already feed `issue468/summaries/spec_speedup_model.md`.
- Use `ds4_spec_bench.c` / `ds4-spec-bench` as the default DSpark benchmark
  substrate, extending it as needed rather than replacing it with many narrow
  drivers.
- Preserve batch submission from one bulk config, low-file-count output, and
  reuse of a single loaded engine while respecting the DS4 lock and using fresh
  sessions where needed.
- Do not validate an optimization by greedy exactness plus temp>0 parity alone if
  it changes draft acceptance, confidence values, or scheduled verify lengths.
- Compare both a fixed low-K reference (`verify_k=1`) and the scheduled path,
  because the scheduled path can lose for two different reasons: verifier cost or
  drafter execution cost.

## Carried-forward facts from milestone 1

These findings remain load-bearing.

1. The current DSpark runtime is correctness-gated and benchmarkable. Greedy
   exactness and the retained temp>0 distribution gate pass after the support-state
   and `main_proj` semantic fixes.
2. The current measured DSpark runtime is weaker than the earlier provisional
   optimism. With correct support-state advancement, throughput falls back toward
   roughly **~22–30 t/s** rather than the earlier more optimistic band.
3. The verifier is still an important ceiling term. Lead 06 made anchor reuse
   real, but the surviving exact path is still dominated by target GPU layer
   execution rather than by trivial host overhead.
4. Fixed `verify_k=1` remains the cleanest local DSpark reference point, but it is
   still not a positive DSpark speedup claim.
5. The current scheduled DSpark path still pays substantial token-serial draft
   overhead on the CPU reference path: per-row body/head/confidence work, host-side
   argmax, and stop/continue policy decisions are all paid inside each cycle.

## Why milestone 2 is GPU-targeted

The remaining CPU overhead is real, but not all of it is worth optimizing in
place. The right question is not "how much more CPU micro-optimization is
available?" but "which changes are still desirable if the drafter migrates toward
device residency?"

The target direction is a drafter shape with:

- persistent DSpark window/state on device;
- GPU body execution;
- GPU output-head path;
- GPU confidence-logit path;
- GPU draft selection / argmax or a minimal host-readback equivalent; and
- host control flow reduced to the few scalars needed to choose verify span and
  manage speculative scheduling.

This direction is justified because host-side drafter work is paid every
speculative cycle, while DSpark state-push and final-logits readback have already
measured as relatively small terms.

## Design constraints

Any milestone-2 migration step should preserve the following contract.

1. **Scheduled-verification semantics stay authoritative.**
   The migrated path must preserve draft ids, confidence logits, computed
   `verify_n`, accepted chunking, and final outputs against the trusted reference.
2. **The "confident prefix + 1 drafted token" policy remains intentional.**
   The point of scheduled DSpark is not to maximize full accepts; it is to draft
   one token past the confident prefix so the next cycle is less likely to pay a
   standalone anchor decode.
3. **Variable verify span matters more than preserving CPU row-by-row drafting as
   an implementation detail.**
   The value comes from confidence-informed verify truncation, not from the
   current token-serial host implementation itself.
4. **The benchmark substrate stays fixed while internals move.**
   `ds4-spec-bench`, the retained corpora, and the parity gates remain the default
   validation harness.

## Iteration protocol

Each optimization cycle should follow the same order.

1. Implement one GPU-relevant or parity-critical change.
2. Run smoke gates first:
   `--dspark-schedule-parity`, greedy exactness where applicable, and the temp>0
   logit/distribution gate.
3. If smoke passes, commission adversarial review before benchmarking.
4. Benchmark with `ds4-spec-bench` on the retained corpora.
5. Commission a second adversarial review on the measured result before
   summarizing it into the worklog and downstream summaries.
6. Update `issue468/inventories/ds4_env_and_instrumentation_inventory.md` if the
   iteration added, removed, or changed any instrumentation or env-gated runtime
   variation.
7. Update this document's `Next steps` section after the iteration completes or is
   abandoned.

## Retained artifacts

- greedy exactness: `issue468/artifacts/dspark_exactness_compare/summary.json`
- temp>0 logit/distribution parity:
  `issue468/artifacts/dspark_temp_distribution_compare/summary.json`
- short powered-corpus sample:
  `issue468/artifacts/dspark_corpus_bench/summary.json`
- long-prompt cycle profile:
  `issue468/artifacts/dspark_phaseA_profile/summary.json`
- scheduled batching cap sweep:
  `issue468/artifacts/dspark_corpus_bench/spec_sched_batchcap_v3_summary.json`
- adversarial optimization review:
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_dspark_opt_review.md`
- retained batch-cap reviews:
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batchcap_prebench_review_v2.md`
  and
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batchcap_postbench_review.md`
- reusable bulk speculative bench binary:
  `ds4-spec-bench` from `ds4_spec_bench.c`

## Current status

What is established now:

- DSpark has a retained correctness and benchmarking harness strong enough to
  support runtime migration work.
- Verifier cost is still a real ceiling, but scheduled drafter execution still
  has enough remaining host-side overhead that a purely `cap3` versus `cap4`
  tuning discussion is premature.
- The current CPU scheduled path is valuable as a policy oracle and migration
  reference even if it is not the final placement.

What is not established:

- a baseline-beating local DSpark runtime;
- the winning GPU cut line for DSpark drafter execution;
- whether the first useful migrated shape is fully scheduled DSpark, a staged
  hybrid, or a lower-K intermediate path.

## Next steps

This section is the current workload and recommendation snapshot. Update it after
each iteration.

1. Add direct instrumentation to localize why the current GPU-assisted scheduled
   output-head experiment diverges from CPU schedule parity, starting with rowwise
   CPU-vs-GPU base-logit comparison on identical DSpark hidden rows.
2. Use that instrumentation to decide whether the current GPU-head cut line is
   salvageable under exact parity or should be abandoned in favor of a different
   migration boundary.
3. Keep the CPU path as the semantic oracle and avoid broader CPU-local tuning
   unless the same structural change would still be wanted after GPU migration.
4. Preserve the explicit parity contract for draft ids, confidence logits,
   `verify_n`, accepted chunking, and DSpark window-state semantics before any new
   benchmark cycle.
5. Defer renewed `cap3` versus `cap4` scheduling-policy optimization until the
   drafter/verifier cost balance changes under a more device-resident execution
   shape.

## Worklog

### 2026-07-12 — milestone 2 note created

Created this milestone note by carrying forward the retained measurement
methodology and the load-bearing runtime findings from the DSpark runtime work so
far, then reframing the active goal around a GPU-targeted DSpark drafter path.
The CPU implementation is now treated primarily as a correctness oracle, policy
prototype, and benchmark reference.

### 2026-07-12 — first migration attempt: scheduled GPU-assisted output head failed parity

Tested an opt-in scheduled-drafter experiment that offloads the base output-head
projection from the CPU path under `DS4_DSPARK_SCHEDULE_GPU_HEAD=1`, while
retaining CPU-side confidence, Markov bias, and scheduling logic. The temp>0
logit/distribution parity gate still passed, but real `--dspark-schedule-parity`
on Metal failed: draft ids, confidence logits, computed `verify_n`, accepted
chunking, and final speculative progression diverged from the CPU reference. This
iteration is therefore not benchmarkable and does not yet justify review or
downstream model updates; the next task is to instrument CPU-vs-GPU row parity and
either tighten or abandon this cut line.
