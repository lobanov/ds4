# Lead 08 iteration 19: bounded-divergence contract and ledger reset

Date: 2026-07-18. Status: **contract REDESIGN; corrected audit COMMIT.** This iteration changes experiment admission, not runtime code or a
measured performance result.

## Decision and scope

The user accepts the same interim tradeoff used by Milestone 3: close sampled
distributions above temperature 0 and task-quality non-regression at
temperature 0. Exact greedy output remains the ultimate project criterion.
Lead 08 therefore stops reflexively extending the row-wise exactification
frontier and instead pursues a faster verifier inside the frozen M3 quality
envelope.

The independent challenger returned **REDESIGN, then PROCEED**. Its central
correction was to require fresh same-binary controls for every candidate rather
than treating historical M3 measurements as interchangeable with a new run.

## Reference and candidate compositions

- **Plain control:** target-only ds4, same binary, corpus, run order, and machine
  state used for the candidate.
- **M3 control:** the frozen full M3 composition (committing batch verifier,
  anchor reuse, prefix checkpoint, Metal drafter, and STS), with the proposed
  verifier optimization disabled.
- **Candidate:** the same M3 composition with exactly one verifier mechanism
  enabled.
- **Exact quality anchor:** sequential target verification on the same fixed
  rows used by the distribution probe.

The fresh matched controls are authoritative. Historical M3 values below are
outer caps and regression alarms, not permission to compare across unmatched
harnesses.

## Interim quality gates

### Temperature 0: task-quality non-regression

On the canonical 92-question gate the candidate must:

- score at least **61/92 and at least the fresh M3-control score**;
- agree with plain on at least **83/92** verdicts (90.2%) **and at least the
  fresh M3-control agreement**;
- introduce no more than **4** plain-PASS to candidate-FAIL changes **and no
  more than the fresh M3 control**; and
- report all candidate-versus-M3 verdict and output changes.

The retained sample is an empirical interim screen, not a powered proof of
semantic equivalence. If an optimization is claimed to be numerically neutral
relative to M3, it must additionally reproduce M3 tokens exactly.

Committed-token divergence against plain must be reported on the same fixed
prompts and lengths, but is not an admission metric under the user-defined
temperature-0 functional gate. Exceeding the matched M3 control or historical
56.6% benchmark / 40% exactness-corpus values is a mandatory investigation and
must be called out; a claim of numerical neutrality still requires token
identity to M3.

### Temperature above 0: fixed-frontier distribution non-regression

The retained M2 result used 90 cycles / 156 accepted positions on its own
trajectory; those are historical sample counts, not a reusable frontier. Every
candidate instead runs the batch verifier non-committing while exact sequential
verification owns the committed trajectory. The optimization-off M3 control
and candidate use identical workload metadata and must match on prompt IDs,
emitted tokens, cycle count, and every cycle's drafts, `verify_n`, accepted
count, confidence logits, and `n_compared`. This makes their cycle/row ordering
paired; any mismatch or restore failure is a hard integrity failure.

Reconstruct and run the paired workload as follows, substituting exactly one
candidate mechanism variable:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
jq -c '{id,mode,prompt_file,system,frontier_tokens,gen_tokens:.gen_tokens_requested,exclude_eos,seed,temperature,top_k,top_p,min_p}' \
  issue468/artifacts/dspark_m3_bench/distprobe_batched_vs_exact_fresh.jsonl \
  > /tmp/lead08_v14_dist_config.jsonl
run_probe() {
  label="$1"; shift
  env "$@" DS4_DSPARK_VERIFY_DIST_PROBE=1 DS4_DSPARK_VERIFY_BATCHED=0 \
    DS4_DSPARK_ANCHOR_REUSE=1 DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 \
    DS4_DSPARK_DRAFT_METAL=1 DS4_DSPARK_DRAFT_METAL_STS=1 \
    ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
      --bulk-config /tmp/lead08_v14_dist_config.jsonl \
      --jsonl-out "/tmp/lead08_v14_dist_${label}.jsonl"
}
run_probe m3
run_probe candidate DS4_V14_MECHANISM=1
```

The final variable name is mechanism-specific and must be recorded in that
iteration. After the pairing/integrity check, candidate-versus-exact metrics
must be no worse than the fresh M3-control values and must also stay within the
historical M2 envelope:

| Metric | Maximum admitted value |
|---|---:|
| argmax flip rate | 1/156 (0.641%) |
| mean TV | 0.0104 |
| median TV | 0.0035 |
| p90 TV | 0.0308 |
| max TV | 0.104 |
| fraction of compared cycles with TV > 0.05 | 3/90 (3.33%) |
| mean KL(seq || candidate) | 0.0022 |
| max KL(seq || candidate) | 0.054 |
| max absolute logit difference | 4.56 |

Only a small declared serialization tolerance may be used for scalar metrics.
A changed trajectory or cycle/row work pattern is an integrity failure, not a
substitute for the paired comparison.

### Runtime integrity

Restore, cache, or state failures are hard failures. Scheduler configuration,
acceptance, `verify_n`, expert work, accepted chunks, and correction tokens are
retained and reported so a speedup cannot be attributed silently to doing less
work.

## Economic gates

- A fixed-work prototype proceeds only with at least **15%** improvement on an
  identified hot stage and a credible path to at least **8.7 ms/cycle** composed
  saving.
- Integration requires K=4 `verify_ms <= 50.5 ms`, at least **8.7 ms/cycle**
  scheduled full-stack saving, and throughput **>= max(45.8 t/s, 1.20 x fresh
  plain)** on the 176-entry corpus.
- Quality, memory, replay, and state-integrity gates remain conjunctive with
  performance.

## Ledger consequence

V1 becomes the `INTERIM_BASELINE`. V5 and its V6-V12 evidence remain valid but
the exact-hybrid track is deferred. The former V13 exact-accumulation inventory
is also deferred. T1 becomes P0 counter attribution now that full Xcode exposes
Metal counters, and new V14 owns M3-bounded verifier acceleration.

The relaxed contract does **not** revive the failed grouped gate/up, address
layout, thread geometry, mapped residency, shared M1/MK, top-r, or margin-guard
mechanisms. Their economic or structural falsifiers are independent of exact
output preservation. D5 becomes only a conditional later composition; it
cannot close the verifier gap by itself.

## Ranked next experiment

Capture matched warm fixed-K4 M3 verification and M1 decode with Metal GPU
counters and no dump/stage-sync instrumentation. Attribute top kernels by GPU
duration, DRAM/read bandwidth, cache behavior, ALU utilization, occupancy or
stalls, and GPU idle/dispatch gaps. Counters are a selector, not a GO:

- bandwidth/cache-bound evidence selects one distinct traffic/cache mechanism;
- ALU/dequant-bound evidence selects one quant-kernel mechanism;
- occupancy/barrier evidence selects one register/threadgroup mechanism;
- idle/dispatch evidence selects one encode/fusion mechanism; and
- unavailable or ambiguous counters require a controlled separator benchmark,
  not an attribution claim.

Only the selected mechanism may advance to one bounded fixed-work prototype.

The first mandatory post-iteration audit returned **NO-COMMIT** because the
draft contract did not make fresh-M3 functional gates conjunctive, treated the
historical trajectory as reusable, allowed the absolute or relative throughput
gate instead of both, and contradicted itself on committed-token divergence.
After the corrections above, the repeat audit returned **COMMIT** with no
remaining blockers.

## Evidence basis

- `summaries/dspark_runtime_milestone_2_progress.md`: fixed-frontier batch versus
  exact distribution measurements.
- `summaries/dspark_runtime_milestone_3_progress.md`: full-stack performance and
  retained functional-quality methodology.
- `artifacts/lead08_reassessment/01_reorient_bar_redrivation.md`: live-cycle
  prize and verifier dominance.
- `summaries/solution_space_ledger.md`: mechanism closures and ranked queue.
