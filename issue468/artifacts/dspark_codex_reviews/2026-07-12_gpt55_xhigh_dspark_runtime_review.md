# DSpark runtime adversarial review — gpt-5.5 xhigh

Date: 2026-07-12
Reviewer: `gpt-5.5` (`xhigh`, subagent)
Dispatcher verification: partial, focused on load-bearing claims
Scope:
- `ds4.c`
- `tests/ds4_test.c`
- `issue468/run_dspark_exactness_compare.py`
- `issue468/run_dspark_temp_distribution_compare.py`
- `issue468/run_dspark_corpus_bench.py`
- `issue468/run_dspark_phaseA_profile.py`
- `issue468/summaries/dspark_runtime_initial_benchmark.md`
- `issue468/summaries/confidence_scheduled_verification.md`
- `issue468/summaries/spec_speedup_model.md`

## Dispatcher verification

The following review claims were independently checked against the current tree and
confirmed before retention here:

- The DSpark speculative path currently performs normal target decode first, then
  CPU draft, then serial verification:
  [`ds4.c:28011`](../../../ds4.c#L28011),
  [`ds4.c:28024`](../../../ds4.c#L28024),
  [`ds4.c:28046`](../../../ds4.c#L28046).
- `decode_ms` includes `dspark_session_push_graph_hidden()` work because target
  decode calls it before the speculative branch records `dspark_t_after_commit`:
  [`ds4.c:27559`](../../../ds4.c#L27559),
  [`ds4.c:27595`](../../../ds4.c#L27595),
  [`ds4.c:28012`](../../../ds4.c#L28012).
- The current CPU draft path allocates `x_hc`, `next_hc`, and stage-local
  `win_kv` buffers per cycle, and copies the session KV ring into linear
  per-stage windows:
  [`ds4.c:27850`](../../../ds4.c#L27850),
  [`ds4.c:27855`](../../../ds4.c#L27855),
  [`ds4.c:27861`](../../../ds4.c#L27861).
- `dspark_block_forward_batch()` allocates a large set of temporaries on every
  call and hardcodes `n_tok = DS4_DSPARK_BLOCK`:
  [`ds4.c:27732`](../../../ds4.c#L27732),
  [`ds4.c:27740`](../../../ds4.c#L27740).
- The temp-parity test intentionally forces `DS4_DSPARK_VERIFY_K=0` and only
  accepts `ntok == 1` on the speculative path:
  [`tests/ds4_test.c:2455`](../../../tests/ds4_test.c#L2455),
  [`tests/ds4_test.c:2460`](../../../tests/ds4_test.c#L2460),
  [`tests/ds4_test.c:2500`](../../../tests/ds4_test.c#L2500).
- The corpus harness skip-checkpoints pre-existing per-prompt records, so mixed
  artifacts are possible if the output directory is reused:
  [`issue468/run_dspark_corpus_bench.py:126`](../../run_dspark_corpus_bench.py#L126).
- The greedy exactness harness compares emitted stdout bytes:
  [`issue468/run_dspark_exactness_compare.py:77`](../../run_dspark_exactness_compare.py#L77).

The remaining recommendations below are plausible engineering guidance from the
reviewer, but were not independently benchmarked or fully re-derived by the
dispatcher at retention time.

## Subagent review output

### Verdict per current claim

Mostly correct for the current path: CPU DSpark draft is the largest measured cost
when `verified=0`. But the attribution is too clean. The DSpark branch first runs
normal target decode, then CPU draft, then serial verification. Also, `decode`
timing includes DSpark hidden readback/push work, not just target decode.

So: CPU draft is first-order now, but after fixing/offloading it, the next
bottlenecks are likely serial verifier cost, target decode per cycle, and
host/device hidden transfer.

### Optimization opportunities

- Highest impact: replace `dspark_eval_draft_block_cpu()` with a DSpark-specific
  GPU graph. The CPU path allocates `x_hc`, `next_hc`, and per-stage `win_kv`
  every cycle, copies the whole DSpark KV window, and runs all 3 stages on host.
- High-feasibility CPU cleanup: add persistent DSpark scratch.
  `dspark_block_forward_batch()` allocates many large temporaries per stage, and
  the existing CPU decode path already demonstrates the no-malloc scratch pattern.
- Avoid duplicated window movement. Session state stores a 128-row ring, but
  each draft linearizes it into new `win_kv` buffers. Attention can instead
  consume the ring with wrap-aware indexing, or a persistent linear scratch can
  be updated incrementally.
- Stop always computing 5 draft rows when only fewer are useful. The CPU block
  hardcodes `n_tok = DS4_DSPARK_BLOCK`, even when fixed `verify_k=1` or remaining
  room is smaller.
- Fuse/top-only the output head. The current head loop runs 5 full vocab base
  projections, 5 Markov projections, materializes `base_logits` and
  `markov_bias`, then scans vocab on CPU.
- Reduce repeated weight decode/conversion. `confidence_proj` is copied /
  deconverted every draft cycle.
- Move hidden capture/mean/push off host. The target graph captures DSpark hidden
  tensors, then host code averages and projects them back into DSpark KV state.

### Best GPU migration path

Build a DSpark-specific graph path analogous to the existing target/MTP graph,
not scattered incremental kernels around the CPU routine.

The suggested staging is:

1. GPU hidden mean / `main_proj` / persistent DSpark KV update.
2. GPU 3-stage DSpark block body.
3. GPU head / Markov / confidence / argmax.
4. Re-integrate confidence scheduling on top of that path.

Partial offload of only the head or only MoE may help profiling, but the review's
position is that it leaves the fundamental host/device boundary in place.

### Benchmark / methodology risks

- Temp parity does not validate multi-token speculative commit because the test
  forces `verify_k=0`.
- “Verifier is cheap” is only true on miss-heavy cycles; accepted cycles still
  pay serial target decode work.
- Current timing buckets are not cleanly separated into pure target decode vs
  pure drafter time.
- The six-prompt corpus sample in the current summary is too small to support a
  broad throughput claim.
- Exactness is a smoke gate, not a full state-parity proof after every accepted
  speculative token.

### One recommended next implementation step

Implement an env-gated `metal_graph_eval_dspark_draft_block()` skeleton with
persistent DSpark graph tensors and parity it against
`dspark_eval_draft_block_cpu()` for draft token ids and confidence logits.
Start with GPU hidden mean / `main_proj` and persistent DSpark KV state, then
port the 3-stage block and head inside that same graph path.
