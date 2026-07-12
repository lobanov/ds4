# DSpark runtime path — initial benchmark

Date: 2026-07-11. Status: **partial / provisional**. This note records the
first end-to-end `ds4 --dspark` runtime result after wiring the retained Q4_K
DSpark drafter GGUF into the engine with confidence scheduling and exact target
verification. It is **not** the final dossier verdict: the current path uses a
CPU drafter body and only a small first-pass corpus sample has been benchmarked.

Artifacts:

- greedy exactness: `issue468/artifacts/dspark_exactness_compare/summary.json`
- temp>0 logit/distribution parity: `issue468/artifacts/dspark_temp_distribution_compare/summary.json`
- short powered-corpus sample: `issue468/artifacts/dspark_corpus_bench/summary.json`
- long-prompt cycle profile: `issue468/artifacts/dspark_phaseA_profile/summary.json`

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
  - baseline: **40.04 t/s**
  - DSpark: **14.84 t/s**
  - range: **11.00 .. 15.79 t/s**

**Temp>0 logit/distribution parity: PASS under the current conservative parity mode.**

- 640 total sampled steps, 574 eligible DSpark steps.
- `max_abs = 0.0`, `rms = 0.0`, `sampled_lp_diff = 0.0`

Important caveat: the current DSpark temp-parity gate forces
`DS4_DSPARK_VERIFY_K=0`, so it validates the speculative entry path with DSpark
enabled while preventing extra speculative commits beyond the sampled target
token. This is the right conservative gate for the current implementation, but
it is narrower than a full multi-token speculative parity proof.

## Throughput result

**The current DSpark runtime path is well below baseline on the first powered
sample.**

On the first 6 Stage-2 prompts (`n=32`):

| impl | mean gen t/s | range | mean verified tokens/cycle |
|---|---:|---:|---:|
| baseline | **39.00** | 38.04 .. 39.75 | 0.000 |
| DSpark default scheduling | **20.49** | 15.94 .. 26.21 | 0.902 |
| DSpark fixed `verify_k=1` | **19.14** | 16.57 .. 21.75 | 0.380 |

So the current DSpark path is roughly **0.53x baseline** on this small powered
sample. Default confidence scheduling is slightly better than fixed `verify_k=1`
because a few prompts do realize multi-token verified prefixes, but the gain is
small relative to the overall slowdown.

## Cycle-cost profile

The long-prompt profile shows the current bottleneck clearly:

- `code_8k`, scheduled:
  - gen **15.96 t/s**
  - median verified **0.0**
  - median cycle total **62.41 ms**
  - median decode **28.08 ms**
  - median draft **34.05 ms**
  - median verify **0.009 ms**
- `synthesis_8k`, scheduled:
  - gen **16.22 t/s**
  - median verified **0.0**
  - median cycle total **62.51 ms**
  - median decode **28.06 ms**
  - median draft **33.51 ms**
  - median verify **0.009 ms**
- `grounded_8k`, scheduled:
  - gen **16.13 t/s**
  - median verified **0.0**
  - median cycle total **61.83 ms**
  - median decode **28.42 ms**
  - median draft **33.40 ms**
  - median verify **0.009 ms**

The special case where verification starts to matter is `synthesis_8k` with
fixed `verify_k=1`:

- gen **19.68 t/s**
- median verified **1.0**
- median cycle total **87.04 ms**
- median decode **28.40 ms**
- median draft **33.05 ms**
- median verify **27.27 ms**

Interpretation:

- When the DSpark cycle dies immediately (`verified = 0`), the path still pays
  about **28 ms** for the committed target token plus about **33–34 ms** for the
  CPU drafter body/head. That alone lands near the observed **~16 t/s** regime.
- When the path does verify an extra token, the verifier adds another
  **~20–30 ms** on top of the same decode+draft cost.
- So the **first-order bottleneck is the CPU draft pass**, not the current
  verifier. The verifier is still relevant on prompts where DSpark survives past
  the first draft token, but it is not the main reason the current runtime path
  misses baseline so badly.

## Current recommendation

**Do not treat the current DSpark runtime path as evidence for a worthwhile local
speedup.** The path is now correctness-gated and benchmarkable, which is the
important engineering milestone, but the observed economics are poor:

- exactness is preserved on the retained greedy gate,
- temp>0 logits/distribution are preserved in the current conservative parity
  mode,
- but throughput is still roughly **half of baseline** on the first powered
  sample and around **16–20 t/s** on retained long prompts.

The next optimization priority is therefore straightforward:

1. move the DSpark drafter body/head off the CPU fast path, or otherwise cut the
   **~33–36 ms/cycle** draft cost materially;
2. only then revisit whether confidence scheduling and low-K verification policy
   tuning can recover meaningful additional speedup.

Until that draft-cost problem is addressed, the current DSpark path does **not**
justify proceeding directly to wider benchmark sweeps or to an updated global
speedup claim in `spec_speedup_model.md`.
