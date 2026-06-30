# DSpark Q4_K + Imatrix Assignment

Date: 2026-06-30.

Purpose: turn the DSpark drafter into an imatrix-calibrated `Q4_K` GGUF and
validate that it improves drafter quality enough to justify shipping.

This note is the working assignment brief for the current implementation pass.
It records the fixed success criteria, what is already working, what was
measured on the generic imatrix path, and what the next implementation pass
should do instead.

---

## 1. Fixed success target

User steer for this pass:

1. Expected drafter quality improvement target: **+5% average acceptance**
2. Measurement sweep: average across context lengths
   `8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k`
3. The imatrix collector must be **drafter-specific**
4. Reusing the target model's `blk.*` routed-expert imatrix is **not acceptable**

Interpretation for this assignment:

- The acceptance metric is the DSpark drafter acceptance metric used by the
  existing research harness, not target-model NLL.
- The new drafter GGUF must stay on the existing `Q4_K` routed-expert kernel
  path; imatrix may change quantization weighting, but not the runtime tensor
  type.
- Exit is based on measured acceptance uplift, not just a successful build.

---

## 2. What is already working

### 2.1 Drafter-specific collector in `ds4`

Implemented in `ds4.c`:

- `--imatrix-out` now branches to a DSpark-specific collector whenever
  `--dspark` is loaded.
- The collector captures target `main_hidden` at L40/41/42, runs the DSpark
  input stage, seeds anchor KV for each DSpark block, runs the 3 drafter
  blocks, and accumulates routed-expert statistics from the drafter path.
- The saved imatrix entries are emitted from `e->dspark_weights.block`, so the
  file names are `mtp.*`, not `blk.*`.

Validated on Metal with:

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /tmp/dspark-test.imatrix.dat \
  --imatrix-max-prompts 1 \
  --imatrix-max-tokens 4 \
  --ctx 4096
```

Observed result:

- collector completed successfully
- wrote `/tmp/dspark-test.imatrix.dat`
- 1 prompt, 4 anchors, 360 routed expert observations

The file structure was inspected directly:

- entry count: `9`
- names:
  - `mtp.0.ffn_gate_exps.weight`
  - `mtp.0.ffn_up_exps.weight`
  - `mtp.0.ffn_down_exps.weight`
  - `mtp.1.ffn_gate_exps.weight`
  - `mtp.1.ffn_up_exps.weight`
  - `mtp.1.ffn_down_exps.weight`
  - `mtp.2.ffn_gate_exps.weight`
  - `mtp.2.ffn_up_exps.weight`
  - `mtp.2.ffn_down_exps.weight`

This proves the collector is now drafter-scoped.

### 2.2 Quantizer accepts DSpark imatrix entries

`gguf-tools/deepseek4-quantize` already supports DSpark expert tensors:

- it parses `mtp.N.ffn_*_exps.weight`
- it maps them to HF `mtp.N.ffn.experts.<expert>.*`
- it slices packed per-expert imatrix entries correctly

Validated with a real single-tensor regeneration:

```sh
gguf-tools/deepseek4-quantize \
  --hf /Users/lobanov/Projects/ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /tmp/dspark-imatrix-q4k.gguf \
  --imatrix /tmp/dspark-test.imatrix.dat \
  --compare-tensor mtp.0.ffn_gate_exps.weight \
  --compare-gguf ../ds4/gguf/dspark.gguf
```

Observed result:

- imatrix loaded successfully
- `mtp.0.ffn_gate_exps.weight` regenerated as `q4_K`
- tensor generation completed without mapping or size errors
- byte compare failed, which is expected for a changed quantization weighting

### 2.3 Full pilot DSpark imatrix GGUF build works

Validated with:

```sh
gguf-tools/deepseek4-quantize \
  --hf /Users/lobanov/Projects/ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /tmp/dspark-q4k-imatrix-pilot.gguf \
  --overwrite \
  --imatrix /tmp/dspark-test.imatrix.dat
```

Observed result:

- build completed successfully
- wrote `/tmp/dspark-q4k-imatrix-pilot.gguf`
- `81` tensors
- same tensor types as the template (`type_changes: 0`)

### 2.4 The pilot GGUF loads and runs as a drafter

Validated with:

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /tmp/dspark-q4k-imatrix-pilot.gguf \
  -c 4096 -n 8 --temp 0.0 \
  -p 'Briefly say hello.'
```

Observed result:

- DSpark drafter loaded from the pilot GGUF
- generation completed successfully
- no load or runtime-format errors

This proves the full artifact path is now live:

`HF DSpark -> drafter-specific imatrix -> Q4_K+imatrix DSpark GGUF -> ds4 runtime`

### 2.5 Acceptance sweep harness is now reusable

Additional tooling is now in place:

- `issue468/baseline/dspark_capture/measure_metal_b2.py`
  - rewritten as a parameterized Metal B2 evaluator
  - consumes arbitrary DSpark q-dumps, target top-k JSON, and greedy-token JSON
  - writes structured JSON summaries
- `issue468/run_dspark_context_acceptance_sweep.py`
  - builds exact rendered prompts at target token frontiers
  - captures target hidden states and target top-k distributions once
  - runs baseline and candidate DSpark probes on the same capture bundle
  - writes per-context JSON/TSV summaries

This tooling has already been validated at:

- `512` prompt tokens (smoke run)
- `8192` prompt tokens (real pilot)

### 2.6 Balanced full-corpus render order now exists

The original dataset builder wrote `rendered_prompts.txt` in plain
`(category, source, mode)` sort order, which front-loaded the file with
`agent` prompts and delayed large parts of the corpus until very late.

That made partial-budget imatrix collection behave like a biased sub-corpus.

The builder now writes:

- the same prompt inventory
- the same `prompts.jsonl` records
- but `rendered_prompts.txt` in a deterministic balanced
  category/mode round-robin order

This means early collection budgets now see:

- `long_context`
- `source`
- `agent`
- `language`
- `eval_reasoning`
- `translation`
- `programming`
- `general`
- `algorithms`

within the first cycle of prompts, instead of mostly `agent`.

### 2.7 Generic-path pilot evidence

Using a small imatrix candidate built from only `256` anchors on the shared
model prompt corpus, the original 8k pilot result was:

- greedy avg prefix: `4.26 -> 4.37` (`+2.58%`)
- B2 accepted: `4.0263 -> 4.0526` (`+0.65%`)
- B2 committed: `4.4211 -> 4.4704` (`+1.12%`)

This was below the assignment gate and should be treated only as an early
pilot.

Stronger same-corpus balanced-order candidates were then measured:

1. `32768` anchors, balanced order, `512` per-prompt cap:
   - candidate GGUF: `/tmp/dspark-q4k-imatrix-balanced-32768.gguf`
   - 8k pilot:
     - greedy avg prefix: `4.26 -> 4.21` (`-1.17%`)
     - B2 accepted: `4.0929 -> 4.0950` (`+0.05%`)
     - B2 committed: `4.5045 -> 4.5095` (`+0.11%`)
2. `16384` anchors, balanced order, uncapped per prompt:
   - candidate GGUF: `/tmp/dspark-q4k-imatrix-balanced-16384-uncapped.gguf`
   - 8k pilot:
     - greedy avg prefix: `4.26 -> 4.26` (`+0.00%`)
     - B2 accepted: `4.0929 -> 4.0716` (`-0.52%`)
3. `65536` anchors, balanced order, `512` per-prompt cap:
   - candidate GGUF: `/tmp/dspark-q4k-imatrix-balanced-65536.gguf`
   - 8k pilot:
     - greedy avg prefix: `4.26 -> 4.11` (`-3.52%`)
     - B2 accepted: `4.0929 -> 4.0435` (`-1.21%`)

Interpretation:

- the balanced render order is a real corpus/tooling improvement
- it improves early coverage from the same prompt inventory
- but the first three real candidate GGUFs still do **not** show the expected
  quality lift at `8k`
- the largest same-corpus balanced candidate is worse than baseline at the `8k`
  pilot
- so better prompt ordering and more same-corpus anchors alone are not enough
  to satisfy the assignment gate

---

## 3. Research expectation to carry forward

Two prior estimates matter:

1. `issue468/28_final_verdict_outcome_a.md`
   recorded a prior estimate of **`Q4_K+imatrix: +6% improvement`**
2. `gguf-tools/imatrix/README.md`
   recorded a measured target-model Q4 imatrix improvement of about **`-1.95%`
   avg NLL** on 100 official continuations

The assignment target for this pass is therefore stricter and more specific
than the old heuristic:

- previous DSpark expectation: roughly `+6%`
- current required gate: **`>= +5%` average acceptance**

That makes the main remaining task a measurement problem, not a tooling
problem.

---

## 4. Remaining work

The evidence above changes the next pass. The generic same-corpus imatrix path
is now implemented and measured, but it has failed to improve the `8k` pilot
acceptance metric. The next implementation pass should therefore target
**acceptance-relevant imatrix collection**, not more generic prompt traversal.

### 4.1 New goal: acceptance-targeted imatrix

Build an **acceptance-targeted imatrix** collection path for the DSpark
drafter, quantize a new `Q4_K` drafter GGUF from that imatrix, and validate
whether it improves DSpark drafter quality by **at least `+5%` average
acceptance** across:

- `8k`
- `16k`
- `24k`
- `32k`
- `40k`
- `48k`
- `56k`
- `64k`

Core hypothesis:

- generic imatrix optimizes average activation preservation
- DSpark acceptance is dominated by a sequential Markov head
- small early-token errors compound across the drafted suffix
- acceptance-targeted collection may allocate the fixed `Q4_K` precision budget
  toward the states that actually decide accepted prefix length

This remains a measurement hypothesis, not a guaranteed fix.

### 4.2 Why the generic path is no longer enough

The strongest same-corpus balanced candidates now show:

- improved expert coverage
- but no positive `8k` pilot signal
- and at `65536` anchors, outright regression vs baseline

So the likely problem is now an **objective mismatch**, not just insufficient
anchor count:

- the collector is seeing real drafter activations
- but it is still averaging over many states that are not verifier-critical
- the acceptance metric is much harsher than NLL or generic greedy overlap

### 4.3 Acceptance-relevant data collection plan

Instrument the drafter at the **acceptance boundary**, not at generic prompt
decode.

For each speculative cycle, collect routed-expert inputs from the actual DSpark
path at the states that determine whether draft tokens are accepted.

Target states:

1. `draft[0]`
2. `draft[1]`
3. optionally `draft[2..]`

with priority on early draft positions because:

- `draft[0]` errors are most likely to poison the whole drafted suffix
- `draft[1]` errors are next most important
- later positions matter, but the sequential head makes them downstream of the
  earlier ones

Data should be gathered on:

- acceptance-sweep prompt frontiers
- teacher-forced target continuations
- hard / low-margin cases where baseline Q4_K is close to verifier rejection

### 4.4 Proposed instrumentation

Reuse the existing DSpark imatrix collection site in `ds4.c`:

- after target `main_hidden` capture
- after DSpark input stage
- inside each DSpark block at the routed-expert collection point already used
  by the current drafter-specific collector

This is the correct tensor surface already. The key change is **when** samples
are taken, not **what** tensor is sampled.

Add a new collection mode conceptually like:

```text
--imatrix-acceptance-bundle DIR
```

where `DIR` is generated by the acceptance harness and contains:

- exact prompt frontier
- target hidden states
- target greedy tokens
- target top-k distributions

Collection behavior:

1. load the prompt frontier
2. reproduce the DSpark drafter state at the verifier frontier
3. run the drafter in **teacher-forced** mode on the target continuation
4. at each drafted position of interest, collect routed-expert inputs for all 3
   DSpark layers
5. accumulate weighted imatrix statistics

Teacher forcing matters because:

- free-running collection lets drafter mistakes alter later states
- acceptance-targeted imatrix should see the states on the **correct target
  trajectory**

Position weighting plan:

- collect separate statistics by:
  - layer
  - expert
  - draft position
- merge with explicit weights at save time

Initial weighting proposal:

- position 0: highest weight
- position 1: medium weight
- position 2+: lower weight

Hard-case emphasis:

- preferentially sample or upweight cases where:
  - baseline accepted prefix is short
  - target-vs-drafter logit margin is small
  - verifier rejects early

### 4.5 Generic-path evidence retained for reference

Current measured coverage on the shared model prompt corpus:

`2048` anchors, `3` prompts, `184320` routed expert observations:

```text
layer 0: mean 240, min 0, p10 0, p50 44, p90 580,  p99 2876, max 5131, zero  67
layer 1: mean 240, min 0, p10 0, p50 41, p90 686,  p99 1686, max 7049, zero  80
layer 2: mean 240, min 0, p10 0, p50  7, p90 726,  p99 2728, max 3729, zero 112
```

`16384` anchors, `11` prompts, `1474560` routed expert observations:

```text
layer 0: mean 1920, min 0, p10 0, p50 285, p90 4402, p99 25801, max 45355, zero  56
layer 1: mean 1920, min 0, p10 0, p50 425, p90 5732, p99 12761, max 42645, zero  68
layer 2: mean 1920, min 0, p10 0, p50  47, p90 5109, p99 22261, max 30153, zero 101
```

Same-corpus anchor redistribution at the same `16384` total anchors:

`512` anchors per prompt cap, `32` prompts, `1474560` routed expert observations:

```text
layer 0: mean 1920, min 0, p10 0, p50 374, p90 4982, p99 28690, max 40098, zero  66
layer 1: mean 1920, min 0, p10 0, p50 342, p90 5238, p99 12736, max 59408, zero  78
layer 2: mean 1920, min 0, p10 0, p50  44, p90 5384, p99 23256, max 42380, zero 110
```

`256` anchors per prompt cap, `64` prompts, `1474560` routed expert observations:

```text
layer 0: mean 1920, min 0, p10 0, p50 304, p90 4730, p99 25856, max 40332, zero  76
layer 1: mean 1920, min 0, p10 0, p50 360, p90 5436, p99 15220, max 66132, zero  84
layer 2: mean 1920, min 0, p10 0, p50  38, p90 5896, p99 30194, max 52600, zero 118
```

Balanced full-corpus order, same prompt inventory:

`16384` anchors, `512` per-prompt cap, `64` prompts, `1474560` routed expert observations:

```text
layer 0: mean 1920, min 0, p10 0, p50 458, p90 4015, p99 26583, max 43801, zero  53
layer 1: mean 1920, min 0, p10 0, p50 483, p90 4890, p99 15873, max 50148, zero  67
layer 2: mean 1920, min 0, p10 0, p50  99, p90 5352, p99 24899, max 32390, zero 103
```

`16384` anchors, balanced order, uncapped per prompt, `22` prompts:

```text
layer 0: mean 1920, min 0, p10 0, p50 373, p90 4452, p99 24984, max 47061, zero  52
layer 1: mean 1920, min 0, p10 0, p50 519, p90 5431, p99 17702, max 44625, zero  68
layer 2: mean 1920, min 0, p10 0, p50  84, p90 5531, p99 28996, max 40121, zero 103
```

`32768` anchors, balanced order, `512` per-prompt cap, `133` prompts, `2949120` routed expert observations:

```text
layer 0: mean 3840, min 0, p10 0, p50  876, p90  8552, p99 48353, max  87378, zero  52
layer 1: mean 3840, min 0, p10 0, p50 1031, p90 10271, p99 31840, max 101796, zero  65
layer 2: mean 3840, min 0, p10 0, p50  233, p90 10849, p99 55338, max  66432, zero 101
```

`65536` anchors, balanced order, `512` per-prompt cap, `274` prompts, `5898240` routed expert observations:

```text
layer 0: mean 7680, min 0, p10 0, p50 1699, p90 16756, p99  94776, max 175940, zero  51
layer 1: mean 7680, min 0, p10 0, p50 2118, p90 19602, p99  62915, max 207804, zero  63
layer 2: mean 7680, min 0, p10 0, p50  433, p90 110799, p99 140794, max 140794, zero  99
```

Interpretation:

- total observation count scales as expected
- prompt diversity can be increased materially on the same corpus
- balanced full-corpus ordering is better than the original front-loaded order
  for early-budget coverage
- but the long tail remains poor:
  - `p10` is still `0` on all layers
  - even at `65536` anchors, layer 2 still has `99 / 256` zero-hit experts
- improving same-corpus coverage alone is not translating into better 8k pilot
  acceptance on the measured candidates

### 4.6 Validation plan

Before building a new GGUF:

1. prove the acceptance collector emits drafter-specific `mtp.*` entries
2. verify that teacher-forced / position-aware collection runs end-to-end on
   real acceptance bundles
3. dump diagnostics showing coverage by:
   - layer
   - expert
   - draft position
4. compare those distributions against the generic corpus collector

### 4.7 Build the real Q4_K+imatrix DSpark GGUF

Use the real imatrix to build a new drafter GGUF from:

- HF source: `/Users/lobanov/Projects/ds4/hf-dspark`
- template: `../ds4/gguf/dspark.gguf`

The output should be treated as the evaluation candidate, not the smoke
artifact.

### 4.8 Measure acceptance uplift

Need a baseline-vs-new comparison using the existing DSpark acceptance harness.

Baseline:

- `../ds4/gguf/dspark.gguf`

Candidate:

- new Q4_K+imatrix DSpark GGUF

Need acceptance runs at:

- `8k`
- `16k`
- `24k`
- `32k`
- `40k`
- `48k`
- `56k`
- `64k`

Need final report:

- per-context acceptance
- average acceptance across all 8 contexts
- delta vs baseline

Current status:

- reusable measurement tooling exists
- multiple real pilots at `8k` now exist, including a `65536`-anchor balanced
  candidate
- the next candidate should be validated at `8k` first
- only proceed to the full `8k..64k` sweep if the `8k` pilot is clearly
  positive

### 4.9 Decide ship/no-ship

If the measured average uplift is below `+5%`, the assignment fails its current
gate even if the GGUF is valid and loadable.

---

## 5. Exit criteria

This assignment is complete only if all of the following are true:

1. An acceptance-targeted imatrix collector exists and is reproducible.
2. The collector gathers drafter-specific `mtp.*` entries from the real DSpark
   runtime path.
3. The collector uses acceptance-relevant data:
   - acceptance frontiers
   - teacher-forced target continuation states
   - explicit draft-position handling
4. A new DSpark drafter GGUF has been built from that imatrix with routed
   experts still quantized as `Q4_K`
5. The new GGUF loads and runs in `ds4` as a DSpark drafter
6. Acceptance has been measured for baseline and candidate at
   `8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k`
7. The candidate achieves **`>= +5%` average acceptance uplift** across those 8
   context lengths
8. The measurement commands, artifact paths, and summary numbers are written
   down in `issue468`
9. Before spending on the full 8-context sweep, the `8k` pilot must be clearly
   positive; if acceptance-targeted collection still fails there, stop and
   reassess before expanding the sweep

Anything short of item 7 is not a success for this assignment.

---

## 6. Current recommendation

Do not treat the current generic same-corpus imatrix as production-quality yet.

The next decision point is:

- either implement acceptance-targeted imatrix collection
- or accept that the current generic imatrix path is not aligned enough with the
  metric to keep iterating productively

Given the current measured coverage shape, the failed same-corpus
redistribution experiments, and the failed `65536`-anchor balanced pilot, the
default technical recommendation is to pursue acceptance-targeted imatrix
collection before spending more time on wide acceptance sweeps.

---

## 6. Immediate next commands

### Collector implementation target

Add a bundle-driven collection path conceptually like:

```text
--imatrix-acceptance-bundle DIR
```

and drive it from the existing acceptance harness artifacts.

### Build an acceptance-targeted drafter imatrix

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  --imatrix-acceptance-bundle /tmp/dspark_acceptance_bundle \
  --imatrix-out /tmp/dspark-real.imatrix.dat \
  --ctx 32768
```

### Build the real candidate drafter GGUF

```sh
gguf-tools/deepseek4-quantize \
  --hf /Users/lobanov/Projects/ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /tmp/dspark-q4k-imatrix.gguf \
  --overwrite \
  --imatrix /tmp/dspark-real.imatrix.dat
```

### Load sanity check

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /tmp/dspark-q4k-imatrix.gguf \
  -c 4096 -n 8 --temp 0.0 \
  -p 'Briefly say hello.'
```

### Acceptance evaluation sweep

Run `8k` first, and only proceed to the full sweep if the pilot is clearly
positive.

Then use the existing DSpark acceptance / evaluation harness from `issue468`
and record baseline vs candidate at:

- `8k`
- `16k`
- `24k`
- `32k`
- `40k`
- `48k`
- `56k`
- `64k`

---

## 7. Current status

Status today:

- collector path: **working**
- imatrix naming/layout: **working**
- quantizer DSpark imatrix ingestion: **working**
- full DSpark imatrix GGUF build: **working**
- runtime load sanity: **working**
- balanced full-corpus dataset ordering: **working**
- 8k pilot quality checks on three stronger candidates: **done, all below target**
- acceptance-targeted imatrix goal + instrumentation plan: **defined**
- real quality evaluation against the `+5%` gate across all 8 contexts: **not done yet**

So the project has moved from "tooling uncertain" to "acceptance-targeted
collection plan defined."
