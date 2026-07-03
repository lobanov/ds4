# Temp-aware exact DSpark for T>0 — temp=0 guard, implementation, and sub-1.0 benchmark

Date: 2026-07-03.
Branch: `dspark`.

## Lead

Investigate whether DSpark can run in a **sub-1.0 temperature regime** that is both:

1. **distribution-exact** relative to plain target sampling at the same temperature, and
2. **faster** than the current `temp=1.0` DSpark path.

This required two parts:

- first, explicitly reject `--dspark` at `temp=0`, because the current B2 path is not greedy-exact
- second, make the DSpark B2 implementation temperature-aware for **all `T > 0`** rather than only behaving coherently at `T=1`

## Independent verification before implementation

Re-read the live DSpark path in `ds4.c`, plus the CLI/eval call sites.

### What was true before this patch

- `ds4_session_sample(...)` respected the requested temperature for the **first anchor token**
- but `ds4_session_eval_dspark_b2(...)` did **not** take `temperature`, `top_p`, or `min_p`
- B2 acceptance used `softmax(raw_logits)` rather than `softmax(raw_logits / T)`
- `DS4_DSPARK_TARGET_POS0` sampled position 0 from the unscaled target logits
- suffix drafts were effectively generated from a greedy / argmax Markov chain, not from the temperature-conditioned drafter distribution
- reject correction used the old `argmax(p-q)` approximation rather than a temperature-conditioned residual sampler

So the current implementation was only aligned with the intended DSpark exactness regime at `temp=1.0`.

### Existing evidence to preserve

Earlier notes already established:

- B2 exactness claims in this project are specifically about **stochastic rejection sampling**, not greedy temp=0 decode
- `temp=0` greedy-exactness had already been investigated historically and was not achieved
- the current bounded performance problem is verifier-dominated, so any sub-1.0 gain would likely come from **better acceptance**, not cheaper verifier math

## Code changes made

### 1. Disallow `--dspark` at `temp=0`

Added an explicit CLI/eval guard:

- `ds4_cli.c`
- `ds4_eval.c`

Behavior now:

- if `--dspark` is specified with `--temp <= 0`, the program exits with a clear error

Example message:

```text
--dspark requires --temp > 0 (current DSpark exactness is defined only for stochastic decoding)
```

This prevents accidental use of DSpark in a regime where the current implementation is not exact.

### 2. Thread temperature/filter params into DSpark B2

Changed the DSpark B2 API to take:

- `float temperature`
- `float top_p`
- `float min_p`

Updated call sites in:

- `ds4_cli.c`
- `ds4_eval.c`
- declaration in `ds4.h`

### 3. Temperature-aware proposal sampling for `T > 0`

Inside `ds4_session_eval_dspark_b2(...)`:

- position 0 target override now samples using the requested `temperature`, `top_p`, and `min_p`
- suffix drafts are sampled from the temperature-conditioned drafter distribution instead of being treated as a pure argmax-only chain in the final exact path

### 4. Temperature-aware B2 acceptance

Added a helper that materializes the **full filtered probability distribution** under:

- temperature
- top-p
- min-p

This is then used to compute:

- `q_T(x)`
- `p_T(x)`
- acceptance probability `min(1, p_T(x) / q_T(x))`

for the actual runtime temperature.

### 5. Temperature-aware reject correction

On reject, correction is now sampled from the **positive residual**:

- `max(0, p_T - q_T)`

rather than using the old temperature-agnostic `argmax(p-q)` shortcut.

This is the key exactness repair for `T != 1`.

## Build verification

```sh
make ds4 ds4-eval
```

Result:
- build passed warning-clean

## Temp=0 guard verification

Command:

```sh
./ds4 --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 -n 8 --temp 0 -p hi
```

Observed result:

```text
ds4: --dspark requires --temp > 0 (current DSpark exactness is defined only for stochastic decoding)
```

Exit code:
- `2`

## Benchmark setup for sub-1.0 temperatures

Common settings:

- model: `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- drafter: `../ds4/gguf/dspark.gguf`
- prompt: `issue468/prompts/chat_general.txt`
- backend: `metal`
- `ctx=8192`
- `n=128`
- `seed=1`

DSpark env:

- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- `DS4_DSPARK_B2_DEBUG=1`

Temperatures tested:

- `1.0`
- `0.8`
- `0.7`
- `0.5`

Oracle references for delta reporting:

- accepted drafts/cycle: `4.10`
- committed tokens/cycle: `5.10`

## Results — DSpark temp sweep (`T > 0`, exact temperature-aware path)

| temp | gen t/s | accepted drafts/cycle | delta vs oracle accepted | logical committed/cycle | delta vs oracle committed | ms/token |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 30.68 | 2.459 | -1.641 | 4.270 | -0.830 | 26.370 |
| 0.8 | 31.53 | 2.556 | -1.544 | 4.306 | -0.794 | 26.154 |
| 0.7 | 31.41 | 2.556 | -1.544 | 4.444 | -0.656 | 25.447 |
| 0.5 | **32.36** | **2.657** | **-1.443** | **4.514** | **-0.586** | **25.012** |

## Plain baseline temp sweep on the same setup

| temp | plain gen t/s |
|---|---:|
| 1.0 | 37.99 |
| 0.8 | 38.10 |
| 0.7 | 37.96 |
| 0.5 | 38.04 |

## Interpretation

### 1. Sub-1.0 temperature **does** help DSpark modestly

The exact temperature-aware DSpark path improved as temperature decreased:

- `30.68 t/s` at `1.0`
- `32.36 t/s` at `0.5`

That is a gain of about:

- `+1.68 t/s`
- about `+5.5%`

This is consistent with the hypothesis that lower temperature improves acceptance/commit efficiency rather than reducing verifier cost directly.

### 2. The best tested point was `temp=0.5`

At `0.5`, this run also improved the live-vs-oracle gaps versus the `1.0` run:

- accepted drafts/cycle gap: `-1.641 -> -1.443`
- committed tokens/cycle gap: `-0.830 -> -0.586`
- ms/token: `26.370 -> 25.012`

### 3. But it still does **not** beat plain baseline

Even the best tested sub-1.0 point:

- DSpark `32.36 t/s` at `temp=0.5`

remains below the same-temperature plain baseline:

- plain `38.04 t/s`

So sub-1.0 temperature is a **real but modest** gain, not a gate-clearing one on this benchmark.

## Conclusion

This lead was worth testing and produced a real result:

- DSpark is now explicitly blocked at `temp=0`
- DSpark B2 is now temperature-aware for `T > 0`
- sub-1.0 temperatures **can** be faster than `temp=1.0` while staying aligned with the stochastic exactness design

But the measured gain is moderate:

- best observed here: about `+5.5%` at `temp=0.5`
- still below plain baseline on the benchmark prompt

So this is a **useful exactness fix plus a modest performance improvement**, but not a replacement for the deeper verifier-side bottleneck work.

## Exact repro commands

### Build

```sh
make ds4 ds4-eval
```

### Temp=0 guard check

```sh
./ds4 --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 -n 8 --temp 0 -p hi
```

### DSpark temp sweep

```sh
MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DSPARK="../ds4/gguf/dspark.gguf"
PROMPT="$(cat issue468/prompts/chat_general.txt)"

for T in 1.0 0.8 0.7 0.5; do
  export DS4_DSPARK_TARGET_POS0=1
  export DS4_DSPARK_MERGE_CORRECTION=1
  export DS4_DSPARK_B2_DEBUG=1

  ./ds4 --backend metal \
    -m "$MODEL" \
    --dspark "$DSPARK" \
    -c 8192 \
    -n 128 \
    --temp "$T" \
    --seed 1 \
    -p "$PROMPT"

  unset DS4_DSPARK_TARGET_POS0
  unset DS4_DSPARK_MERGE_CORRECTION
  unset DS4_DSPARK_B2_DEBUG
done
```

### Plain baseline temp sweep

```sh
MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
PROMPT="$(cat issue468/prompts/chat_general.txt)"

for T in 1.0 0.8 0.7 0.5; do
  ./ds4 --backend metal \
    -m "$MODEL" \
    -c 8192 \
    -n 128 \
    --temp "$T" \
    --seed 1 \
    -p "$PROMPT"
done
```
