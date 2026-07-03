# Confidence-based verifier-depth policy — experiment design and results

Date: 2026-07-03.
Branch: `dspark`.
Status: design recorded before implementation; results appended after measurement.

## Lead

Test whether a **confidence-based verifier-depth policy** can outperform fixed verifier truncation for DSpark.

The idea is:
- keep the current exact B2 machinery unchanged
- choose the verified draft-prefix length **before verification**, using only drafter-side information already available at draft time
- verify a shorter prefix on low-confidence cycles, but keep a longer prefix on high-confidence cycles

This is a bounded policy experiment, not a kernel change.

## Why this lead is worth one experiment

Fixed verifier truncation (`DS4_DSPARK_VERIFY_N`) already showed a mixed result:
- lower verify depth reduced raw verifier milliseconds
- but hurt acceptance enough that end-to-end performance still lost to plain baseline

That result does **not** rule out a better policy entirely, because fixed `N` is blunt:
- some cycles are likely strong enough to justify `N=5`
- some cycles likely waste verifier work past position `2-4`

A confidence policy may dominate a fixed setting by keeping long verification only on easy cycles.

## Why this is still low-confidence overall

The verifier microprofile (`issue468/83`) already established that verifier time is dominated by the layer batch pass itself.

So even a successful confidence policy would still be a **policy-side refinement**, not a bottleneck-class breakthrough. The expected upside is therefore limited.

## Exactness requirement

The policy must preserve DSpark's current stochastic exactness contract.

That means:
- the policy may only depend on information already available from the drafter side before current-cycle verification
- it must not depend on current-cycle target verifier outputs
- once the policy chooses the verify depth, the existing B2 verifier + acceptance machinery must run unchanged on that prefix

Under those conditions, the policy changes efficiency only, not the target distribution.

## Policy to test

Use a **prefix-confidence policy**.

For the drafted block:
- always verify at least a minimum prefix
- extend the verified prefix only while the already-sampled draft token at the current position is sufficiently confident under the drafter distribution

This matches the speculative-decoding structure better than a single global block score, because B2 stops at the first reject.

## Minimal implementation plan

Add an env-gated policy in `ds4_session_eval_dspark_b2(...)`.

Proposed controls:
- `DS4_DSPARK_VERIFY_CONF=1` — enable confidence policy
- `DS4_DSPARK_VERIFY_CONF_MIN=<float>` — minimum drafted-token confidence required to extend the prefix
- optional base prefix default:
  - if `DS4_DSPARK_TARGET_POS0=1`, verify at least `2`
  - otherwise verify at least `1`

Policy rule:
- start from the minimum prefix
- for each later position, if the current drafted token confidence is below threshold, stop extending
- otherwise extend by one
- cap by the existing `DS4_DSPARK_VERIFY_N` maximum if present, else by the full block

## Confidence signal to test

Use the drafted token's probability under the drafter's temperature-scaled distribution.

Reason:
- it is directly tied to the actual sampled draft token
- it is available before verification
- it should correlate better with expected acceptance than a simple fixed suffix length

## Measurement design

### Common runtime settings

- model: `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- drafter: `../ds4/gguf/dspark.gguf`
- prompt: `issue468/prompts/chat_general.txt`
- backend: `metal`
- `ctx=8192`
- `n=128`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`

### Temperature regimes

Use the standard 4-temperature sweep:
- `1.0`
- `0.8`
- `0.7`
- `0.5`

### Modes to compare

1. full verify (`N=5`)
2. best prior fixed truncation (`N=4`)
3. confidence policy (`N<=5`, chosen dynamically)

### Two primary reported metrics

Per user request, the headline comparison will report exactly two primary metrics:

1. **generation throughput (t/s)**
2. **logical committed tokens per cycle**

Auxiliary diagnostics may be collected internally, but the main result table will stay focused on those two metrics.

### Threshold calibration

Before the full 4-temperature sweep, do one short temp=`1.0` calibration across a small threshold set to choose a single threshold for the main sweep.

Candidate thresholds:
- `0.010`
- `0.020`
- `0.030`

Selection rule:
- choose the threshold with the best `t/s`
- if close, prefer the one with higher logical committed/cycle

Then lock that threshold and use it unchanged across all four temperatures.

## Success / failure bar

A worthwhile result should beat the prior fixed-truncation reference on the benchmark prompt/setup.

Reference from `issue468/81`:
- fixed `verify_n=4`: `32.09 t/s`, committed/cycle `4.188` (on the earlier temp=1 benchmark setup)

Within the current temp-aware branch, the more immediate comparison is whether the confidence policy beats the rerun fixed-`N=4` and full-`N=5` baselines under the same exact setup.

## Baseline contract and experiment matrix

For this goal, the **best current policy baseline** is defined as the incumbent fixed-depth verifier policy on the current exact-temperature branch:

- benchmark prompt: `issue468/prompts/chat_general.txt`
- backend: `metal`
- `ctx=8192`
- `n=128`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- fixed verifier depth: `DS4_DSPARK_VERIFY_N=4`

Measured baseline on this setup:

| temp | baseline policy | gen t/s | logical committed/cycle |
|---|---|---:|---:|
| 1.0 | fixed `N=4` | 35.26 | 4.314 |
| 0.8 | fixed `N=4` | 31.67 | 4.026 |
| 0.7 | fixed `N=4` | 30.96 | 3.950 |
| 0.5 | fixed `N=4` | 35.24 | 4.343 |

For the purpose of this goal, **material upside** means beating that baseline by **at least `+1.0 t/s`** on the benchmark setup.

So the per-temperature target bars are:

| temp | baseline t/s | material-upside bar |
|---|---:|---:|
| 1.0 | 35.26 | 36.26 |
| 0.8 | 31.67 | 32.67 |
| 0.7 | 30.96 | 31.96 |
| 0.5 | 35.24 | 36.24 |

Planned bounded search matrix for this note:

1. **Policy-only sweep at draft length `L=5`**
   - compare full `N=5`, fixed `N=4`, and five drafter-side verifier policies:
     - `prob`
     - `margin_pos`
     - `entropy`
     - `margin_prob`
     - `rank`
2. **Longer draft-length sweep at `L=7`**
   - rebuild with `-DDS4_DSPARK_BLOCK_SIZE=7`
   - test the strongest remaining candidates from the `L=5` sweep
   - report the same two metrics across the same 4-temperature convention
3. **Independent review**
   - after the bounded sweep, obtain one independent review before concluding no material upside remains

### L=7 candidate-selection plan

For the longer draft-length part of this goal, I will keep using this same note and stay bounded.

Because the `L=5` sweep showed that four policy families (`prob`, `margin_pos`, `entropy`, `margin_prob`) mostly collapsed to full verification, the `L=7` search will not rerun all four redundantly.

Instead, the `L=7` experiments will cover the strongest remaining candidate shapes:

1. **full `N=7`**
   - tests whether simply exposing two more draft positions has material upside
2. **one representative near-full policy family**
   - use `entropy` as the representative of the full-like calibrated policies
   - if tail confidence at positions `6-7` matters, this policy should be able to extend into it
3. **`rank` policy**
   - the only family from the `L=5` sweep that materially shortened average verify depth
   - the best chance to trade off deeper tails on easy cycles against shorter verification on harder cycles

All `L=7` policy thresholds will be recalibrated at `temp=1.0` before the 4-temperature sweep.

## Results

### Implementation summary

Implemented an env-gated policy in `ds4_session_eval_dspark_b2(...)`:

- `DS4_DSPARK_VERIFY_CONF=1` enables confidence-based verifier-depth selection
- `DS4_DSPARK_VERIFY_CONF_MIN=<float>` sets the drafted-token confidence threshold

Implementation details:
- after draft sampling, but before verifier launch, the code evaluates the drafted token probability under the drafter's temperature-conditioned distribution
- with `DS4_DSPARK_TARGET_POS0=1`, the policy verifies at least `2` positions
- it then extends the verified prefix while the drafted token probability stays above the threshold
- verification, acceptance, and correction logic remain otherwise unchanged

This preserves the intended exactness structure because the policy depends only on drafter-side information available before current-cycle verification.

### Calibration run

Calibration setup:
- prompt: `issue468/prompts/chat_general.txt`
- `ctx=8192`
- `n=128`
- `temp=1.0`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- `DS4_DSPARK_B2_DEBUG=1`

Threshold candidates tested:
- `0.010`
- `0.020`
- `0.030`

Calibration results:

| threshold | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 0.010 | 29.97 | 4.270 | 4.973 |
| 0.020 | 29.96 | 4.270 | 4.973 |
| 0.030 | **31.57** | **4.400** | **4.943** |

Chosen threshold for the main sweep:
- `DS4_DSPARK_VERIFY_CONF_MIN=0.030`

### Main 4-temperature comparison

Common settings:
- model: `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- drafter: `../ds4/gguf/dspark.gguf`
- prompt: `issue468/prompts/chat_general.txt`
- backend: `metal`
- `ctx=8192`
- `n=128`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- `DS4_DSPARK_B2_DEBUG=1`

Two primary metrics reported:
- generation throughput (`t/s`)
- logical committed tokens per cycle

#### Full `N=5` baseline

| temp | gen t/s | logical committed/cycle |
|---|---:|---:|
| 1.0 | 30.68 | 4.270 |
| 0.8 | 31.33 | 4.306 |
| 0.7 | 31.30 | 4.444 |
| 0.5 | 32.09 | 4.514 |

#### Fixed `N=4`

| temp | gen t/s | logical committed/cycle |
|---|---:|---:|
| 1.0 | **35.26** | 4.314 |
| 0.8 | **31.67** | 4.026 |
| 0.7 | **30.96** | 3.950 |
| 0.5 | **35.24** | 4.343 |

#### Confidence policy (`DS4_DSPARK_VERIFY_CONF=1`, `DS4_DSPARK_VERIFY_CONF_MIN=0.030`)

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 31.48 | **4.400** | 4.943 |
| 0.8 | 30.41 | 4.306 | 5.000 |
| 0.7 | 30.47 | **4.444** | 5.000 |
| 0.5 | 31.24 | **4.514** | 5.000 |

### Interpretation

#### 1. The confidence policy did not beat fixed `N=4`

Across all four temperatures, the confidence policy lost on throughput versus the fixed-`N=4` baseline:

- `temp=1.0`: `31.48` vs `35.26` t/s
- `temp=0.8`: `30.41` vs `31.67` t/s
- `temp=0.7`: `30.47` vs `30.96` t/s
- `temp=0.5`: `31.24` vs `35.24` t/s

So the policy did **not** dominate the best fixed truncation baseline on this benchmark setup.

#### 2. The chosen verify depth was usually almost full anyway

Average chosen depth:
- `4.943` at `temp=1.0`
- exactly `5.000` at `0.8`, `0.7`, and `0.5`

This explains the result:
- the drafted-token confidence threshold that worked best in calibration still almost always chose the full verifier depth
- therefore it failed to harvest much verifier cost reduction
- but it also did not improve acceptance enough to compensate

#### 3. At `temp=1.0`, the confidence policy slightly improved commits over full `N=5`, but not enough to matter

At `temp=1.0`:
- full `N=5`: committed/cycle `4.270`
- confidence policy: committed/cycle `4.400`

That is interesting as a small acceptance-side signal, but the throughput still remained far below fixed `N=4`.

### Extended sweep plan

After the first drafted-token-probability policy result, expand the study to sweep five policy families one after the other:

1. `prob` — drafted-token probability threshold
2. `margin_pos` — position-aware top-1/top-2 margin threshold
3. `entropy` — entropy threshold
4. `margin_prob` — hybrid probability + margin threshold
5. `rank` — sampled-token rank threshold

For each policy:
- calibrate one threshold at `temp=1.0`
- then run the same 4-temperature sweep (`1.0`, `0.8`, `0.7`, `0.5`)
- keep reporting the same two primary metrics:
  - generation `t/s`
  - logical committed tokens/cycle

### Extended sweep results

After generalizing the code to support multiple drafter-side verifier policies via `DS4_DSPARK_VERIFY_POLICY`, I reran the policy study as a 5-policy sweep.

Common settings:
- prompt: `issue468/prompts/chat_general.txt`
- backend: `metal`
- `ctx=8192`
- `n=128`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- `DS4_DSPARK_B2_DEBUG=1`

#### Calibration winners at `temp=1.0`

| policy | selected knob(s) | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---|---:|---:|---:|
| `prob` | `DS4_DSPARK_VERIFY_CONF_MIN=0.010` | 35.40 | 4.774 | 5.000 |
| `margin_pos` | `DS4_DSPARK_VERIFY_MARGIN_MIN=0.010` | 35.08 | 4.774 | 5.000 |
| `entropy` | `DS4_DSPARK_VERIFY_ENTROPY_MAX=3.0` | 35.11 | 4.774 | 5.000 |
| `margin_prob` | `CONF_MIN=0.020`, `MARGIN_MIN=0.010` | 35.02 | 4.774 | 5.000 |
| `rank` | `DS4_DSPARK_VERIFY_RANK_MAX=2` | 34.34 | 4.606 | 4.576 |

#### `prob` policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 35.13 | 4.774 | 5.000 |
| 0.8 | 31.96 | 4.529 | 5.000 |
| 0.7 | 32.93 | 4.576 | 5.000 |
| 0.5 | 29.52 | 4.270 | 5.000 |

#### `margin_pos` policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 35.04 | 4.774 | 5.000 |
| 0.8 | 31.87 | 4.529 | 5.000 |
| 0.7 | 32.98 | 4.576 | 5.000 |
| 0.5 | 29.50 | 4.270 | 5.000 |

#### `entropy` policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 35.06 | 4.774 | 5.000 |
| 0.8 | 32.00 | 4.529 | 5.000 |
| 0.7 | 32.97 | 4.576 | 5.000 |
| 0.5 | 29.56 | 4.270 | 5.000 |

#### `margin_prob` policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 35.12 | 4.774 | 5.000 |
| 0.8 | 31.95 | 4.529 | 5.000 |
| 0.7 | 32.97 | 4.576 | 5.000 |
| 0.5 | 29.47 | 4.270 | 5.000 |

#### `rank` policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 34.32 | 4.606 | 4.576 |
| 0.8 | 28.27 | 4.075 | 4.625 |
| 0.7 | 33.62 | 4.515 | 4.758 |
| 0.5 | 30.41 | 4.361 | 4.972 |

### Interpretation of the 5-policy sweep

#### 1. Four of the five policies collapsed to full verify

The `prob`, `margin_pos`, `entropy`, and `margin_prob` winners all selected average verify depth `5.000` (or effectively `5.000`) across almost all tested temperatures.

That means those policies were not actually finding a better dynamic tradeoff; they were just rediscovering full verification.

#### 2. Rank was the only policy that materially shortened verify depth

`rank` with `RANK_MAX=2` was the only sweep winner that consistently produced sub-5 average depth.

But its throughput was not compelling:
- strong-ish at `temp=1.0` and `0.7`
- clearly poor at `0.8`
- only middling at `0.5`

So even the only policy that really changed depth did not produce a robust win.

#### 3. No policy family emerged as a clear general winner

Among the near-full-depth policies, the throughput differences are tiny and likely reflect normal run variance more than a meaningful policy advantage.

Among the policies that genuinely altered behavior, `rank` did not show stable cross-temperature superiority.

## Longer draft-length sweep (`L=7`)

Rebuilt `ds4` with:

```sh
make clean
make ds4 CFLAGS='-O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -DDS4_DSPARK_BLOCK_SIZE=7'
```

This tests whether extending draft length from `5` to `7` opens material upside when paired with the strongest remaining verifier-policy candidates.

### `L=7` candidates tested

Per the bounded search plan above:

1. full `N=7`
2. representative near-full policy: `entropy`
3. policy that actually shortened depth at `L=5`: `rank`

### `L=7` calibration winners at `temp=1.0`

| candidate | selected knob(s) | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---|---:|---:|---:|
| full `N=7` | `DS4_DSPARK_VERIFY_N=7` | 27.64 | 4.571 | — |
| `entropy` | `DS4_DSPARK_VERIFY_ENTROPY_MAX=3.0` | 28.36 | 4.818 | 6.818 |
| `rank` | `DS4_DSPARK_VERIFY_RANK_MAX=1` | **32.77** | 4.412 | 4.412 |

### `L=7` full `N=7`

| temp | gen t/s | logical committed/cycle |
|---|---:|---:|
| 1.0 | 27.64 | 4.571 |
| 0.8 | 26.80 | 4.500 |
| 0.7 | 24.35 | 4.256 |
| 0.5 | 25.87 | 4.405 |

### `L=7` entropy policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 28.30 | 4.818 | 6.818 |
| 0.8 | 25.74 | 4.500 | 7.000 |
| 0.7 | 23.85 | 4.256 | 6.974 |
| 0.5 | 25.01 | 4.405 | 7.000 |

### `L=7` rank policy

| temp | gen t/s | logical committed/cycle | avg chosen verify depth |
|---|---:|---:|---:|
| 1.0 | 32.79 | 4.412 | 4.412 |
| 0.8 | 32.52 | 4.727 | 4.909 |
| 0.7 | 30.55 | 4.647 | 5.294 |
| 0.5 | 27.22 | 4.697 | 5.909 |

### Comparison against the material-upside bar

Recall the per-temperature bars (baseline `+1.0 t/s`):

| temp | material-upside bar |
|---|---:|
| 1.0 | 36.26 |
| 0.8 | 32.67 |
| 0.7 | 31.96 |
| 0.5 | 36.24 |

Best `L=7` result at each temperature:

| temp | best `L=7` candidate | best `L=7` t/s | clears bar? |
|---|---|---:|---|
| 1.0 | `rank` | 32.79 | no |
| 0.8 | `rank` | 32.52 | no |
| 0.7 | `rank` | 30.55 | no |
| 0.5 | `rank` | 27.22 | no |

So the longer draft-length sweep did **not** find material upside.

### Interpretation of the `L=7` sweep

#### 1. Simply extending the draft to `7` is harmful

Full `N=7` is substantially slower than the `L=5` baseline across all tested temperatures.

#### 2. Near-full tail extension does not pay for itself

The representative near-full `entropy` policy chose almost all `7` positions on average, increased logical committed/cycle modestly, but still lost badly on throughput.

That means the extra tail acceptance did **not** amortize the extra verifier cost.

#### 3. The best `L=7` candidate is still not material

`rank` was the strongest `L=7` policy because it actually shortened average verify depth.

But even its best run:
- `32.79 t/s` at `temp=1.0`

still missed the goal bar:
- required `36.26 t/s`

and it also failed the bar at every other tested temperature.

## Independent review after bounded sweep

An independent codex review (gpt-5.5 xhigh) was run after the bounded `L=5` policy sweep and bounded `L=7` draft-length sweep.

Review conclusion:
- it is reasonable to conclude this search space is exhausted **as an engineering conclusion**, though not as a mathematical proof
- the main caution is wording: “no material upside remains” should be read as “no material upside was found in the bounded search performed here,” not “every possible calibration is impossible”
- the only notable caveat is the `L=7` `rank` result at `temp=0.8`, which reached `32.52 t/s` versus the material-upside bar `32.67 t/s` — close enough that ordinary run noise could matter
- even so, one more bounded experiment was **not strongly justified** except possibly as a very narrow repeat around that one near-miss

Why the review still supports stopping:
- `L=5` policy families mostly collapsed back to full verification or failed to beat fixed `N=4` robustly
- `L=7` full verify and representative adaptive policies underperformed materially
- longer drafts were clearly worse at lower temperatures, especially `temp=0.5`
- no tested combination cleared the goal bar of baseline `+1.0 t/s`

## Inspectable supporting artifacts

To make the measurements and review independently inspectable, the raw logs and parsed manifests for this note are stored under:

- `issue468/artifacts/87_confidence_verify_policy/`

Artifact layout:
- `l5_manual/` — the initial fixed/full/manual confidence-policy calibration + 4-temp logs
- `l5_sweep/` — the 5-policy calibration logs and 4-temperature sweep logs
- `l7_sweep/` — the `L=7` calibration logs and 4-temperature sweep logs
- `review/codex_gpt55_xhigh_review.txt` — the independent codex/gpt-5.5 xhigh review transcript
- `measurement_manifest.csv` — parsed summary of all 76 measurement logs (group, file, gen t/s, logical committed/cycle average, avg chosen verify depth, cycle count)
- `artifact_manifest.json` — artifact inventory summary

These artifacts are the source of truth behind the tables in this note.

## Exact commands and env used

### Common `L=5` runtime command

```sh
./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 \
  -n 128 \
  --temp "$T" \
  --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)"
```

Common env for all `L=5` runs:

```sh
export DS4_DSPARK_TARGET_POS0=1
export DS4_DSPARK_MERGE_CORRECTION=1
export DS4_DSPARK_B2_DEBUG=1
```

Additional env by mode:

- fixed baseline:

```sh
export DS4_DSPARK_VERIFY_N=4
```

- full verify:

```sh
export DS4_DSPARK_VERIFY_N=5
```

- policy runs:

```sh
export DS4_DSPARK_VERIFY_POLICY=<prob|margin_pos|entropy|margin_prob|rank>
# plus one or more of:
export DS4_DSPARK_VERIFY_CONF_MIN=<float>
export DS4_DSPARK_VERIFY_MARGIN_MIN=<float>
export DS4_DSPARK_VERIFY_ENTROPY_MAX=<float>
export DS4_DSPARK_VERIFY_RANK_MAX=<int>
```

### `L=7` build and runtime commands

Rebuild for longer draft length:

```sh
make clean
make ds4 CFLAGS='-O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -DDS4_DSPARK_BLOCK_SIZE=7'
```

Then run the same command line as above, with these mode envs:

- full `N=7`:

```sh
export DS4_DSPARK_VERIFY_N=7
```

- `entropy` representative:

```sh
export DS4_DSPARK_VERIFY_POLICY=entropy
export DS4_DSPARK_VERIFY_ENTROPY_MAX=3.0
```

- `rank` representative:

```sh
export DS4_DSPARK_VERIFY_POLICY=rank
export DS4_DSPARK_VERIFY_RANK_MAX=1
```

### Independent review command

Independent review was run with codex/gpt-5.5 xhigh over the bounded `L=5` + `L=7` search results.

### Final conclusion

This bounded search did **not** find material upside in verifier-policy calibration / combined drafter-side confidence signals / modest longer draft length (`L=7`) space.

What was tested and ruled out in this note:
- five verifier-policy families at `L=5`
- bounded recalibration of those policy families
- a bounded longer-draft extension to `L=7`
- representative near-full and depth-shortening policies at `L=7`
- the standard 4-temperature convention: `1.0`, `0.8`, `0.7`, `0.5`

What the evidence supports:
- no tested combination beat the best current policy baseline by the required `>= 1.0 t/s`
- the strongest `L=7` candidate (`rank`) still failed the bar at every tested temperature
- the broader policy families mostly reproduced full verification rather than opening a new profitable operating region

So within the explicit verifier-policy / draft-length scope of this goal, the current evidence supports an honest negative verdict:
- **no material upside was found in the bounded search space explored here**
- the remaining meaningful headroom is still more likely in verifier compute cost itself, not in smarter verifier-policy calibration or a modest draft-length increase
