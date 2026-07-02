# Opp-a retry investigation — merged correction-anchor implementation and first measurement

Date: 2026-07-02.

## Lead

Retry Opp-a using P0's accepted-prefix frontier commit: on partial accept, skip the standalone correction decode and carry the correction token `C` into the next cycle as an already-emitted pending anchor.

## Independent verification before implementation

Re-read the live `ds4_session_eval_dspark_b2` path and both call sites (`ds4_cli.c:run_sampled_generation`, `ds4_eval.c` speculative loop).

### What is already true

- P0 already makes the accepted-prefix KV committable via `spec_frontier_commit_dspark_prefix(...)`.
- On partial accept, the only remaining decode is the correction token `C`.
- If the next cycle decodes `C` as its anchor, that anchor forward installs `KV[C]` and produces `predict-after-C`, which is exactly what the skipped correction decode would have produced.

### What blocks the naive patch

A naive "skip correction decode" patch is incomplete for two reasons:

1. **Stale logits**
   - without the correction decode, `s->logits` still predicts after the rejected draft token, not after `C`
   - the main loop must therefore skip `ds4_session_sample(...)` on the next cycle and feed `C` directly as the anchor

2. **Duplicate emission**
   - `C` is already emitted in the current cycle
   - the next cycle must decode `C` internally but must **not** return/emit `C` again
   - this requires an explicit `first_token_already_emitted` mode and pending-anchor session state

## Implementation

Implemented on branch `dspark`:

### Session/API state
- added pending-anchor state to `ds4_session`
- added `ds4_session_take_dspark_pending_anchor(ds4_session *s, int *token)`
- extended `ds4_session_eval_dspark_b2(...)` to take `bool first_token_already_emitted`

### Partial-accept path
- under `DS4_DSPARK_MERGE_CORRECTION=1`, partial-accept now:
  - commits the accepted-prefix frontier
  - sets `pending_anchor = correction_token`
  - skips the standalone correction decode
- the next cycle decodes that pending anchor but does **not** re-emit it

### Call-site updates
- `ds4_cli.c` now checks for a pending anchor before calling `ds4_session_sample(...)`
- `ds4_eval.c` speculative loop does the same

### Instrumentation
- debug cycle log now prints:
  - returned `n_accept`
  - `logical_commits` (adds back the already-emitted pending anchor on merged cycles)
  - whether the cycle started from an already-emitted anchor

## Build verification

`make ds4 ds4-eval` passes warning-clean.

## First temp=1 measurement (ctx=8192, n=64, seed=1, prompt: short Python fibonacci function)

Common setting:
- `DS4_DSPARK_TARGET_POS0=1`

### A/B results

| mode | gen t/s | accepted drafts/cycle | returned committed/cycle | logical committed/cycle | ms/logical-commit |
|---|---:|---:|---:|---:|---:|
| merge OFF | 34.69 | 2.786 | 4.571 | 4.571 | 28.652 |
| merge ON  | 36.71 | 3.000 | 4.000 | 4.812 | 22.609 |

Interpretation:
- merged correction-anchor removes almost all `kv` time on the measured partial-accept cycles (`~25.9ms -> ~0.4ms` in the tail example)
- end-to-end speed improves **34.69 -> 36.71 t/s** (**+2.02 t/s**, +5.8%)
- logical committed tokens/cycle improve **4.571 -> 4.812** (+0.241)
- accepted drafts/cycle improve **2.786 -> 3.000** (+0.214)
- measured `ms/logical-commit` improves **28.652 -> 22.609**

### Example tail cycle

OFF:
- `n_accept=4 (drafts=2)`
- `kv=25.9ms`
- `total=132.9ms`

ON:
- `n_accept=3 logical_commits=3 emitted_anchor=0 (drafts=2)`
- `MERGED correction-anchor pending C=342`
- `kv=0.4ms`
- `total=107.4ms`

## Current status against Opp-a contract

Met so far:
- make warning-clean ✅
- correction decode reduced/eliminated on merged partial-accept cycles ✅
- cycle time materially reduced ✅
- DSpark runs end-to-end with clean output at temp=1 ✅

Still pending for this task:
- larger measurement sweep (this first run is only one prompt / n=64)
- direct live-vs-oracle delta report on the standard benchmark prompt set
- baseline/MTP non-regression under the final patch set
- check whether the gain is enough to cross baseline on the goal benchmark

## Quick non-regression spot checks

- plain decode (same prompt, temp=1, ctx=8192, n=64): **38.21 t/s**
- MTP spot run (`--mtp`, temp=0, same prompt/context): **39.02 t/s**

So Opp-a merged correction-anchor clearly helps, but this first measurement is
still below the plain baseline on the sampled prompt (`36.71 < 38.21`).

## Outcome of this iteration

Opp-a retry is **viable and beneficial**. The merged correction-anchor path is
correctly implemented and delivers a real speedup (+2.0 t/s on the first temp=1
measurement), but this first measurement does **not yet** clear the overall
baseline gate. Next step: run the standard temp=1 benchmark/acceptance comparison
and report live-vs-oracle deltas on the intended measurement setup.
