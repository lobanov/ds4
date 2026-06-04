# Issue 304 Logit Comparisons

## Phase 1 local `DSV4` resume

The new `./ds4_test --local-payload-resume` path compares:

1. local session A after prefill
2. fresh local session B after `DSV4` load
3. short greedy continuation from both sessions

### Summary

- Backend pair: Metal -> Metal
- Result: exact match on all tested frontiers
- Non-finite logits: none
- Top-1 mismatches: none
- Greedy continuation mismatches: none

### Frontier results

| Frontier | Payload bytes | Top-1 match | Top-5 overlap | Top-20 overlap | RMS drift | Max abs drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 12,811,664 | Yes | 5/5 | 20/20 | 0 | 0 |
| 3 | 12,987,800 | Yes | 5/5 | 20/20 | 0 | 0 |
| 4 | 13,129,628 | Yes | 5/5 | 20/20 | 0 | 0 |
| 5 | 13,217,696 | Yes | 5/5 | 20/20 | 0 | 0 |
| 127 | 25,574,792 | Yes | 5/5 | 20/20 | 0 | 0 |
| 128 | 25,757,580 | Yes | 5/5 | 20/20 | 0 | 0 |
| 129 | 25,757,584 | Yes | 5/5 | 20/20 | 0 | 0 |

### Greedy continuation

- Continuation length checked: 8 tokens
- Result: identical token sequence from the saved session and the restored session at every tested frontier

## Phase 1 cross-backend matrix spot check

The focused cross-backend helper used:

- prompt: `tests/long_context_story_prompt.txt`
- `ctx=192`
- frontier `129`
- greedy continuation length `8`

### Results

| Save backend | Load backend | Top-1 match | Top-5 overlap | Top-20 overlap | RMS drift | Max abs drift | Greedy continuation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| CUDA | CUDA | Yes | 5/5 | 20/20 | 0 | 0 | Match |
| Metal | CUDA | Yes | 5/5 | 20/20 | 0 | 0 | Match |
| CUDA | Metal | Yes | 5/5 | 20/20 | 0 | 0 | Diverged at step 2 (`5655` vs `305`) |

### Interpretation

- The current `CUDA -> Metal` problem is subtler than a bad immediate restore.
- The restored Metal session reproduces the CUDA reference logits exactly at the handoff point, then diverges only after subsequent token evaluation.

## Forced-token trace follow-up

To separate token-selection drift from post-load decode evolution, the same `CUDA -> Metal` payload was loaded on both backends and then advanced with the exact same forced token sequence from the CUDA reference continuation.

### Result

| Compared traces | First bad step | Forced token before bad step | Top-1 at bad step | Top-5 overlap | Top-20 overlap | RMS drift | Max abs drift |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| CUDA ref vs Metal candidate | 1 | `3737` | same (`14`) | 5/5 | 19/20 | 0.304555088 | 1.55702209 |

### Interpretation

- The traces are identical at restore step `0`.
- After one identical forced eval token (`3737`), the logits already diverge.
- This rules out “the backends only diverge because they sampled different next tokens” for the current failure.
- The remaining likely causes are:
  - backend-specific interpretation of restored state used by the next decode step, or
  - backend-specific decode-state evolution from the same restored state.

## Phase 2 distributed-prefill handoff

The new `tests/issue304_phase2_handoff` helper compared:

1. distributed coordinator session after mixed Metal/CUDA prefill
2. fresh local Metal session after merged `DSV4` load
3. distributed vs local greedy continuation
4. distributed-reference forced-token trace vs local resumed forced-token trace

### Summary

- Route: `local 0:21 -> 10.77.0.2:43045 Q2 22:output`
- Prompt: `README.md`
- Prompt tokens: `14,318`
- Result:
  - handoff logits matched exactly
  - 16-token greedy continuation matched exactly
  - forced-token trace diverged at the first post-load eval step

### Handoff result

| Compared states | Top-1 match | Top-5 overlap | Top-20 overlap | RMS drift | Max abs drift | Non-finite |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| distributed handoff vs local resumed handoff | Yes | 5/5 | 20/20 | 0 | 0 | 0 |

### Greedy continuation

| Compared traces | Steps checked | Result |
| --- | ---: | --- |
| distributed decode vs local resumed decode | 16 | Exact match |

### Forced-token trace

| Compared traces | First bad step | Forced token before bad step | Top-1 at bad step | Top-5 overlap | Top-20 overlap | RMS drift | Max abs drift |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| distributed reference vs local resumed Metal | 1 | `5` | same (`420`) | 5/5 | 20/20 | 0.0802887231 | 0.507860184 |

### Interpretation

- Phase 2 removes staging and initial restore correctness as leading suspects.
- The mixed-backend drift still shows up only after resumed decode evolution begins.
- The fact that greedy continuation matched for 16 tokens while forced-trace drift still appeared means the drift is currently too small to perturb top-token choice in this sampled window.
- Phase 3 should not require bit-exact cross-engine logits. It should decide whether the distributed-prefill-to-local path matches fully local inference on the same decode backend and satisfies the existing official-vector/local-golden correctness envelope.

## Phase 3 planned official/local parity

The Phase 3 logit comparison target is:

1. fully local inference on the decode backend
2. distributed prefill + merged `DSV4` + local decode on that same backend
3. official API top-logprob vectors in `tests/test-vectors/official.vec`
4. local golden-vector drift thresholds where full local logits are available

Acceptance should use the existing local correctness semantics:

- `./ds4_test --logprob-vectors` for official selected-token and top-logprob agreement, allowing the existing skipped `long_memory_archive` caveat.
- `./ds4_test --local-golden-vectors` for local top-k/logit drift regression.
- A new or extended issue-304 harness should compare distributed-handoff outputs against fully local outputs before classifying any cross-engine forced-logit drift as a defect.

## Baseline surrounding checks

These were rerun after the Phase 1 changes to confirm the new harness did not perturb existing local correctness surfaces.

| Check | Result | Notes |
| --- | --- | --- |
| `./ds4_test --local-golden-vectors` | Pass | `top1=4371`, `top20_overlap=16/20`, `top64_overlap=56/64`, `top20_max_abs=3.2229` |
| `./ds4_test --metal-short-prefill` | Pass | Existing short prefill regression remained green |
| `./ds4_test --metal-tensor-equivalence` | Pass | Worst observed `rms=0.024882`, `max_abs=0.105766`, `top20_max_abs=0.0233078` on `long_code_audit` |
| `make test` | Pass | Full default test target passed with the new resume test included |
