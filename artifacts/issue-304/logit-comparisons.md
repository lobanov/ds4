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

## Baseline surrounding checks

These were rerun after the Phase 1 changes to confirm the new harness did not perturb existing local correctness surfaces.

| Check | Result | Notes |
| --- | --- | --- |
| `./ds4_test --local-golden-vectors` | Pass | `top1=4371`, `top20_overlap=16/20`, `top64_overlap=56/64`, `top20_max_abs=3.2229` |
| `./ds4_test --metal-short-prefill` | Pass | Existing short prefill regression remained green |
| `./ds4_test --metal-tensor-equivalence` | Pass | Worst observed `rms=0.024882`, `max_abs=0.105766`, `top20_max_abs=0.0233078` on `long_code_audit` |
| `make test` | Pass | Full default test target passed with the new resume test included |
