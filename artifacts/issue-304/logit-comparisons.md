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

## Phase 3 official/local parity

The new `tests/issue304_phase3_vectors` helper compared:

1. fresh fully local Metal inference
2. distributed prefill -> merged `DSV4` -> local Metal load
3. official API vector acceptance for official cases
4. the existing local golden fixture for the `long_story_4096` frontier

Local and DGX GGUF hashes matched exactly before these runs:

- `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`

### Official vectors, worst@5

The official top-logprob gate passed on every run below, but same-backend parity against a fresh local Metal baseline did not stay close enough to classify the remaining drift as harmless backend variance.

| Case | Ctx | Worker `prefill_cap` | Official gate | Top-5 | Top-20 | Top-64 | RMS | Max abs | Top-20 max abs | Envelope |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_code_completion` | 4096 | 2048 | Pass | 5/5 | 17/20 | 62/64 | 0.356403291 | 1.69900322 | 0.734794617 | Tolerant only |
| `short_reasoning_plain` | 4096 | 2048 | Pass | 5/5 | 20/20 | 62/64 | 0.327032387 | 1.77348614 | 0.5119133 | Tolerant only |
| `short_italian_fact` | 16384 | 2048 | Pass | 5/5 | 19/20 | 62/64 | 0.21716401 | 1.00239372 | 0.559177399 | Tolerant only |
| `long_code_audit` | 16384 | 2048 | Pass | 4/5 | 15/20 | 55/64 | 0.876981199 | 4.63468361 | 2.87808609 | Tolerant floor |

### Local-golden frontier, worst@5

The resumed handoff still satisfied the existing local-golden fixture itself on all five runs, but the same resumed state diverged sharply from a fresh local Metal baseline over the next eight greedy steps.

| Case | Ctx | Worker `prefill_cap` | Golden fixture | Golden top-20 | Golden top-64 | Golden top-20 max abs | Same-backend top-5 | Same-backend top-20 | Same-backend top-64 | Same-backend RMS | Same-backend max abs | Same-backend top-20 max abs | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `long_story_4096` | 5000 | 4096 | Pass | 16/20 | 56/64 | 3.20196915 | 3/5 | 12/20 | 46/64 | 2.1708498 | 12.3051796 | 7.73109055 | Same-backend parity fail |

### Interpretation

- Official selected-token/top-logprob agreement is not strong enough to prove handoff equivalence.
- The local-golden fixture is also not sufficient by itself: the resumed checkpoint can satisfy that coarse regression while still diverging materially from a fresh local Metal baseline immediately afterward.
- Phase 3 therefore classifies the remaining issue as a handoff-specific resumed-decode mismatch, not merely cross-engine numeric variance.

## Baseline surrounding checks

These were rerun after the Phase 1 changes to confirm the new harness did not perturb existing local correctness surfaces.

| Check | Result | Notes |
| --- | --- | --- |
| `./ds4_test --local-golden-vectors` | Pass | `top1=4371`, `top20_overlap=16/20`, `top64_overlap=56/64`, `top20_max_abs=3.2229` |
| `./ds4_test --metal-short-prefill` | Pass | Existing short prefill regression remained green |
| `./ds4_test --metal-tensor-equivalence` | Pass | Worst observed `rms=0.024882`, `max_abs=0.105766`, `top20_max_abs=0.0233078` on `long_code_audit` |
| `make test` | Pass | Full default test target passed with the new resume test included |

## Phase 3.5 six-route variance matrix

Phase 3.5 extends the comparison surface to six execution routes and measures each at worst@5:

1. pure local CUDA prefill+generation
2. pure local Metal prefill+generation
3. distributed generation `CUDA -> Metal`
4. distributed generation `Metal -> CUDA`
5. distributed prefill `CUDA -> Metal`, then resumed pure Metal generation
6. distributed prefill `Metal -> CUDA`, then resumed pure CUDA generation

The DGX worker and DGX-side helper were run with `--power 50`.

### Official vectors, worst@5 by route

`short_code_completion`

| Route | Official gate | First selected mismatch | Max logprob delta | Top-5 | Top-20 | Top-64 | RMS | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` local CUDA | Fail | 1 | 1.42368603 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `2` local Metal | Pass | -1 | 0.188549042 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `3` distributed `CUDA -> Metal` | Fail | 1 | 0.85852015 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `4` distributed `Metal -> CUDA` | Pass | -1 | 0.178744242 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `5` resumed `CUDA -> Metal` | Fail | 1 | 0.85852015 | 3/5 | 14/20 | 54/64 | 1.25698996 | 6.82095432 |
| `6` resumed `Metal -> CUDA` | Pass | -1 | 0.178744242 | 3/5 | 14/20 | 49/64 | 1.30479431 | 6.16060066 |

`short_reasoning_plain`

| Route | Official gate | Max logprob delta | Top-5 | Top-20 | Top-64 | RMS | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` local CUDA | Pass | 0.0258535184 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `2` local Metal | Pass | 0.0112166749 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `3` distributed `CUDA -> Metal` | Pass | 0.034484677 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `4` distributed `Metal -> CUDA` | Pass | 0.0139614902 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `5` resumed `CUDA -> Metal` | Pass | 0.034484677 | 5/5 | 20/20 | 61/64 | 0.382280707 | 2.11430359 |
| `6` resumed `Metal -> CUDA` | Pass | 0.0139614902 | 5/5 | 18/20 | 59/64 | 0.504765928 | 2.56024551 |

`short_italian_fact`

| Route | Official gate | Max logprob delta | Top-5 | Top-20 | Top-64 | RMS | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` local CUDA | Pass | 0.000240828318 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `2` local Metal | Pass | 0.000227290249 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `3` distributed `CUDA -> Metal` | Pass | 0.000264710223 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `4` distributed `Metal -> CUDA` | Pass | 0.000175568683 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `5` resumed `CUDA -> Metal` | Pass | 0.000264710223 | 4/5 | 18/20 | 56/64 | 0.663591683 | 3.45166492 |
| `6` resumed `Metal -> CUDA` | Pass | 0.000175568683 | 4/5 | 18/20 | 58/64 | 0.688694775 | 4.14918375 |

`long_code_audit`

| Route | Official gate | Max logprob delta | Top-5 | Top-20 | Top-64 | RMS | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` local CUDA | Pass | 0.107287943 | 4/5 | 16/20 | 52/64 | 0.893455684 | 4.59287167 |
| `2` local Metal | Pass | 0.0687521324 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `3` distributed `CUDA -> Metal` | Pass | 0.0741011128 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `4` distributed `Metal -> CUDA` | Pass | 0.0563559905 | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `5` resumed `CUDA -> Metal` | Pass | 0.0503566675 | 4/5 | 16/20 | 50/64 | 0.972514927 | 4.36004543 |
| `6` resumed `Metal -> CUDA` | Pass | 0.0933624879 | 4/5 | 15/20 | 53/64 | 0.903749108 | 4.15296125 |

### Local-golden frontier, worst@5 by route

`long_story_4096`

| Route | Golden fixture | Golden top-20 | Golden top-64 | Golden top-20 max abs | Same-route parity | Top-5 | Top-20 | Top-64 | RMS | Max abs |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `1` local CUDA | Pass | 17/20 | 52/64 | 3.07355976 | Fail | 2/5 | 13/20 | 44/64 | 2.12094522 | 15.0152826 |
| `2` local Metal | Fail | 16/20 | 46/64 | 5.2168293 | Strict | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `3` distributed `CUDA -> Metal` | Pass | 16/20 | 50/64 | 3.56194687 | Strict | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `4` distributed `Metal -> CUDA` | Pass | 16/20 | 53/64 | 3.55953407 | Strict | 5/5 | 20/20 | 64/64 | 0 | 0 |
| `5` resumed `CUDA -> Metal` | Pass | 16/20 | 51/64 | 2.24486637 | Fail | 3/5 | 13/20 | 42/64 | 2.14919782 | 14.7165461 |
| `6` resumed `Metal -> CUDA` | Pass | 16/20 | 52/64 | 3.03334045 | Tolerant | 3/5 | 14/20 | 40/64 | 1.60283542 | 8.42455864 |

### Interpretation

- The official API gate is materially weaker than route parity. Three `short_code_completion` routes missed official acceptance, while the longer official cases still passed even when resumed-route parity had already drifted.
- Resumed payload routes remain the weakest cells. They are the only routes that repeatedly combine nonzero official deltas with visibly looser top-k overlap and RMS on otherwise passing cases.
- The local-golden fixture itself is also route-sensitive. Pure local Metal (`route 2`) missed the stored fixture while both distributed direct-generation routes (`3` and `4`) passed it exactly against their own references.

## Phase 5 worker-owned local-decode acceptance surface

Phase 5 does not add a new strict token-parity gate. The acceptance surface for
the worker-owned handoff path is narrower:

- fresh-worker one-shot local-decode runs must complete through the intended
  handoff path,
- the handoff frontier must remain valid enough for coordinator catch-up and
  subsequent decode control flow,
- and the `ds4-eval --nothink` local-decode path must behave like the same
  worker-owned handoff workflow rather than the old per-token loop.

For the current Phase 5 closeout evidence:

- plain CLI fresh-worker runs in both directions already establish the intended
  handoff shape,
- the full `92`-question `CUDA -> Metal --local-decode` evaluator run completes
  with `65/92` versus `67/92` local Metal and `69/92` local CUDA,
- and the remaining reusable-session drift is classified as a follow-up-sync
  caveat rather than a fresh-worker handoff failure.

Interpretation:

- The authoritative Phase 5 question is now “does worker-owned local decode run
  correctly as a user-visible workflow?” rather than “does every reused session
  remain parity-identical to a fresh baseline?”
- Strict same-token parity on reused worker state remains a post-Phase-5
  classification item unless it turns into a concrete stale-KV, token-hash, or
  session-integrity failure.
