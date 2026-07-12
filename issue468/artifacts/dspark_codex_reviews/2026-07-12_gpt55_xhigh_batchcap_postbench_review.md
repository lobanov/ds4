# DSpark scheduled batch-cap sweep post-benchmark adversarial review

Date: 2026-07-12  
Reviewer: `gpt-5.5` (`xhigh`)  
Scope: short retained batch-cap sweep using the current `ds4-spec-bench`
harness with one reused engine and a fresh session per run

## Verdict

This sweep establishes a **real short-sample signal in favor of capped batched
scheduling**, with **cap3 as the strongest arm on this exact 6-prompt retained
sample**, but it does **not** establish a general or production-stable optimum.

What is actually supported by the artifacts:

- All reviewed rows are `status="ok"`, emit `32` tokens, have
  `accepted_total=32`, and show the intended scheduler flags for each arm.
- The throughput pass is usable for end-to-end comparison because
  `DS4_DSPARK_TIMING` was explicitly unset.
- The diagnostic pass is usable for attribution because
  `DS4_DSPARK_TIMING=1` was explicitly enabled and the same ordering survives
  there.
- `cap5` is not attractive on this sample.
- `cap4` is a modest positive vs serial on throughput.
- `cap3` is the clearest positive vs serial and also beats `cap4` on all 6
  paired prompts in both the throughput pass and the timing pass.

## Safe claims

- On this exact retained 6-prompt sample, `cap3` has the best observed mean
  throughput:
  - serial `24.7115` tps
  - cap5 `24.0584` tps
  - cap4 `25.5452` tps
  - cap3 `26.9908` tps
- The `cap3` win is not just a mean artifact from one row:
  - `cap3` beats serial on all 6 paired prompts
  - `cap3` beats `cap4` on all 6 paired prompts
- The timing pass points to a **draft-side** improvement, not a verifier-side
  improvement:
  - serial `draft_ms 27.27`, `verify_ms 29.12`, `total_ms 82.35`
  - cap4 `draft_ms 25.84`, `verify_ms 29.07`, `total_ms 81.12`
  - cap3 `draft_ms 20.18`, `verify_ms 28.58`, `total_ms 74.93`
- `cap5` looks over-batched on this sample: it raises rows computed materially
  without improving acceptance or verification means, and throughput regresses.
- `cap4` preserves the reported acceptance means exactly vs serial on this
  sample.
- `cap3` gets its speedup while computing fewer rows and slightly fewer
  verification tokens per cycle than serial and `cap4`.

## Unsafe claims

- “`cap3` is the best cap in general.”
- “The optimal cap for DSpark is now proven.”
- “Batched scheduling is universally faster.”
- “Acceptance economics are unchanged across caps.”
- “This result is production-stable.”
- “Verifier cost was meaningfully reduced.”
- “Correctness / exact output parity across caps is proven by this sweep.”

## Best interpretation of `cap4` vs `cap3`

`cap4` is the **conservative** choice and `cap3` is the **higher-leverage**
choice.

Why `cap4` looks conservative:

- It improves throughput vs serial, but only modestly.
- It keeps `accepted_mean` and `verified_mean` identical to serial on this
  sample.
- It still computes more rows than serial, so some extra draft work remains in
  the system.

Why `cap3` looks stronger:

- It beats `cap4` on every paired prompt.
- It cuts `rows_computed_mean` and `verify_n_mean` on all 6 paired prompts
  relative to `cap4`.
- The timing pass shows a large draft-time drop from `cap4` to `cap3` while
  `verify_ms` stays nearly flat.
- That pattern is consistent with `cap3` reducing over-drafting and verifier
  setup burden enough to matter end-to-end.

Why `cap3` is not yet a free claim:

- Its reported `accepted_mean` and `verified_mean` are slightly lower than
  serial / `cap4`.
- That change comes from one prompt (`dolly_0080`), where `cap3` adds one cycle
  relative to the other arms.
- With only 6 prompts, that is not enough to reject `cap3`, but it is enough
  to block a “strictly no tradeoff” claim.

Net: `cap4` is the safer local default if you want the smallest behavioral
delta; `cap3` is the better-performing candidate if you are willing to carry a
small acceptance-metric regression risk into the next validation round.

## Recommended next step

Run a **larger interleaved paired validation focused on `cap3` vs `cap4` vs
serial**, not another broad cap sweep.

Required shape:

- Keep the same harness semantics: one reused engine, fresh session per run.
- Use a materially larger prompt set and interleave run order to reduce
  thermal/order bias.
- Keep a throughput pass with `DS4_DSPARK_TIMING` unset.
- Keep a separate attribution pass with `DS4_DSPARK_TIMING=1`.
- Add or retain a stable output-equality check or output hash so the next
  review can make a stronger correctness statement.
- Promote `cap3` only if the larger paired run preserves its all-prompt or
  near-all-prompt throughput lead without widening the small acceptance/verified
  regression seen on `dolly_0080`.

Bottom line: this sweep is enough to justify **`cap3` as the next candidate to
validate**, not enough to declare it the final cap.
