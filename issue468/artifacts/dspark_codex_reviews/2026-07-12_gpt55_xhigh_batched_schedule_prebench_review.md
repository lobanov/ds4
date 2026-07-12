# DSpark batched scheduled drafting — pre-benchmark adversarial review

Date: 2026-07-12
Reviewer: `gpt-5.5` (`xhigh`)
Scope: experimental `DS4_DSPARK_SCHEDULE_BATCHED=1` path in `ds4.c` and the new
`--dspark-schedule-parity` smoke regression in `tests/ds4_test.c`

## Findings

1. **The current batched path is not equivalent to the serial scheduler by
   construction.** `dspark_eval_draft_block_cpu()` drafts all rows together,
   and `dspark_block_forward_batch()` lets every row attend over
   `n_real + n_tok` KVs. `layer_attention_rows_one()` has no causal row mask,
   so row 0 can see future/noise block rows. The serial scheduler instead
   drafts one row at a time and appends each generated DSpark KV before the
   next row. Different proposals and confidence values are therefore expected.
2. **Full-block drafting does not preserve the serial scheduler's economics.**
   The serial scheduler stops as soon as cumulative survival crosses the STS
   threshold and verifies only the prior surviving prefix. The batched path
   always pays the full draft block first and truncates verify afterward. Even
   if final committed tokens stay exact, changed chunking matters because
   verifier target decode is still the dominant cycle cost.
3. **There was also a separate dispatch regression in the first opt-in
   implementation.** Non-batched cases were incorrectly routed through the
   scheduled helper even when scheduling was disabled. That was fixed locally in
   this cycle before any benchmark attempt.
4. **`verify_k=0` temp parity is not evidence for batched scheduled parity.**
   That test returns immediately after the first committed target token and does
   not exercise either scheduled drafter.
5. **The new `--dspark-schedule-parity` smoke regression is an appropriate gate
   if the goal is serial-equivalent speculative economics.** It is too strict
   only if the batched path is treated as an intentionally different policy
   rather than a drop-in optimization of the current scheduler.

## Recommendation

Do **not** benchmark or summarize the current full-block batched path as a
valid runtime optimization of the existing scheduled DSpark policy.

Two legitimate next routes remain:

1. **Serial-equivalent route:** refine the batched drafter so row `t` only
   attends to `n_real + t + 1`, not the full `n_real + n_tok`, and re-test
   against the chunk-parity smoke gate.
2. **Alternative-policy route:** keep the full-block mode as an explicitly
   different speculative policy and benchmark it only with a different contract:
   final token exactness plus direct economics reporting (drafted rows, verified
   rows, accepted chunk sizes, target decodes, wall time), not serial chunk
   parity.

If the alternative-policy route does not beat the retained serial scheduler on
those economics, abandon it.
