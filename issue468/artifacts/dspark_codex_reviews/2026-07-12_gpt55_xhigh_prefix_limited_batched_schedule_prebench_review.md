# DSpark prefix-limited batched scheduler pre-benchmark adversarial review

Date: 2026-07-12
Reviewer: `gpt-5.5` (`xhigh`)
Scope: current prefix-limited batched scheduled DSpark drafter in `ds4.c` and
the associated parity coverage in `tests/ds4_test.c`

## Verdict per claim

- `sound`: prefix-limited visibility removes the rejected future-row leak. The
  batched path now calls `layer_attention_rows_one(..., n_real + t + 1)` when
  `prefix_visible_only=true`, so row `t` only attends to the real prefix plus
  its own causal intra-block prefix.
- `sound, modulo numeric parity`: the batched scheduled body is a plausible
  semantic match to the retained serial scheduler. Both use anchor on row 0,
  noise on later rows, then apply sequential Markov/confidence scoring.
- `questionable`: committed-token / accepted-chunk parity alone is not a strong
  enough proof. Draft ids, confidence logits, and `verify_n` also need direct
  parity coverage.
- `likely-wrong`: batched scheduling should not be treated as cost-equivalent
  to the retained serial scheduler, because serial can stop at the STS frontier
  while the batched path computes the full short block before truncating
  verification.

## Residual risks before benchmarking

- Batched kernels may still introduce small numeric drift versus repeated
  `n_tok=1` calls, especially around STS threshold cliffs.
- Final-output exactness and chunk parity can still miss draft-side economic
  drift if the corpus slice is too small or too easy.
- Runtime interpretation must distinguish `rows_computed` from `verify_n`,
  because the batched path may intentionally overcompute relative to the serial
  scheduler.

## Missing tests or instrumentation called out by the review

- Per-cycle serial-vs-batched trace of draft ids, confidence logits, and
  computed `verify_n`
- Direct tolerance gate for confidence-logit parity near scheduling thresholds
- Better runtime instrumentation separating rows computed from rows verified

## Optimization opportunities highlighted

- If stronger parity holds, benchmark this CPU shape; it is the right CPU-side
  stepping stone toward GPU migration.
- Keep verifier cost central: drafter wins alone may not move end-to-end
  throughput enough while target decode dominates verifier cost.
- Consider block-size tuning rather than assuming the full short block is always
  the right scheduled batch size.

## Best GPU migration path highlighted

- Port the prefix-limited full-block DSpark microbatch, not the retained
  token-serial scheduler.
- Keep DSpark KV/window state resident on device.
- Have the GPU return draft ids and confidence logits; host-side scheduling can
  still choose `verify_n`.

## Dispatcher verification

The dispatcher independently confirmed the key code-path claims against the
current sources:

- prefix-limited attention is implemented in `dspark_block_forward_batch()`
- the batched scheduled path uses the same anchor/noise setup and sequential
  Markov/confidence scoring shape as the serial scheduler
- the old `--dspark-schedule-parity` smoke gate was indeed too weak before the
  stronger probe-based parity coverage was added

Follow-up implementation after this review:

- added `ds4_session_dspark_schedule_probe(...)` in `ds4.h` / `ds4.c`
- upgraded `tests/ds4_test.c --dspark-schedule-parity` so it now checks
  per-cycle draft-id, confidence-logit, and `verify_n` parity before checking
  committed-token / accepted-chunk parity
