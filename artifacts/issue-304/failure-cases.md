# Issue #304 Failure Cases

This artifact records mismatches that should remain rejected during distributed-prefill-to-local-decode work.

## Current Status

- No Phase 3 official/local parity failure has been recorded yet.
- Phase 2 observed `CUDA -> Metal` forced-token logit drift after resumed eval, but handoff logits and the sampled greedy continuation matched exactly. This is a Phase 3 classification case, not currently a proven rejection case.

## Phase 3 Rejection Criteria

A drift case should be recorded here as rejected if it:

- makes distributed prefill -> merged `DSV4` -> local decode differ from fully local inference on the same decode backend,
- fails the selected-token/top-logprob envelope used by `./ds4_test --logprob-vectors`,
- fails the local-golden drift envelope used by `./ds4_test --local-golden-vectors`,
- changes greedy continuation where the same-backend fully-local baseline stays stable,
- or localizes to payload ABI, shard metadata, token hash, context, layer range, output-head, or cache-state corruption.

Cross-engine forced-logit differences alone are not a rejection case if same-backend handoff parity and official/local gates pass.

## Future Failure Classes To Preserve

- Token hash mismatch.
- Layer range mismatch.
- Context/raw-window/compression-cap mismatch.
- Model id or quant profile mismatch.
- Missing output head when logits are requested.
- Worker reconnect before shard fetch.
- Stale worker session after route rebuild.
- Incomplete or stale incremental KV return chunks.
