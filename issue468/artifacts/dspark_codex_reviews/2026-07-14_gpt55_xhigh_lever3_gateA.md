**Findings**

- **High: Metal DSpark state is not reset on session/prefill reset.**  
  [ds4.c:24081](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:24081) resets only `s->dspark_n_real`, and [ds4.c:20896](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:20896) resets `mtp_n_raw` but not `g->dspark_n_real`. The PR resets it in `metal_graph_reset_prefill_state` at [ds4-pr502/ds4.c:21426](/Users/lobanov/Projects/ds4-pr502/ds4.c:21426). After `ds4_session_sync`, snapshot restore, invalidate, or rewind, the Metal drafter can attend to stale `dspark_kv_cache` rows from an old prompt. This can produce wrong drafts and poor acceptance. Fix: reset `s->graph.dspark_n_real = 0` anywhere DSpark session state or prefill state is reset.

- **High: accepted drafts are not persisted on sequential fallback / batched verifier disabled.**  
  `DS4_DSPARK_VERIFY_BATCHED` is default-off at [ds4.c:28548](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28548). If the Metal drafter runs but the batched verifier does not commit, the sequential loop at [ds4.c:29946](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29946) updates the CPU DSpark window via `dspark_session_push_graph_hidden`, not `g->dspark_kv_cache`. Then [ds4.c:30080](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:30080) advances Metal `dspark_n_real` by only the anchor when `metal_refresh_done == false`, dropping accepted drafts from the Metal context. Target correctness is still protected by verification, but the drafter KV becomes incomplete. Fix: either require batched verify for Metal or call `metal_graph_dspark_refresh_current_row` per sequentially accepted draft.

- **Medium: SWA wrap is a cold-start, not a ring.**  
  [ds4.c:29504](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29504) and [ds4.c:30084](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:30084) reset `dspark_n_real` at the SWA boundary. The cache is addressed with `raw_start=0`, so there is no logical ring/slide. This is faithful to PR behavior in practice, but it discards the whole drafter context at 128 tokens rather than retaining the last window. Not a target-correctness bug, but expect an acceptance cliff around the boundary.

**Gate Answers**

The batched fast-commit call is correct when it succeeds: `row0 = metal_base_real + 1`, `pos0 = bstart`, `n_tokens = verified` matches the forward path, where the anchor is stored at row `base_real` with rope `checkpoint.len - 1`, and draft rows start at `base_real + 1` with rope `checkpoint.len`.

The cold-start fix is sound for normal cycles: when `verified == 0`, advancing to `base_real + 1` preserves the standalone anchor row, and the next cycle writes the next anchor at the next row rather than overwriting it.

The batch capture layout is compatible. [ds4.c:21769](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21769) writes `[row][stage][embd]`; [ds4.c:29351](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29351) reads that same row-major stage layout. It is wired after the batch `cur_hc` swap at [ds4.c:19764](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19764), so it captures post-FFN hiddens.

I did a static adversarial review only; I did not run the smoke/bench.