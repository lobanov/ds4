## Verdict Per Hypothesis

- **H1: sound.** Persistent Metal drafter KV is never advanced locally: refresh funcs exist at `ds4.c:29221-29353`, but there are no local call sites; PR calls them on commit and updates `graph.dspark_n_real` at `/Users/lobanov/Projects/ds4-pr502/ds4.c:29329-29335`, `29462-29467`, `29511-29516`, `29564-29569`, `29621-29627`.
- **H2: likely-wrong as stated.** It is not `main_proj` *vs* token embedding. DSpark uses both: `main_proj(hidden40..42)->main_x` plus target token embeddings for `[anchor, noise...]` (`ds4.c:28935-28967`, CPU equivalent `ds4.c:28696-28703`).
- **H3: sound, incomplete.** Empty persistent KV breaks attention, but there is also a local off-by-one: local passes `pos=s->checkpoint.len` to Metal draft (`ds4.c:29633-29636`); PR passes `checkpoint.len - 1` (`ds4-pr502/ds4.c:29181-29188`).
- **H4: likely-wrong as primary cause.** Output head writes `g->spec_logits` and markov applies afterward (`ds4.c:29380-29420`, `29502-29518`); GGUF markov dims/type match. Check later, not first.

## Metal Drafter Data Flow

1. Anchor target decode captures post-FFN HC mean for layers 40/41/42 into `dspark_metal_main_hidden`: `ds4.c:17458-17472`, `21739-21755`.
2. Input stage:
   - `main_proj(dspark_metal_main_hidden)` + `main_norm` -> `dspark_main_x`: `ds4.c:28935-28950`.
   - `[anchor, noise_token...]` target embeddings -> `batch_cur_hc`: `ds4.c:28954-28967`.
3. Per drafter stage:
   - candidate token KV rows stored at `n_real+1..`: `ds4.c:29118-29123`.
   - current anchor/main KV stored at `n_real`: `ds4.c:29125-29164`.
   - noncausal attention reads `n_real + 1 + n_tokens`: `ds4.c:29166-29178`.
4. Output:
   - HC head + norm + target output matmul -> `spec_logits`: `ds4.c:29380-29420`.
   - CPU readback, markov bias, argmax: `ds4.c:29502-29518`.

Broken locally: `g->dspark_n_real` stays 0, `dspark_verify_hidden` is not filled, refresh is unused, and the Metal call uses the wrong RoPE/start position.

## KV Fill Gap

PR flow:
- Prefill resets `g->dspark_n_real=0`; it does **not** prefill DSpark KV: `ds4-pr502/ds4.c:21422-21426`.
- Draft eval transiently writes current anchor + draft KVs inside `metal_graph_dspark_encode_attention`: `ds4-pr502/ds4.c:20025-20030`, `20066-20071`.
- Verify captures accepted target hiddens into `dspark_verify_hidden`: `ds4-pr502/ds4.c:17446-17483`, called at `22305`.
- Commit refresh projects those verified hiddens into drafter KV: `ds4-pr502/ds4.c:20128-20243`.
- Commit then advances `graph.dspark_n_real`: `ds4-pr502/ds4.c:29329-29335`.

Missing locally:
- `metal_graph_dspark_refresh_verified_rows` / `_current_row` have no call sites.
- Local verify captures HC into `dspark_batch_capture_hc`, not reduced `dspark_verify_hidden`: `ds4.c:19767-19775`; local `metal_graph_verify_suffix_tops` has no PR-style capture call (`ds4.c:21813-21841`).
- Local commit paths only update CPU DSpark state via `dspark_session_push_*`: `ds4.c:29782`, `29812`, `29831`, `29886`.

## Input-Path Question

Verified actual `dspark.gguf` metadata: `dspark.target_layer_ids=[40,41,42]`, `dspark.noise_token_id=128799`, `mtp.0.main_proj.weight=[12288,4096] Q8_0`, and no token embedding tensors. The inventory says the same: `main_proj` consumes concat mean hidden `[3*4096]` (`issue468/inventories/dsv4_flash_dspark_model.md:98-104`).

So:
- Token embedding input is **correct** for the draft residual stream `x_hc`.
- Hidden-state `main_proj` input is **required** for `main_x` / context KV.
- CPU is not token-only: its push path computes `main_proj` from captured hidden before filling `s->dspark_win_kv`: `ds4.c:28097-28113`.

One parity trap: CPU `dspark_eval_draft_block_cpu_scheduled_batched` passes `prefix_visible_only=true` (`ds4.c:28709-28718`), while PR Metal is noncausal/full-block. Compare Metal against `dspark_eval_draft_block_cpu`, not scheduled-batched, unless you intentionally want that approximation.

## Ranked Leads

1. **Wire PR commit lifecycle into local Metal path.** Fill `dspark_verify_hidden`, call refresh on full/prefix/replay commits, update `g->dspark_n_real`. Expected signal: `base_real` grows after cycle 1; acceptance jumps. Effort: medium.
2. **Fix Metal draft position.** Pass `s->checkpoint.len - 1`, matching PR. Expected signal: immediate draft/logit parity improvement. Effort: trivial.
3. **Unify verified-hidden source.** Either port PR `metal_graph_capture_dspark_batch_main_hidden` or adapt refresh to consume existing `dspark_batch_capture_hc`. Expected signal: refresh reads real hiddens, not empty buffer. Effort: small-medium.
4. **Build parity harness with same state.** Same `main_hidden`, same KV rows, same `n_real`, same noncausal visibility, compare CPU non-scheduled vs Metal logits/drafts. Effort: medium.
5. **Only then inspect output head/markov.** Current evidence does not put them ahead of KV/position bugs.

Runtime smoke was blocked by sandbox: binary tries to create `/tmp/ds4.lock` (`ds4.c:23952-23990`) and read-only permissions reject it. GGUF metadata verification did run.