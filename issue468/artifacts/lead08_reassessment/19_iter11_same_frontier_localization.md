# Lead 08 iteration 11: same-frontier M=1/M=K localization

Date: 2026-07-17. Status: **COMMIT after corrected repeat red-team audit.** The first audit
found that the probe ignored restoration failure. The retained implementation now hard-fails the
cycle if restoration fails, and the full corpus and both localization captures reproduce.

## Question

Iteration 10 proved that complete M=1 and M=K trajectories diverge, but cache/frontier differences
could not identify the first M-dependent operation. Iteration 11 uses the existing noncommitting
distribution probe from a shared frontier, disables committing M=K, and replays accepted drafts
through the iteration-10 batched-M1 target. This compares M=K row logits with M=1 row logits before
the M=K state can become canonical.

## Corpus result

The retained iteration-10 speculative config was run with:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
DS4_LEAD08_BATCH_M1_TARGET=1 DS4_DSPARK_VERIFY_DIST_PROBE=1 \
DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1 \
DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
DS4_DSPARK_DRAFT_METAL_STS=1 \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/18_iter10_spec_config.jsonl \
  --jsonl-out /tmp/lead08_iter11_dist_batchm1_out_v2.jsonl
```

All 10 runs completed, emitting 640 tokens. Across 291 cycles, 267 ran the distribution probe and
176 produced 349 accepted M=K/M=1 row pairs with **2 argmax flips**. Every singleton target evaluation
used batched M=1 (`640` batched, `0` raw). Weighted mean TV was `0.00439885`; maximum absolute logit
difference was `3.93787`. The flips were:

| prompt | zero-based cycle | emitted before cycle | compared rows | batched top-2 margin |
|---|---:|---:|---:|---:|
| `code_sort_pairs` | 24 | 55 | 2 | 0.241680 |
| `synthesis_incident_json` | 26 | 51 | 3 | 0.047100 |

The `16.54 t/s` diagnostic throughput is not economic evidence: the distribution probe runs both
paths and synchronizes for logits.

After the M=K call, the diagnostic resets the checkpoint length and requires
`spec_frontier_restore` to succeed. A failed restore frees the diagnostic buffers, invalidates the
checkpoint, and returns an error. Therefore all 267 probe executions in these 10 successful runs
restored their saved frontier before the canonical M=1 replay.

## First divergent stage

The `code_sort_pairs` flip is zero-based cycle 24. Its suffix starts at model position 104: frontier
48 + 55 tokens emitted by prior cycles + the current standalone anchor token (`66`, distinct from
draft row 0 token `469`). A second pair of post-fix runs used the retained one-prompt config
`19_iter11_code_sort_pairs_config.jsonl` and existing
`DS4_METAL_GRAPH_DUMP_{PREFIX,NAME,LAYER,POS}` hooks. The M=K diagnostic path is tagged `b_`; the M=1
replay is tagged `s_`. The dump run reproduced the same flip at cycle 24 with the same drafts and
`max_abs=1.00074`, so synchronization did not change the observed trajectory.

Exact self-contained post-fix capture commands:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
export DS4_LEAD08_BATCH_M1_TARGET=1 DS4_DSPARK_VERIFY_DIST_PROBE=1
export DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1
export DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1
export DS4_DSPARK_DRAFT_METAL_STS=1
mkdir -p /tmp/lead08_iter11_code_sort_pairs_dump_v3 \
  /tmp/lead08_iter11_code_sort_pairs_dump_v4
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/lead08_iter11_code_sort_pairs_dump_v3/dump \
  DS4_METAL_GRAPH_DUMP_NAME=hc_attn_pre,hc_attn_post,hc_ffn_post \
  DS4_METAL_GRAPH_DUMP_LAYER=all DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter11_code_sort_pairs_growth_out_v3.jsonl
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/lead08_iter11_code_sort_pairs_dump_v4/dump \
  DS4_METAL_GRAPH_DUMP_NAME=hc_attn_pre,attn_norm,q_lora,KVraw,q_lora_norm,KVnorm,Qraw,Qnorm,Qcur,KVrope,KVcur,raw_cache,kqv_out,kqv_back,attn_low,attn_out,hc_attn_post \
  DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter11_code_sort_pairs_stage_out_v4.jsonl
```

Ordinary stage tensors contain two row-contiguous rows in `b_` and one in `s_`. `raw_cache` is the
position-major cache prefix: `b_` contains 106x512 F32 values and `s_` contains 105x512. In every
case the retained CSV compares the first `s_`-sized prefix of `b_` with `s_`, element by element as
F32; for `raw_cache`, that is the common prefix through row 0.

Only row 0 is an identical-frontier/identical-input comparison. Later M=K rows share the cycle-start
frontier, while sequential M=1 has committed earlier rows. Row 0 is bit-identical through
`hc_attn_pre` and `attn_norm` at layer 0. The first captured
difference is `q_lora`: 799/1024 F32 words differ, max `3.35e-8`, RMS `9.85e-9`. The independent
KV projection also differs immediately (405/512 words, max `2.98e-8`). The attention error grows
through Q/KV normalization and attention, reaching max `1.72e-5` at `attn_out`. The post-attention
HC state differs in 9563/16384 words. Propagation amplifies the initially tiny shape-dependent
difference: post-FFN max absolute difference grows from `6.85e-7` at layer 0 to `0.94` at layer 42.

This is operation-boundary localization, not proof that Q/KV projections are the only
source of divergence. Later batch operations may independently be M-dependent.

## Red-team audit gate

The first required post-iteration audit returned **NO-COMMIT**:

1. `spec_frontier_restore` failure was ignored, so successful restore was not enforced for the 349
   counted comparisons.
2. The draft conflated 267 probe-present cycles with the 176 cycles that produced compared rows.
3. Reproduction needed exact dump filters, row-0 prefix convention, zero-based cycle notation,
   position-104 derivation, and row-0-only identical-input scope.

The audit independently validated the raw 349-row/2-flip/640-batched/0-raw arithmetic, both flip
cycles, ordinary-stage row layout and raw-cache prefix layout, stage/growth CSV values, and bounded
next-step rationale. The
restore check and full post-fix rerun above resolve finding 1; the corrected counts and reproduction
detail resolve findings 2 and 3.

The repeat audit returned **COMMIT**. It independently recomputed the corpus arithmetic, flip
coordinates, tensor layouts, CSV values, command completeness, restore hard-failure behavior, build,
and ledger gates. Its only nonblocking residual is that a restore-failure return can leave the global
debug dump path tag as `b_`; the session cannot continue computation, but a later bulk diagnostic
could inherit that filename tag before resetting it.

## Decision

Do not implement the whole exact hybrid yet. The next bounded falsifier should route layer-0 Q/KV
projections through row-wise M=1 kernels while leaving downstream batch work unchanged, then repeat
the position-104 capture. Continue stage-by-stage only if the first divergent boundary moves and a
matched timing probe leaves a credible path to `verify_ms(4) <= 50.5 ms`. Abort if exactifying the
necessary attention projection/reduction family consumes the available verifier budget.
