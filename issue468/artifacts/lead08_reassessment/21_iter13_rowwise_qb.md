# Lead 08 iteration 13: row-wise M=1 Q-b projection

Date: 2026-07-17. Status: **causal GO; post-iteration audit COMMIT.** This advances one
captured row at layer 0, not full-verifier exactness.

## Contract and preflight

V7 asked whether V6's first `Qcur` difference originates in Q-b changing from the M=1 Q8 reduction
to the multi-row reduction. The pre-experiment challenger returned **REDESIGN**: on Metal,
`ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor` is a stub that returns zero. Production actually
falls through to separate Q-b, head RMS norm, and RoPE. V7 therefore changes Q-b only and uses later
stages as checkpoints.

`DS4_LEAD08_ROWWISE_QB_LAYERS=N` is saved/restored only inside
`metal_graph_verify_suffix_tops`, engages only for `n_tokens > 1`, and calls the existing named Q8
primitive once per row. The correctness candidate also sets V6's Q/KV cap to one. Prompt prefill,
the M=1 comparator, normalization, RoPE, cache, attention, FFN, and output-head code are unchanged.
The shared layer-count parser now rejects negative values instead of clamping them to 43.

The first attempted aliases began with `Qraw`/`Qnorm`; a check showed the dump matcher is
prefix-based, so those aliases triggered `q_path_debug`. Although Metal's skipped fused hook is a
no-op, that evidence was rejected. The retained aliases `ProdQB` and `ProdQHeadNorm` do not match
either predicate; retained stderr confirms no `Qraw` or `Qnorm` dump fired.

## Correctness result

The retained `code_sort_pairs` capture starts at position 104 and compares the M=K row-0 F32 prefix
with complete M=1. Both modes use row-wise Q/KV at layer 0; only the candidate uses row-wise Q-b.

| boundary | Q-b batch | row-wise Q-b |
|---|---:|---:|
| `q_lora_norm` | 0 / 1,024 different | 0 / 1,024 |
| `ProdQB` | 26,044 / 32,768, max 1.19e-7 | **0 / 32,768** |
| `ProdQHeadNorm` | 25,855 / 32,768, max 1.91e-6 | **0 / 32,768** |
| `Qcur` | 25,988 / 32,768, max 1.91e-6 | **0 / 32,768** |
| `KVcur`, common raw-cache prefix | both exact | both exact |
| `kqv_out` | 14,198 / 32,768, max 6.44e-6 | **0 / 32,768** |
| `kqv_back` | 14,216 / 32,768, max 7.15e-6 | **0 / 32,768** |

Thus Q-b passes causally, and unchanged batch head norm, Q RoPE, attention, and inverse RoPE are
M-invariant for this captured row when given exact inputs. The cycle-24 final-logit flip remains,
but its maximum logit difference drops from `0.927542` to `0.133129`. The next unresolved boundary
is the two-stage attention-output projection; it was not captured here because its existing debug
names alter the attempted F16-hook dispatch. That Metal hook is currently also a stub, but V8 should
still use branch-neutral production aliases before changing either projection.

Reproduction:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
for QB in 0 1; do
  mkdir -p "/tmp/lead08_iter13_qb${QB}_dump"
  DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1 \
  DS4_LEAD08_ROWWISE_QB_LAYERS="$QB" DS4_DSPARK_VERIFY_DIST_PROBE=1 \
  DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_DSPARK_DRAFT_METAL_STS=1 \
  DS4_METAL_GRAPH_DUMP_PREFIX="/tmp/lead08_iter13_qb${QB}_dump/dump" \
  DS4_METAL_GRAPH_DUMP_NAME=q_lora_norm,ProdQB,ProdQHeadNorm,Qcur,KVcur,raw_cache,kqv_out,kqv_back \
  DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
    --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
    --jsonl-out "/tmp/lead08_iter13_qb${QB}.jsonl" \
    2>"/tmp/lead08_iter13_qb${QB}.err"
done
```

## Matched K=4 cost

Eight order-balanced, process-first calls compare V6 with and without Q-b at layer 0, then with and
without Q-b across all 43 layers. Dumps, distribution probes, and stage-sync profiling were off.
`DS4_MTP_VERIFY_PROFILE=1` adds the same expert capture/statistics instrumentation described in
artifact 20, so these are profile-instrumented supporting measurements, not production gate times.

| comparison | reference mean | candidate mean | paired delta mean |
|---|---:|---:|---:|
| Q/KV `N=1`, Q-b `0 -> 1` | 55.868 ms | 56.008 ms | +0.140 ms |
| Q/KV `N=43`, Q-b `0 -> 43` | 56.387 ms | 63.852 ms | **+7.465 ms** |

Every retained first call reports the same aggregate selected-byte and unique-count volume: 4.69
GiB, average 16.5 unique experts, range 13-23. This does not prove identical expert identities or
memory locality; A/B later finish with different cycle counts, so changed downstream numerics and
routing remain possible timing confounders even though the first timed call shares the same input.
The layer-0 delta is noise-scale. The all-layer whole-graph delta is large enough that a literal row
dispatch per layer nearly consumes the modeled 43-to-50.5 ms headroom, but it is still a naive
implementation upper-bound estimate: it may include downstream effects and repeats dispatch and
weight traversal rather than using a shared row-independent kernel. It is not added to the
proven-unavoidable lower bound, and profiled absolute times are not compared directly with the 50.5
ms production gate.

Exact timing reproduction for the post-alias binary:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
run_one() {
  label="$1"; qkv="$2"; qb="$3"
  DS4_LEAD08_ROWWISE_QKV_LAYERS="$qkv" DS4_LEAD08_ROWWISE_QB_LAYERS="$qb" \
  DS4_DSPARK_VERIFY_K=4 DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_MTP_VERIFY_PROFILE=1 ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
    --bulk-config "$ONE" --jsonl-out "/tmp/lead08_iter13_v2_${label}.jsonl" \
    2>"/tmp/lead08_iter13_v2_${label}.err"
}
run_one A1 1 0; run_one B1 1 1; run_one B2 1 1; run_one A2 1 0
run_one C1 43 0; run_one D1 43 43; run_one D2 43 43; run_one C2 43 0
awk '/mtp verify profile start=48 tokens=4/{print FILENAME ":" $0; nextfile}' \
  /tmp/lead08_iter13_v2_{A1,B1,B2,A2,C1,D1,D2,C2}.err
```

The exact retained values are in `21_iter13_rowwise_qb_timing.csv`.

## Decision

Close V7 positive for the captured row. V8 should first add branch-neutral production checkpoints
after inverse RoPE, attention-output low projection, and final output projection, then exactify only
the first differing projection. Do not run a full corpus yet. Stop V5 only if an unavoidable exact
kernel cost, rather than this diagnostic row-loop overhead, exhausts the 50.5 ms gate.

## Red-team audit

The first audit returned **NO-COMMIT**: the timing command could not independently set both layer
caps, selected-byte volume was described as identical work, whitespace-prefixed negatives bypassed
the parser check, and the ledger retained stale V7 names/actions. The exact post-alias timing sequence
was rerun; the command, CSV, parser, cost caveats, and ledger were corrected. A live
`DS4_LEAD08_ROWWISE_QB_LAYERS=' -1'` run emitted no activation.

The repeat audit returned **COMMIT**. It independently validated scope/restoration, the Metal stub
assumption, dump-filter neutrality, every stage CSV value, the cycle-24 flip and error reduction,
timing logs and paired arithmetic, reproduction commands, and V8 ordering. Residual risks: one
prompt/row/layer correctness capture; two profile-instrumented timing pairs per mode; unretained
expert identities/locality; and no forced-error test of scope restoration.
