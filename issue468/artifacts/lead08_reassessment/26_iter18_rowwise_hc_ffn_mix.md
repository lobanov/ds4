# Lead 08 iteration 18: row-wise M1 HC FFN mixer

Date: 2026-07-18. Status: **causal GO; audit COMMIT.** This is a captured-row
frontier result plus isolated diagnostic timing, not full-verifier exactness.

## Contract and implementation

V12 tests V11's exact-input/different-output boundary. The challenger returned **PROCEED**: reuse
the audited F16 rows-as-M1 helper only for batched `hc_ffn_fn` (`16384 -> 24`). The new numeric cap
`DS4_LEAD08_ROWWISE_HC_FFN_MIX_LAYERS` is saved, set, announced, and restored only inside
multi-token suffix verification. Attention, split, normalization, router, experts, drafter, and
output head are unchanged.

Both correctness arms retain V6-V10 at layer 0 and differ only in the new cap (`0` versus `1`).
Branch-neutral router captures stop at logits/probabilities/top-k/weights; no `ffn_out` or
`ffn_shexp` dump is requested.

## Correctness

| boundary | batched FFN mixer | row-wise M1 FFN mixer |
|---|---:|---:|
| residual / flat RMS | exact | exact |
| `ProdFFNHCMix` | 14 / 24, max 1.53e-5 | **exact** |
| `ProdFFNHCSplit` | 16 / 24 | **exact** |
| `hc_ffn_pre` | 752 / 4,096 | **exact** |
| `ffn_norm` | 701 / 4,096 | **exact** |
| `ffn_moe_logits` | 171 / 256 | first difference: 186 / 256, max 9.54e-7 |
| `ffn_moe_topk` | exact | exact |

V12 passes causally and moves the captured frontier through FFN normalization. The first remaining
operation is the F16 router projection; router probabilities and two of six scaled weights differ,
but the selected top-6 expert IDs remain exact. `hc_ffn_post` still differs and inherits earlier
router/expert state, so it is not independently implicated.

Both arms emit 64 tokens over 27 cycles with 64 batch-M1 and zero raw-M1 evaluations, 37 comparisons,
and one flip. The maximum M=K/M1 logit error changes 3.21119 -> 3.28608. These global values are not
V12 pass/fail criteria.

Exact correctness command:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
for MIX in 0 1; do
  RUN=$(mktemp -d "/tmp/lead08_iter18_mix${MIX}_repro_XXXXXXXX") || exit 1
  DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1 \
  DS4_LEAD08_ROWWISE_QB_LAYERS=1 DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=1 \
  DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS=1 \
  DS4_LEAD08_ROWWISE_HC_FFN_MIX_LAYERS="$MIX" DS4_DSPARK_VERIFY_DIST_PROBE=1 \
  DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_DSPARK_DRAFT_METAL_STS=1 DS4_METAL_GRAPH_DUMP_PREFIX="$RUN/dump" \
  DS4_METAL_GRAPH_DUMP_NAME=ProdFFNHCResidual,ProdFFNHCFlat,ProdFFNHCMix,ProdFFNHCSplit,hc_ffn_pre,ffn_norm,ffn_moe_logits,ffn_moe_probs,ffn_moe_topk,ffn_moe_weights_scaled,hc_ffn_post \
  DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
    --jsonl-out "$RUN/out.jsonl" 2>"$RUN/run.err"
done
```

## Diagnostic timing

Eight process-first runs use the synchronized layer-0 stage profiler and stop the measured envelope
at FFN `hc_pre`, before router/expert work. The reference mean is 0.2195 ms and candidate mean is
0.1903 ms; four paired deltas are -0.042, -0.043, -0.011, and -0.021 ms (mean -0.0293 ms, median
-0.0315 ms). This supports no obvious local penalty for the naive row loop. Profiler synchronization
changes scheduling, so this is not production verifier cost, an all-layer speedup, or a lower-bound
update.

Exact timing reproduction:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
run_one() {
  label="$1"; mix="$2"
  DS4_LEAD08_ROWWISE_QKV_LAYERS=1 DS4_LEAD08_ROWWISE_QB_LAYERS=1 \
  DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=1 DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS=1 \
  DS4_LEAD08_ROWWISE_HC_FFN_MIX_LAYERS="$mix" DS4_DSPARK_VERIFY_K=4 \
  DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_DSPARK_DRAFT_METAL_STS=1 DS4_METAL_LAYER_STAGE_PROFILE=1 \
  DS4_METAL_LAYER_STAGE_PROFILE_LAYER=0 ./ds4-spec-bench --metal -m "$MODEL" \
    --dspark "$DSPARK" --bulk-config "$ONE" --jsonl-out "/tmp/lead08_iter18_${label}.jsonl" \
    2>"/tmp/lead08_iter18_${label}.err"
}
run_one A1 0; run_one B1 1; run_one B2 1; run_one A2 0
run_one A3 0; run_one B3 1; run_one B4 1; run_one A4 0
for label in A1 B1 B2 A2 A3 B3 B4 A4; do
  awk '/metal layer stage part=ffn layer=0 pos=48 tokens=4 hc_pre=/{print; nextfile}' \
    "/tmp/lead08_iter18_${label}.err"
done
```

Exact retained values and run order are in the timing CSV.

## Decision

Close V12 positive for this captured row. The next action is not another reflexive row helper.
Inventory a generic same-accumulation batched-F16 mechanism and its cost across the attention mixer,
FFN mixer, and newly exposed router projection; only then choose an implementation. The mandatory
audit independently reproduced all F32/I32 boundaries and timing arithmetic, validated scope and
causal attribution, and returned **COMMIT**.
