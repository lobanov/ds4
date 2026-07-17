# Lead 08 iteration 15: HC expansion input localization

Date: 2026-07-17. Status: **REDESIGN; post-iteration audit COMMIT.** No compute changed.

The preflight rejected row-wise HC expansion: M1 and M=K use the same independently indexed
`kernel_dsv4_hc_expand4`. Instead, branch-neutral aliases captured every actual F32 expansion input
with V6-V8 enabled at layer 0, position 104.

| boundary | result |
|---|---:|
| post-steering `ProdHCBlock` | exact, 0 / 4,096 |
| `ProdHCResidual` | exact, 0 / 16,384 |
| `ProdHCFlat` normalized mixer input | exact, 0 / 16,384 |
| `ProdHCMix` full row | first difference, 14 / 24, max 6.10e-5 |
| `ProdHCSplit[0:4]` (unused by expansion) | exact |
| `ProdHCSplit[4:8]` post gate | 2 / 4, max 1.44e-15 |
| `ProdHCSplit[8:24]` combination matrix | 13 / 16, max 5.96e-8 |
| `hc_attn_post` | 6,269 / 16,384, max 7.45e-9 |

HC expansion is not independently implicated. The earlier F16 HC mixer projection creates a latent
state difference even though its weighted pre-attention output was exact; the split post/comb state
is reused after attention and exposes the debt. V10 should row-wise exactify only the HC mixer
projection, then recapture mix, split, `hc_attn_pre`, and `hc_attn_post`. If mix becomes exact but
split remains different, split/sinkhorn is the next boundary.

Reproduction:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
RUN=$(mktemp -d /tmp/lead08_iter15_hcinputs_repro_XXXXXXXX) || exit 1
DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1 \
DS4_LEAD08_ROWWISE_QB_LAYERS=1 DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=1 \
DS4_DSPARK_VERIFY_DIST_PROBE=1 DS4_DSPARK_VERIFY_BATCHED=0 \
DS4_DSPARK_ANCHOR_REUSE=1 DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 \
DS4_DSPARK_DRAFT_METAL=1 DS4_DSPARK_DRAFT_METAL_STS=1 \
DS4_METAL_GRAPH_DUMP_PREFIX="$RUN/dump" \
DS4_METAL_GRAPH_DUMP_NAME=ProdHCBlock,ProdHCResidual,ProdHCFlat,ProdHCMix,ProdHCSplit,hc_attn_post \
DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
  --jsonl-out "$RUN/out.jsonl" 2>"$RUN/run.err"
```

The aliases do not contain any path-debug substring; retained stderr emitted no legacy Q or
attention-output dumps. This iteration has no timing claim because it adds only dump hooks.

The first audit validated all V2 evidence and V10 ordering but returned **NO-COMMIT** because the
worklog omitted the executed result and the reproduction directory could retain stale dumps. After
adding the result, a collision-safe `mktemp` directory, and the normalized-input checkpoint, the
repeat audit returned **COMMIT**. Residual risks remain one row/prompt/layer and an untested V10
mixer cost; expansion is not exonerated while its consumed split input differs.
