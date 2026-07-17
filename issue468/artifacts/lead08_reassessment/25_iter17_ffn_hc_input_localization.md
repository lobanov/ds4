# Lead 08 iteration 17: FFN HC input localization

Date: 2026-07-18. Status: **REDESIGN; corrected audit COMMIT.** This is a captured-row
frontier result with no timing claim.

## Contract

V11 localizes the first difference after V10 without changing compute. The preflight challenger
returned **PROCEED** for the complete causal cone of `hc_ffn_pre`: post-attention HC residual
(`16384`) -> plain row RMS (`16384`) -> F16 `hc_ffn_fn` mix (`24`) -> split state (`24`) ->
weighted sum / `hc_ffn_pre` (`4096`). It rejected a synthetic combined FFN block capture because
the optimized HC-post path consumes routed F32 plus shared F16/F32 directly; forcing `ffn_out` would
materialize a sum and change that path.

Four case-sensitive aliases avoid the lowercase legacy dump substrings that alter FFN branch
selection. Flat and mix are immediately post-producer, and split immediately follows its fused
producer. Residual is retained at the FFN-entry consumer boundary after a read-only RMS dispatch and
is corroborated by the producer-adjacent `hc_attn_post` capture. The restored-frontier comparator
retains V6-V10 at layer 0 and compares the M=K row at position 104 with its batch-M1 replay.

## Result

| boundary | different values | max absolute difference |
|---|---:|---:|
| `hc_attn_post` | 0 / 16,384 | 0 |
| `ProdFFNHCResidual` | 0 / 16,384 | 0 |
| `ProdFFNHCFlat` | 0 / 16,384 | 0 |
| `ProdFFNHCMix` | **14 / 24** | **1.52587891e-5** |
| `ProdFFNHCSplit` | 16 / 24 | 1.34110451e-7 |
| `hc_ffn_pre` | 752 / 4,096 | 7.4505806e-9 |

The first exact-input/different-output boundary is the F16 FFN HC mixer. Split state differs in two
of four pre weights, zero of four post weights, and 14 of 16 combination values; these are inherited
from the mixer and do not independently implicate split/sinkhorn. V11 closes as a localization result.
V12 may test the existing F16 rows-as-M1 helper only at `hc_ffn_fn`, with V6-V10 held fixed.

The run emitted 64 tokens over 27 cycles with 64 batch-M1 evaluations, zero raw-M1 evaluations,
37 logit comparisons, and one flip. The maximum M=K/M1 logit error remained 3.21119. These global
values describe the unchanged V10 composition and are not a V11 outcome criterion.

## Reproduction

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
RUN=$(mktemp -d /tmp/lead08_iter17_ffn_inputs_repro_XXXXXXXX) || exit 1
DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1 \
DS4_LEAD08_ROWWISE_QB_LAYERS=1 DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=1 \
DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS=1 DS4_DSPARK_VERIFY_DIST_PROBE=1 \
DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1 \
DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
DS4_DSPARK_DRAFT_METAL_STS=1 DS4_METAL_GRAPH_DUMP_PREFIX="$RUN/dump" \
DS4_METAL_GRAPH_DUMP_NAME=hc_attn_post,ProdFFNHCResidual,ProdFFNHCFlat,ProdFFNHCMix,ProdFFNHCSplit,hc_ffn_pre,ffn_norm,hc_ffn_post \
DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
  --jsonl-out "$RUN/out.jsonl" 2>"$RUN/run.err"
```

Exact retained values are in the stage CSV. The first mandatory audit independently reproduced every
CSV value and validated branch neutrality, causal attribution, and V12 ordering, but returned
**NO-COMMIT** because it described the residual alias as producer-adjacent. The corrected
consumer-boundary wording passed repeat audit: **COMMIT**.
