# Lead 08 iteration 14: row-wise M=1 attention output-B

Date: 2026-07-17. Status: **causal GO; post-iteration audit COMMIT.** This is a one-row,
layer-0 localization result, not full exactness or a production implementation.

## Contract and implementation

V8 tests the first unresolved boundary after V7. The pre-experiment challenger returned
**PROCEED**: Metal's F16 attention-output hook is a stub, the output-A low kernel below 32 tokens is
independently indexed by token/group, and output-B is the generic Q8 matmul that changes kernel shape
between M=1 and M=K.

Branch-neutral `ProdOALow` and `ProdOAOut` aliases do not contain the lowercase `attn_low` or
`attn_out` substrings used by `attn_out_debug`; retained stderr shows neither legacy dump fired.
`DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=N` is saved/restored only in multi-token suffix verification.
After the unchanged combined batch call, it overwrites only `batch_attn_out` using the existing named
M1 Q8 primitive once per row. V6/V7 are enabled in both arms.

## Correctness

At layer 0, position 104, M=K row 0 is compared with complete M1:

| boundary | batch output-B | row-wise output-B |
|---|---:|---:|
| `kqv_back` | exact | exact |
| `ProdOALow` | **exact** | **exact** |
| `ProdOAOut` | 3,319 / 4,096, max 1.01e-6 | **exact** |
| `hc_attn_post` | 8,801 / 16,384, max 2.24e-8 | first difference: 6,269 / 16,384, max 7.45e-9 |

This proves output-A low is M-invariant and output-B is the next causal debt for this row. The
overlay passes and moves the frontier to HC expansion. The cycle-24 final-logit flip remains; its
max error changes from 0.133129 to 1.00982 because later divergent stages produce a different
downstream M=K batch-error pattern. The canonical M1 replay/generated-token trajectory is unchanged;
this is not a regression claim about an exact implementation.

Correctness reproduction:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
for OAB in 0 1; do
  mkdir -p "/tmp/lead08_iter14_oab${OAB}_dump_v1"
  DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1 \
  DS4_LEAD08_ROWWISE_QB_LAYERS=1 DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS="$OAB" \
  DS4_DSPARK_VERIFY_DIST_PROBE=1 DS4_DSPARK_VERIFY_BATCHED=0 \
  DS4_DSPARK_ANCHOR_REUSE=1 DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 \
  DS4_DSPARK_DRAFT_METAL=1 DS4_DSPARK_DRAFT_METAL_STS=1 \
  DS4_METAL_GRAPH_DUMP_PREFIX="/tmp/lead08_iter14_oab${OAB}_dump_v1/dump" \
  DS4_METAL_GRAPH_DUMP_NAME=kqv_back,ProdOALow,ProdOAOut,hc_attn_post,hc_ffn_pre,ffn_norm,hc_ffn_post \
  DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
    --jsonl-out "/tmp/lead08_iter14_oab${OAB}_v1.jsonl" \
    2>"/tmp/lead08_iter14_oab${OAB}_v1.err"
done
```

## Supporting timing

Eight process-first K=4 calls use ABBA/CDDC order. Dumps and distribution probes are off;
`DS4_MTP_VERIFY_PROFILE=1` adds common expert-profile instrumentation. Every first call reports the
same 4.69 GiB and 16.5 average unique-expert volume, not necessarily identical identities/locality.

| comparison | reference mean | candidate mean | paired delta mean |
|---|---:|---:|---:|
| prior caps `N=1`, output-B `0 -> 1` | 55.914 ms | 56.007 ms | +0.093 ms |
| prior caps `N=43`, output-B `0 -> 43` | 63.515 ms | 70.505 ms | **+6.990 ms** |

The candidate deliberately executes batch output-B and then repeats output-B rowwise. Therefore the
all-layer whole-graph delta is redundant-work upper-bound evidence with downstream numeric/routing
confounds. It cannot update the 50.5 ms lower bound. Decision-grade timing requires a low-only batch
API followed by exact output-B, or a combined API that can skip its batch output-B.

Exact timing command:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
run_one() {
  label="$1"; qkv="$2"; qb="$3"; oab="$4"
  DS4_LEAD08_ROWWISE_QKV_LAYERS="$qkv" DS4_LEAD08_ROWWISE_QB_LAYERS="$qb" \
  DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS="$oab" DS4_DSPARK_VERIFY_K=4 \
  DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_MTP_VERIFY_PROFILE=1 ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
    --bulk-config "$ONE" --jsonl-out "/tmp/lead08_iter14_${label}.jsonl" \
    2>"/tmp/lead08_iter14_${label}.err"
}
run_one A1 1 1 0; run_one B1 1 1 1; run_one B2 1 1 1; run_one A2 1 1 0
run_one C1 43 43 0; run_one D1 43 43 43; run_one D2 43 43 43; run_one C2 43 43 0
```

Raw retained values are in the timing CSV.

## Decision and audit

Close V8 positive for the captured row. V9 addresses the first captured unresolved boundary at HC
expansion, starting with branch-neutral captures of every input before any compute change. Do not
build the clean output-B API until the remaining correctness frontier survives.

The mandatory audit returned **COMMIT** after one wording correction: the changed final error is a
different M=K batch-error pattern, not a generated-token trajectory. It independently validated
scope/restoration, overlay placement and dimensions, Metal stub assumptions, alias neutrality, every
stage/flip/timing value, commands, and V9 ordering. Residual risks: one row/prompt/layer; unisolated
HC inputs; deliberately redundant overlay timing; two profile-instrumented timing pairs with routing
confounds; backend-specific stub assumptions; and no forced-error restoration test.
