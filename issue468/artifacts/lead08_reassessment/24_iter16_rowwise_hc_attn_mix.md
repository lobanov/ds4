# Lead 08 iteration 16: row-wise M1 HC attention mixer

Date: 2026-07-18. Status: **causal GO; corrected audit COMMIT.** This is a captured-row
frontier result plus supporting timing, not full-verifier exactness.

## Contract and implementation

V10 tests V9's first exact-input/different-output boundary. The challenger returned **PROCEED**:
Metal routes F16 `hc_attn_fn` (`16384 -> 24`) through the ordinary matvec for M=1 and low-K
`mul_mv_ext` for M=2..8. A new verifier-scoped F16 row helper creates contiguous F32 row views and
calls the existing F16 primitive with `n_tok=1`. `DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS=N` is
saved/restored only inside multi-token suffix verification and affects only attention `hc_attn_fn`.
The FFN mixer, drafter, split, normalization, attention, expansion, and output head are unchanged.

## Correctness

At layer 0, position 104, both arms retain V6-V8; only the candidate sets the mixer cap to one.

| boundary | batched mixer | row-wise M1 mixer |
|---|---:|---:|
| `ProdHCFlat` | exact | exact |
| `ProdHCMix` | 14 / 24, max 6.10e-5 | **exact** |
| consumed `ProdHCSplit[4:24]` | 15 / 20, max 5.96e-8 | **exact** |
| `hc_attn_pre`, `ProdOAOut`, block/residual | exact | exact |
| `hc_attn_post` | 6,269 / 16,384, max 7.45e-9 | **exact** |
| `hc_ffn_pre` | 1,089 / 4,096 | first difference: 752 / 4,096, max 7.45e-9 |

V10 passes causally and closes the attention-side HC path for this row. The first captured
downstream difference is the FFN HC pre path, but its flat/mixer/split inputs are not yet retained;
V11 must localize them before assuming the analogous FFN mixer is at fault. Both arms emit 64 tokens
over 27 cycles with 64 batch-M1 and zero raw-M1 evaluations. One flip remains across 37 comparisons;
the maximum M=K/M1 logit error changes 2.44573 -> 3.21119 as the downstream batch-error pattern
changes, while the canonical M1 replay/generated trajectory remains unchanged.

Exact correctness command:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
for MIX in 0 1; do
  RUN=$(mktemp -d "/tmp/lead08_iter16_mix${MIX}_repro_XXXXXXXX") || exit 1
  DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1 \
  DS4_LEAD08_ROWWISE_QB_LAYERS=1 DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=1 \
  DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS="$MIX" DS4_DSPARK_VERIFY_DIST_PROBE=1 \
  DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_DSPARK_DRAFT_METAL_STS=1 DS4_METAL_GRAPH_DUMP_PREFIX="$RUN/dump" \
  DS4_METAL_GRAPH_DUMP_NAME=ProdHCFlat,ProdHCMix,ProdHCSplit,hc_attn_pre,ProdOAOut,ProdHCBlock,ProdHCResidual,hc_attn_post,hc_ffn_pre,ffn_norm,hc_ffn_post \
  DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
    --jsonl-out "$RUN/out.jsonl" 2>"$RUN/run.err"
done
```

## Supporting timing

Process-first K=4 runs omit dumps and distribution probing and use common
`DS4_MTP_VERIFY_PROFILE=1` expert instrumentation. The layer-0 ABBA pair is noise-scale: reference
55.893 ms, candidate 56.066 ms, paired mean `+0.173 ms`. The surprising all-layer result was
repeated with two balanced CDDC sequences (four pairs total):

| mode | layer execute mean | paired delta mean | paired delta median |
|---|---:|---:|---:|
| prior exactifiers `N=43`, mixer `N=0` | 70.417 ms | reference | reference |
| prior exactifiers `N=43`, mixer `N=43` | 69.164 ms | **-1.252 ms** | **-1.259 ms** |

Every retained first call has 23 cycles and reports the same 4.69 GiB / average 16.5 / range 13-23
aggregate selected-work volume. However, the C and D arms have different accepted-token patterns and
trajectories, so those aggregates do not establish equal downstream expert identities or locality.
The observed sign is route-confounded and uninterpretable for mixer economics; it shows only that
there is no obvious penalty in these profile-instrumented runs. It is not a decision-grade production
speedup and does not change the 50.5 ms lower bound.

Exact timing reproduction:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
run_one() {
  label="$1"; prior="$2"; mix="$3"
  DS4_LEAD08_ROWWISE_QKV_LAYERS="$prior" DS4_LEAD08_ROWWISE_QB_LAYERS="$prior" \
  DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS="$prior" \
  DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS="$mix" DS4_DSPARK_VERIFY_K=4 \
  DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_MTP_VERIFY_PROFILE=1 ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
    --bulk-config "$ONE" --jsonl-out "/tmp/lead08_iter16_${label}.jsonl" \
    2>"/tmp/lead08_iter16_${label}.err"
}
run_one A1 1 0; run_one B1 1 1; run_one B2 1 1; run_one A2 1 0
run_one C1 43 0; run_one D1 43 43; run_one D2 43 43; run_one C2 43 0
run_one C3 43 0; run_one D3 43 43; run_one D4 43 43; run_one C4 43 0
awk '/mtp verify profile start=48 tokens=4/{print FILENAME ":" $0; nextfile}' \
  /tmp/lead08_iter16_{A1,B1,B2,A2,C1,D1,D2,C2,C3,D3,D4,C4}.err
```

Exact retained values are in the timing CSV.

## Decision and audit

Close V10 positive for the captured row. V11 is branch-neutral localization of the FFN-side HC
flat/mixer/split inputs before any compute change. The first mandatory audit validated the code,
correctness captures, timing arithmetic, and V11 ordering but returned **NO-COMMIT** because the
timing sign was attributed to the mixer despite changed downstream routes. That claim and the two
ledger/date inconsistencies it identified were corrected. The repeat audit returned **COMMIT**.
