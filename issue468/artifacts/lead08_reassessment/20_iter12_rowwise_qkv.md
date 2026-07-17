# Lead 08 iteration 12: row-wise M=1 Q/KV projection

Date: 2026-07-17. Status: **COMMIT after post-iteration red-team audit.** This is a causal
localization result, not full verifier exactness.

## Contract and preflight

V6 asked whether the first iteration-11 row-0 difference came from changing Q8 projection shape
from the M=1 matvec to the 2-8-row extended kernel. The required preflight returned **REDESIGN,
then proceed**. It required:

- suffix-verifier-only scope, because the batch layer is also used by prompt prefill;
- a numeric layer cap, with `N=1` for causality and `N=43` for naive implementation cost;
- unchanged normalization, Q-b, RoPE, caches, attention, FFN, and output head;
- production-path dumps that do not request `Qraw` or `Qnorm`, because those names disable the
  fused Q-b path;
- naive row-dispatch cost treated as an implementation upper bound, not an unavoidable lower bound.

`DS4_LEAD08_ROWWISE_QKV_LAYERS=N` is saved and restored only inside
`metal_graph_verify_suffix_tops` and engages only for `n_tokens > 1`. For layers `< N`, Q-a and KV
create row-contiguous input/output views and call the existing named Q8 primitive with
`n_tokens=1`. Metal's paired Q/KV hook is inactive, so this is the same primitive used by batch M=1.
An activation line confirms verifier scope; prompt prefill and the M=1 comparator remain unchanged.

## Correctness result

The retained `code_sort_pairs` cycle 24 capture again starts at position 104 and compares only M=K
row 0 with complete M=1. With `N=1`, `attn_norm`, Q-a, KV, and both Q/KV normalizations become
bit-identical. The entire KV RoPE/storage path remains identical. The first captured difference moves
to production-path `Qcur`, the fused Q-b + head-normalization + RoPE result:

| boundary | result |
|---|---:|
| `attn_norm` | 0 / 4096 different |
| `q_lora` | **0 / 1024** different, down from 799 |
| `KVraw` | **0 / 512** different, down from 405 |
| `q_lora_norm`, `KVnorm` | both bit-identical |
| `Qcur` | first difference: 25,988 / 32,768, max 1.91e-6 |
| `KVrope`, `KVcur`, common `raw_cache` prefix | all bit-identical |

The final row logits still flip at zero-based cycle 24 (`max_abs=0.927542`), as expected because
Q-b and later stages remain M-dependent. V6 passes causally; it does not claim exact logits.

Self-contained reproduction command, using artifact 19's one-prompt config. The retained CSV was
assembled from two actual runs with subsets of this filter; neither requested `Qraw` or `Qnorm`:

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
export DS4_LEAD08_BATCH_M1_TARGET=1 DS4_LEAD08_ROWWISE_QKV_LAYERS=1
export DS4_DSPARK_VERIFY_DIST_PROBE=1 DS4_DSPARK_VERIFY_BATCHED=0
export DS4_DSPARK_ANCHOR_REUSE=1 DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1
export DS4_DSPARK_DRAFT_METAL=1 DS4_DSPARK_DRAFT_METAL_STS=1
mkdir -p /tmp/lead08_iter12_code_sort_pairs_dump_repro
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/lead08_iter12_code_sort_pairs_dump_repro/dump \
  DS4_METAL_GRAPH_DUMP_NAME=attn_norm,q_lora,KVraw,q_lora_norm,KVnorm,Qcur,KVrope,KVcur,raw_cache,kqv_out \
  DS4_METAL_GRAPH_DUMP_LAYER=0 DS4_METAL_GRAPH_DUMP_POS=104 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter12_code_sort_pairs_out_v2.jsonl
```

As in iteration 11, ordinary M=K tensors contain two row-contiguous rows and M=1 tensors one;
`raw_cache` is a position-major prefix. The CSV compares the first M=1-sized F32 prefix.

## Matched K=4 cost

The first full-corpus A/B/C/C/B/A attempt exposed within-run drift, so its aggregate means were
rejected. The retained timing uses four order-balanced process-first `start=48, tokens=4` verifier
calls per mode: two from that sequence and two from a second one-prompt A/B/C/C/B/A sequence. There
was no temperature measurement or cooldown criterion. All runs omit distribution probing, dumps,
and per-stage synchronization. `DS4_MTP_VERIFY_PROFILE=1` does add common expert-profile
instrumentation: each layer captures router IDs/weights before completion and reads statistics after
the layer command synchronizes. Therefore the layer interval is matched but profile-instrumented,
not an uninstrumented production verifier time.

| mode | layer execute mean | median | paired delta mean | paired delta median |
|---|---:|---:|---:|---:|
| baseline | 55.924 ms | 55.925 ms | reference | reference |
| `N=1` | 56.004 ms | 56.001 ms | +0.080 ms | +0.083 ms |
| `N=43` | 56.488 ms | 56.525 ms | **+0.564 ms** | **+0.687 ms** |

The `N=1` paired layer deltas range from -0.215 to +0.370 ms, so its +0.08 ms center is noise-scale.
The first baseline total has a cold encode outlier, so paired total medians are retained only as a
cross-check: +0.081 ms for `N=1` and +0.810 ms for `N=43`. Full-corpus `N=43` first calls also select
4.59 GiB versus 4.65 GiB for baseline, while all one-prompt modes select 4.69 GiB. Thus the observed
`N=43` delta includes changed downstream routed work and is not an isolated Q/KV kernel cost.

Exact timing reproduction and order (`A=0`, `B=1`, `C=43`):

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
FULL=issue468/artifacts/lead08_reassessment/18_iter10_spec_config.jsonl
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
run_full() {
  label="$1"; layers="$2"
  DS4_LEAD08_ROWWISE_QKV_LAYERS="$layers" DS4_DSPARK_VERIFY_K=4 \
  DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_MTP_VERIFY_PROFILE=1 DS4_DSPARK_TIMING=1 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$FULL" \
    --jsonl-out "/tmp/lead08_iter12_timing_${label}.jsonl" \
    2>"/tmp/lead08_iter12_timing_${label}.err"
}
run_one() {
  label="$1"; layers="$2"
  DS4_LEAD08_ROWWISE_QKV_LAYERS="$layers" DS4_DSPARK_VERIFY_K=4 \
  DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
  DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
  DS4_MTP_VERIFY_PROFILE=1 \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
    --jsonl-out "/tmp/lead08_iter12_micro_${label}.jsonl" \
    2>"/tmp/lead08_iter12_micro_${label}.err"
}
run_full A1 0; run_full B1 1; run_full C1 43
run_full C2 43; run_full B2 1; run_full A2 0
run_one A3 0; run_one B3 1; run_one C3 43
run_one C4 43; run_one B4 1; run_one A4 0
awk '/mtp verify profile start=48 tokens=4/{print FILENAME ":" $0; nextfile}' \
  /tmp/lead08_iter12_timing_{A1,B1,C1,C2,B2,A2}.err \
  /tmp/lead08_iter12_micro_{A3,B3,C3,C4,B4,A4}.err
```

The roughly 56 ms profiled baseline includes common router capture/statistics instrumentation and is
not directly comparable with the 50.5 ms production gate. The noisy observed all-layer delta is
about +0.6 ms in matched layer execution and +0.8 ms in paired total median, but it also includes
downstream-work differences. Treat it only as an upper-bound estimate for this naive implementation:
a future row-independent kernel can preserve reduction order while sharing dispatch/weights. No new
unavoidable cost was proved, so the retained optimistic feasibility floor remains about 43 ms rather
than increasing by the row-dispatch overhead.

## Red-team audit

The mandatory post-iteration audit initially returned **NO-COMMIT** because the timing record called
the samples thermally controlled without temperature/cooldown evidence, omitted the exact retained
sequence, treated the profiler as a natural production boundary, and did not expose changed selected
work. After the corrections above, the final audit returned **COMMIT**. It independently validated
suffix-only scope and restoration, row-view lifetimes and primitive identity, the production-path
capture, all CSV values, flip reproduction, command completeness, timing arithmetic, and ledger
logic.

Residual risks are retained rather than generalized away: there is no forced-error unit test for
scope restoration; negative layer-count text currently parses to the 43-layer clamp; correctness is
one row in one prompt at layer 0; and four noisy profile-instrumented timing samples are supporting,
not production decision-grade, evidence.

## Decision

Close V6 positive. The next first-boundary falsifier is row-wise production-path Q-b, per-head norm,
and RoPE at layer 0, leaving KV and every downstream stage unchanged. Repeat the same boundary capture
and matched `N=1`/`N=43` cost discipline. Do not run a full exactness corpus until the layer path is
locally exact, and do not close V5 merely because the naive row-dispatch implementation exceeds the
50.5 ms goal.
