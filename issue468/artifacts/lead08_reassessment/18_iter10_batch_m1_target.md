# Lead 08 iteration 10: existing batch graph as the M=1 target family

Date: 2026-07-17. Verdict: **NO-GO for the existing batch graph as a shared M=1/M=K exactness
strategy; the original hybrid exact-batch build remains untested.**

## Why this branch was reopened

An independent red-team audit found that the dossier had overgeneralized iteration 3. That
iteration rejected one grouped routed-MoE optimization inside the already nonexact batch verifier;
it did not build or falsify Phase B's proposed end-to-end sublinear exact verifier. The cheapest
untested exactness strategy was already named in the plan: make target M=1 and speculative M=K use
one operation family, then re-baseline target throughput.

## Falsifier and audit correction

`DS4_LEAD08_BATCH_M1_TARGET=1` routes every single-token Metal target evaluation through
`metal_graph_verify_suffix_tops(..., n_tokens=1)`. Both M=1 and M=K therefore use
`metal_graph_encode_layer_batch` and `metal_graph_encode_output_head_batch`. With DSpark loaded,
M=1 captures layers 40-42 through the existing batch buffers and pushes row 0 through
`dspark_session_push_batch_hidden`. The shipped path is unchanged when the variable is absent; MTP
is rejected because this diagnostic does not maintain its draft lifecycle.

The first implementation used `metal_graph_prefill_layer_major`, whose singleton output head was
not the M=K batched output head, and left two speculative replay paths on raw decode. The required
post-iteration red-team audit blocked commit. The retained implementation uses the suffix evaluator
for normal target decode and both replay paths, and reports successful per-cycle
`target_batch_m1_evals` / `target_raw_m1_evals` in benchmark JSON. The retained run records 13/0
across 282 speculative cycles. The plain target graph allocates only the verifier logits scratch
when the research flag is enabled.

## Reproduction

The retained 10-prompt exactness corpus uses frontier 48 and 64 emitted tokens per prompt. Exact
inputs are retained as `18_iter10_plain_config.jsonl` and `18_iter10_spec_config.jsonl`; the configs
differ only in `argmax` versus `speculative_argmax` mode.

Model paths on the measurement host:

```text
/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
/Users/lobanov/Projects/ds4/gguf/dspark.gguf
```

Matched commands, with token dump directories omitted only for readability:

```sh
./ds4-spec-bench --metal -m "$MODEL" \
  --bulk-config issue468/artifacts/lead08_reassessment/18_iter10_plain_config.jsonl \
  --jsonl-out /tmp/lead08_iter10_decode_exactness_out_v4.jsonl

DS4_LEAD08_BATCH_M1_TARGET=1 ./ds4-spec-bench --metal -m "$MODEL" \
  --bulk-config issue468/artifacts/lead08_reassessment/18_iter10_plain_config.jsonl \
  --jsonl-out /tmp/lead08_iter10_batch_m1_exactness_out_v4.jsonl

DS4_LEAD08_BATCH_M1_TARGET=1 \
DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1 \
DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
DS4_DSPARK_DRAFT_METAL_STS=1 \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/18_iter10_spec_config.jsonl \
  --jsonl-out /tmp/lead08_iter10_spec_exactness_out_v4.jsonl
```

`DS4_BENCH_DUMP_TOKENS` was set to distinct directories for all three commands. Token IDs were
compared directly. Timings aggregate total emitted tokens over total decode wall time; every mode
emitted 640 tokens without errors. A preceding baseline attempt was discarded because its first
row contained an impossible 946.8-second timing outlier; the clean rerun below has per-prompt decode
times of 1.678-1.718 seconds.

## Result

| mode | exactness against batch-M1 target | aggregate t/s | vs shipped target | vs batch-M1 target |
|---|---:|---:|---:|---:|
| shipped target decode | reference only | 37.902 | reference | +14.9% |
| batch-M1 target | reference | 32.989 | -13.0% | reference |
| batch-M1 + M=K DSpark | **5/10 prompts** | 37.407 | -1.3% | **+13.4%** |

The five first token mismatches occur at emitted positions 29 (`grounded_archive`), 39
(`grounded_observatory`), 21 (`synthesis_incident_json`), 11 (`synthesis_ops_json`), and 53
(`synthesis_timeline_json`). Thus the current end-to-end batch graph and state transition are not
M-invariant even when M=1 and M=K use the same batched layer and output-head functions. This
experiment does not localize the divergence to a particular kernel, cache update, or frontier
transition.

Economics independently fail both useful comparisons. Batch M=1 regresses the target baseline by
13.0%; speculative execution recovers only 13.4% over that slower baseline, below the >=20% gate,
and remains 1.3% slower than shipped target decode.

## Decision

Retain the env-gated path as an exactness/economics falsifier, not as an optimization. Do not
re-baseline target decode onto the existing batch graph and do not claim that a shared call graph
provides exactness by construction. A future same-family design would need M-invariant end-to-end
state transitions, likely requiring the exact/hybrid work Phase B originally identified.

This result does **not** close the unbuilt hybrid strategy: decode-order HC/compressor/attention with
batched sharing only at stages proven invariant. That branch needs an exactness-first stage gate and
must abort before full integration if the rerouted exact stages leave no path to
`verify_ms(4) <= 50.5 ms`.
