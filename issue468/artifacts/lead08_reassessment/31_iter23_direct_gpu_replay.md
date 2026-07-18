# Lead 08 iteration 23: direct mapped-pair GPU replay

Date: 2026-07-18. Status: **VALID arithmetic/issue-sensitive selector; audit COMMIT.**
This is attribution evidence, not a production speedup or quality result.

## Predeclared question

Iteration 22 was invalid because its production/companion-zero controls were not stable enough.
The independent preflight returned **REDESIGN** for one final separator. Replay the exact mapped K4
`tiny_pair_mv` gate/up encoding directly, with one command buffer per layer and Metal
`GPUStartTime`/`GPUEndTime` as the primary interval. Remove activation, down projection, profiler,
and per-layer logging from the timed command. A valid positive result may authorize exactly one
maximum-removable-arithmetic upper-bound kernel; it cannot authorize a production prototype or a
traffic/latency claim.

Validity requires all of the following:

- literal production selector 0, an identical production duplicate selector 2, companion actual
  zero selector 1, and companion R32/R128;
- fixed K4, 24 physical pairs, 18 unique experts, mapped offsets, F32 gate/up, NSG 2, NR0 4, and
  fixed per-layer IDs in overlap and coprime-dispersed strata;
- two complete ten-round warmup schedules discarded, followed by 30 measured Latin/reversed rounds
  per stratum, with each arm in each order position six times;
- all 43 command buffers present with positive finite GPU intervals in every sample;
- bit-exact full routed output and explicit direct gate/up replay over 43 layers times two strata;
- production/duplicate paired 95% interval within +/-1% and production/companion-zero within +/-2%
  in each stratum;
- R32 lower interval at least +25% in both strata and monotonic positive R128 for a positive
  arithmetic/issue-sensitive classification.

## Implementation and retained evidence

`DS4_LEAD08_MAPPED_ALU_DIRECT_REPLAY=1` returns immediately after the existing
`ds4_gpu_encode_mul_mv_id_pair` gate/up encoding. Each layer is committed alone. The harness reads
the completed command buffer's GPU and kernel timestamps, then emits one aggregate record after all
43 layers; that record retains all 43 GPU intervals. No diagnostic print occurs inside a timed
command buffer.

The fidelity harness first runs the full routed candidate and then the direct candidate. Gate/up are
read after direct replay, while weighted mid and routed output remain from the full run. Selectors
`{2,1,32,128}` each pass 86/86 cases with zero differing gate, up, mid, or output F32 bits and zero
argmax flips.

The deterministic analysis validates 300 measured samples and 12,900 positive finite per-layer GPU
intervals. Every arm has 30 samples per stratum and occupies every order position six times. The
layer-level evidence is in `31_iter23_direct_gpu_layers.csv`; aggregate GPU, kernel, and wall
intervals are in `31_iter23_direct_gpu_samples.csv`; summary statistics are in
`31_iter23_direct_gpu_summary.csv`; the validating parser is
`31_iter23_direct_gpu_replay_analysis.py`.

Dumped-source AIR preserves two distinct entry points. Literal production has no probe argument and
no probe FMA loop. The companion loads runtime rounds/multiply/add from its separate 16-byte
buffer-7 argument, contains two `air.fma.f32` calls, and compares its incremented counter with the
runtime rounds value on the backedge.

## Results

All values below are the sum of 43 command-buffer GPU intervals. Paired intervals resample
same-round relative differences with 20,000 deterministic bootstrap draws. R32/R128 comparisons use
companion-zero, as predeclared.

| stratum / arm | median GPU ms | MAD ms | paired relative median and 95% CI |
|---|---:|---:|---:|
| overlap production | 10.733375 | 0.009271 | reference |
| overlap duplicate | 10.734396 | 0.013438 | +0.021% [-0.064, +0.099] vs production |
| overlap companion-zero | 10.864791 | 0.010959 | +1.254% [+1.149, +1.341] vs production |
| overlap R32 | 15.846417 | 0.020396 | +45.762% [+45.652, +45.934] vs zero |
| overlap R128 | 35.040125 | 0.031792 | +222.320% [+222.133, +222.528] vs zero |
| dispersed production | 10.755771 | 0.011437 | reference |
| dispersed duplicate | 10.760333 | 0.014771 | +0.057% [-0.071, +0.154] vs production |
| dispersed companion-zero | 10.876333 | 0.011979 | +1.153% [+1.054, +1.195] vs production |
| dispersed R32 | 15.856291 | 0.016667 | +45.795% [+45.660, +46.014] vs zero |
| dispersed R128 | 35.037563 | 0.018771 | +222.066% [+221.761, +222.578] vs zero |

The production duplicate passes the +/-1% gate and companion-zero passes the +/-2% gate in both
strata. Both R32 lower bounds exceed +45%, and R128 is strongly positive. T2c therefore classifies
the live mapped pair carrier as **arithmetic/issue-sensitive under added dependent FMA work**. This
is deliberately one-sided: it does not establish that production is dequant-compute-bound, nor
does it rule out memory, address, occupancy, or latency limits.

## Decision and economics

Close T2c **positive as a selector**. It authorizes exactly one preflighted upper-bound kernel that
removes the maximum safely removable arithmetic while retaining packed-byte loads, address
calculation, dispatch geometry, and output stores, with AIR proof of what remains. Do not tune a
production kernel yet.

The direct literal-production carrier is only about 10.73-10.76 ms across all 43 layers. Supplying
the full 8.7 ms/cycle prize from it requires an approximately 81% reduction. This supersedes the
iteration-22 percentage estimate derived from the profiler-instrumented 18.4 ms stage, while
preserving the same absolute 8.7 ms gate. The upper-bound experiment must show a paired saving lower
bound of at least 8.7 ms in both strata; otherwise arithmetic reduction closes economically even
though T2c proves sensitivity to added arithmetic.

## Reproduction

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl

make -j4 ds4-spec-bench
env DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_MAPPED_ALU_PROBE=1 \
  DS4_LEAD08_MAPPED_ALU_DIRECT_REPLAY=1 \
  DS4_METAL_DUMP_SOURCE=/tmp/lead08_iter23_full.metal \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
  > /tmp/lead08_iter23_direct_final.stdout \
  2> /tmp/lead08_iter23_direct_final.stderr

python3 issue468/artifacts/lead08_reassessment/31_iter23_direct_gpu_replay_analysis.py \
  /tmp/lead08_iter23_direct_final.stderr \
  issue468/artifacts/lead08_reassessment/31_iter23_direct_gpu_layers.csv \
  issue468/artifacts/lead08_reassessment/31_iter23_direct_gpu_samples.csv \
  issue468/artifacts/lead08_reassessment/31_iter23_direct_gpu_summary.csv

xcrun metal -S -gline-tables-only -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_iter23_full.metal -o /tmp/lead08_iter23_full.air.s
```

The harness's legacy nonzero exit after its older fast-math comparison is expected.

The independent post-iteration audit regenerated all three CSVs byte-identically, independently
checked all 344 fidelity detail records and both AIR entry points, reproduced every interval and
threshold, confirmed the direct-path boundary and 80.9-81.1% calculation, and returned **COMMIT**.
Metal's kernel start/end properties describe driver scheduling rather than GPU shader execution;
they remain supporting records only and no claim is based on them.
