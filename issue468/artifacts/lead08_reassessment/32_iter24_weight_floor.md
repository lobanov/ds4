# Lead 08 iteration 24: mapped packed-weight floor

Date: 2026-07-18. Status: **STOP arithmetic reduction; corrected audit COMMIT.** This nonsemantic research
kernel is not a production candidate and does not establish arithmetic-only attribution.

## Design and validity

Independent preflight authorized one final U1 falsifier. A separate mapped IQ2 pair kernel preserves
the production ID-derived expert addresses, every 8-byte q payload and 2-byte scale load for gate
and up inside the same runtime subblock/row loops, K4/24 pairs/18 unique experts, NSG2/NR04 grid,
SIMD reduction, and both output stores. It replaces dequant/dot work with live integer checksums and
optimistically removes activation loads, LUT/threadgroup setup and barrier, and floating arithmetic.

Arms `{production, production duplicate, U1, U1 duplicate}` ran in two fixed-ID strata after 16
warmup rounds and for 32 balanced measured rounds. All 256 samples contain 43 positive finite GPU
intervals (11,008 retained layer intervals). Different sentinels initialized the two U1 outputs;
86/86 layer/stratum cases per U1 selector have zero unwritten words and bit-identical duplicate
outputs. The 86 distinct U1 hashes prove outputs vary by layer/stratum. Production duplicate also
passes 86/86 direct/full bitwise cases.

AIR marks activation and threadgroup arguments `readnone`, retains the ID load, 8 q halfword loads
plus 2 scale loads across the two mapped bases in runtime loops, `air.simd_sum.u.i32`, and two i32
device stores. The isolated entry point contains no floating FMA/multiply/add, conversion, LUT load,
or barrier.

## Results and decision

| stratum | production median | U1 median | paired saving median and 95% CI |
|---|---:|---:|---:|
| overlap | 10.660812 ms | 5.957584 ms | 4.708104 [4.683041, 4.724125] ms |
| dispersed | 10.685125 ms | 5.962084 ms | 4.723667 [4.715042, 4.735250] ms |

Production-duplicate paired relative intervals are `[-0.016,+0.059]%` and
`[-0.073,+0.030]%`. U1-duplicate intervals are `[-0.313,+0.143]%` and
`[-0.024,+1.432]%`; their absolute intervals are `[-0.019,+0.009]` and
`[-0.001,+0.085]` ms, within the +/-0.10 ms control. No outliers were removed;
same-round paired medians use 20,000 deterministic bootstrap draws.

Both saving upper bounds are below 4.74 ms, far short of the predeclared 8.7 ms gate. U1 therefore
**closes arithmetic/dequant reduction economically**, without tuning or a production prototype.
Even this optimistic nonsemantic kernel leaves about 5.96 ms of packed-weight access/address/store
cost and cannot supply the required prize.

The first capture incorrectly omitted `first_row * nb01` from both weight bases, causing every
output tile to reread rows 0-3 and yielding a false 9.25 ms saving. That capture was discarded. The
corrected kernel and retained evidence use the production row addresses; the corrected result above
is binding.

Retained evidence: `32_iter24_weight_floor_samples.csv`, `32_iter24_weight_floor_layers.csv`, and
`32_iter24_weight_floor_summary.csv`; `32_iter24_weight_floor_analysis.py` validates and regenerates
them.

## Reproduction

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
make -j4 ds4-spec-bench
env DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_MAPPED_ALU_PROBE=1 \
  DS4_LEAD08_MAPPED_ALU_DIRECT_REPLAY=1 DS4_LEAD08_MAPPED_ARITH_FLOOR=1 \
  DS4_METAL_DUMP_SOURCE=/tmp/lead08_iter24_full.metal \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE" \
  > /tmp/lead08_iter24_floor.stdout 2> /tmp/lead08_iter24_floor.stderr
xcrun metal -S -gline-tables-only -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_iter24_full.metal -o /tmp/lead08_iter24_full.air.s
python3 issue468/artifacts/lead08_reassessment/32_iter24_weight_floor_analysis.py \
  /tmp/lead08_iter24_floor.stderr \
  issue468/artifacts/lead08_reassessment/32_iter24_weight_floor_samples.csv \
  issue468/artifacts/lead08_reassessment/32_iter24_weight_floor_layers.csv \
  issue468/artifacts/lead08_reassessment/32_iter24_weight_floor_summary.csv
```

The legacy nonzero harness exit after its older fast-math comparison is expected.

The first audit found the missing row offset and required a corrected rerun. The repeat audit
reproduced address-dependent AIR, all controls, byte-identical CSV generation, and the binding STOP
intervals and returned **COMMIT**.
