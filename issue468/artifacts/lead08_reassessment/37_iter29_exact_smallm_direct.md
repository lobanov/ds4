# Lead 08 iteration 29: exact small-M direct family

Date: 2026-07-18  
Hypothesis: V15 Stage B1 direct capability  
Verdict: **PASS direct capability; final Stage B remains open**

## Scope decision

Independent preflight rejected treating the entire remaining Stage B as one auditable coding
iteration. Iteration 29 isolates the highest-risk direct-family question: can the Stage-A Q8/F16
structure extend from M=2 to static M=2..8 specializations and remain word-exact at every real
dense tensor shape? C5 capture, the nonredundant batch-low/output-B seam, fixed-work M4 timing, and
the final `upper_delta <= 7.5 ms` decision remain mandatory Stage B2 work in iteration 30.

This split does not weaken the final Stage-B contract. Passing B1 is not a speed result, does not
admit graph rollout, and does not satisfy output-B seam validity. Direct output-B here consumes its
real `(M,4096,8192)` weight tensor and synthetic low input only.

## Result

The one research host operation now selects fourteen static pipelines: Q8_0 and F16 for every
M=2..8. Every candidate retains `NR0=2`, Q8 NSG 4 or F16 NSG 8, 256 bytes of reduction scratch,
and exactly one `(ceil(N/2),1,1)` dispatch. No production graph callsite changed.

The decisive matrix covers 43 layers x seven sites x seven M values x C0-C4 = 10,535 cases:

| Site | Format | `(N,K)` | Cases |
|---:|---|---:|---:|
| 0 HC attention mixer | F16 | `(24,16384)` | 1,505 |
| 1 attention Q-a | Q8_0 | `(1024,4096)` | 1,505 |
| 2 attention KV | Q8_0 | `(512,4096)` | 1,505 |
| 3 attention Q-b | Q8_0 | `(32768,1024)` | 1,505 |
| 4 attention output-B | Q8_0 | `(4096,8192)` | 1,505 |
| 5 HC FFN mixer | F16 | `(24,16384)` | 1,505 |
| 6 FFN router | F16 | `(256,4096)` | 1,505 |

Each case compares one candidate dispatch against M unchanged production `n_tok=1` calls in the
same command buffer. All 10,535 cases pass with zero differing F32 words, including signed zero,
and zero maximum absolute difference. Counts are 6,020 Q8 and 4,515 F16, 1,505 per M, 2,107 per
corpus, and 245 per layer. Only the complete unfiltered matrix can report `DIRECT_BIT_EXACT` and
exit zero; filtered bring-up remains nonzero `FILTERED_PASS`.

The evolved deterministic validator reconstructs every input/hash and rejects missing, duplicate,
mis-shaped, wrong-geometry, or nonexact rows. It still validates the committed Stage-A matrix by
default and selects this matrix with `--phase direct`.

## AIR topology

Runtime compilation and `metal -S` both expose all fourteen host and accumulator specializations.
The retained AIR parser finds for each specialization exactly one raw device-load instruction site,
only grid-x consumption, and one unique widest backedge enclosing that raw load. Q8 also has one
scale-load instruction site before its token arithmetic; F16's half4 load/conversion precedes its
token dot loop. The compact line map is `37_iter29_exact_smallm_direct_air.csv`.

This is a logical topology check. Unspecialized AIR and public Metal reflection do not expose
decisive hardware register occupancy or spill traffic. In particular, F16 M=8 keeps a large token
activation array live and may use private memory despite passing topology. Only Stage B2 timing can
decide whether that cost is acceptable. No physical DRAM or no-spill claim is made.

## Independent audit

The post-iteration red team returned `COMMIT`. It forced a clean build, revalidated both the 10,535
direct cases and the prior 430 Stage-A cases, regenerated the fourteen-specialization AIR map
byte-identically, and reproduced invalid-phase and filtered-run fail-closed behavior. Source review
confirmed real weight offsets, bounds, contiguous row views, M independent production M1 references
followed by one candidate in one command buffer, distinct outputs, and no production graph callsite
change. The auditor agreed that the evidence supports logical topology only and explicitly leaves
registers/spills, physical DRAM traffic, output-B seam validity, timing/economics, final Stage B, and
rollout unresolved.

## Reproduction

```sh
make -j4 ds4-spec-bench
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf

DS4_LEAD08_EXACT_SMALLM_GATE=1 \
DS4_LEAD08_EXACT_SMALLM_PHASE=direct \
DS4_LEAD08_EXACT_SMALLM_CSV=/tmp/lead08_iter29_cases.csv \
DS4_METAL_DUMP_SOURCE=/tmp/lead08_iter29.metal \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter29_unused.jsonl \
  2>/tmp/lead08_iter29.err

xcrun metal -S -gline-tables-only \
  -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 \
  -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_iter29.metal -o /tmp/lead08_iter29.air.s

python3 issue468/artifacts/lead08_reassessment/36_iter28_exact_smallm_pair_validate.py \
  --phase direct --air /tmp/lead08_iter29.air.s \
  --air-map /tmp/lead08_iter29_air.csv /tmp/lead08_iter29_cases.csv
```

Optional diagnostic filters add `DS4_LEAD08_EXACT_SMALLM_SITE=0..6` and
`DS4_LEAD08_EXACT_SMALLM_M=2..8` to the Stage-A layer/format/corpus filters. All filters were unset
for retained evidence.

Retained hashes:

- direct cases CSV: `eea32a3f1aefcdf9ec6a696a8bb2c1fafeb3878554f6ab0178f4eea7176e7330`
- compact AIR map: `ff69b6fbf1eb8924efe8c62b1b56234fbeaaf3d429967b0c53e05ab24bbbd707`
- dumped runtime Metal source: `abeaf6c49e26618f08502db5afb6a5eaa3a40f59ec693391e4fb07504fb208be`
- generated AIR text: `4acae75b531576036d9e2bd7fcc84bdd628732e4ec1e58aac478e29c4ac2afe5`

Generated Metal/AIR remain scratch files. Stage B2 must capture/replay C5 for M=2..4, factor and
sentinel-check a nonredundant batch-low output-B seam, run the predeclared balanced M4 all-layer
timing carrier, and enforce its validity and one-sided 95% upper-delta gates. Until then V15 remains
P0 and no graph integration is authorized.
