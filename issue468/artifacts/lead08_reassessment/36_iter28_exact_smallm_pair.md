# Lead 08 iteration 28: exact small-M M=2 capability

Date: 2026-07-18  
Hypothesis: V15 Stage A  
Verdict: **PASS capability; Stage B admitted after audit**

## Result

The research-only cross-quant family passes the iteration-27 Stage-A contract. One host operation
selects Q8_0 or F16, dispatches one M=2 grid with `grid_y=1`, and does not alter a production graph
callsite. Its shared Metal structural template owns the two independent accumulator arrays and the
two unchanged reduction/write calls. Exactly one format specialization is invoked per candidate;
that specialization performs one logical weight traversal and retains the literal M1 arithmetic
boundary for each token.

The decisive matrix has 430 unique cases: 43 real layer offsets x two formats x C0-C4. Q8 Q-a is
`(M,N,K)=(2,1024,4096)` at site 1; the F16 router is `(2,256,4096)` at site 6. Each candidate was
compared by F32 word against two calls to the unchanged production `n_tok=1` primitive in the same
command buffer. Results are 215/215 PASS per format, 86/86 per corpus, zero differing words,
including signed zero, and zero max absolute difference. The deterministic validator independently
reconstructs every input word and the project's explicitly labeled `ds4_64` hash; it rejects
missing, duplicate, mis-shaped, wrong-geometry, or nonexact records.

This is a capability result only. It does not establish an M=4 speed saving, physical DRAM traffic,
the seven-site M=2..8 family, a nonredundant output-B seam, full-path exactness, or the project goal.

## Implementation proof

- Q8 retains M1 `NR0=2`, NSG 4, eight-Q lane grouping, per-token `sumq=+0`, eight ordered
  `sumq += qs[i]*yl[i]` operations, and `sumf += sumq*d`. Only raw `qs` and `d` are loaded outside
  the token loop; `q*d` is not shared.
- F16 retains M1 `NR0=2`, NSG 8, four half4/float4 dot operations, per-token `sumq=+0`, and the M1
  reduction. The four half4 loads/conversions are outside the token loop.
- Host arguments set M=2 and contiguous `K*4` activation rows. Dispatch is exactly
  `(ceil(N/2),1,1)` threadgroups, `(32,NSG,1)` threads, and 256 bytes of threadgroup scratch.
- Runtime source compilation and explicit `metal -S` compilation both pass on Apple M5 Max,
  macOS 26.5.2, Xcode 26.6, Metal 32023.883.

The retained AIR map is `36_iter28_exact_smallm_pair_air.csv`. In the generated AIR, Q8 has one
primary `ib` backedge at line 43102; its raw Q and scale loads are at lines 43048 and 43040 before
the token accumulator loop. F16 has one primary `ib` backedge at line 43307 and one half4 load at
line 43255 before the token dot loop. Neither raw-weight address contains a token term. The generic
F16 tail loop remains in unspecialized AIR, but the host requires `K % 32 == 0`, so it does not run
for admitted shapes. `metal -S` is logical compiled-source evidence; it does not specialize the
function constant or prove physical cache/DRAM transactions. No physical-load claim is made.

## Reproduction

```sh
make -j4 ds4-spec-bench
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf

DS4_LEAD08_EXACT_SMALLM_GATE=1 \
DS4_LEAD08_EXACT_SMALLM_PHASE=pair \
DS4_LEAD08_EXACT_SMALLM_CSV=/tmp/lead08_iter28_cases.csv \
DS4_METAL_DUMP_SOURCE=/tmp/lead08_iter28.metal \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter28_unused.jsonl \
  2>/tmp/lead08_iter28.err

python3 issue468/artifacts/lead08_reassessment/36_iter28_exact_smallm_pair_validate.py \
  /tmp/lead08_iter28_cases.csv

xcrun metal -S -gline-tables-only \
  -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 \
  -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_iter28.metal -o /tmp/lead08_iter28.air.s
```

The early-exit gate intentionally does not create the bench JSONL. Evidence is written to the
required dedicated `DS4_LEAD08_EXACT_SMALLM_CSV` path instead. Optional bring-up filters are
`DS4_LEAD08_EXACT_SMALLM_LAYER`, `DS4_LEAD08_EXACT_SMALLM_FORMAT={q8_0,f16}`, and
`DS4_LEAD08_EXACT_SMALLM_CORPUS={0..4}`; all were unset for the retained decisive run.
Filtered runs report `FILTERED_PASS` for successful bring-up evidence but deliberately exit nonzero;
only an unfiltered 430-case run can report `BIT_EXACT` and exit zero.

Retained hashes:

- cases CSV: `36707a1b7e64270608306b0755d2e89be1dbb7671b49cb6f00569c6ec19219e6`
- dumped runtime Metal source: `b7533fad9fd1043356c3a18183295e65b8d621b97454a9ab26f5da4ceae87319`
- generated AIR text: `7fea4f0cc63f05d48725deccf08e044d6e351f028901946d74d9e2e0d99d2d3f`

Generated source/AIR remain scratch files; the compact case matrix, AIR map, and validator are
retained. Stage B is now the next protocol step, subject to the independent iteration audit. It must
still implement both formats for M=2..8, all seven sites, C5 scope, the nonredundant output-B seam,
and the fixed-work M=4 all-layer `upper_delta <= 7.5 ms` gate before any graph rollout.

## Audit record

The first independent red-team returned `FIX`. It independently validated the 430-case matrix and
AIR topology, but found that a filtered one-case run could report `BIT_EXACT`/exit zero and that an
odd output row count could make the unconditional `NR0=2` second-row load unsafe. It also requested
the predeclared F16 `K % 32` host check; that check had already landed before its snapshot.

The host now rejects odd N and requires `K % 32` for both formats. A filtered success reports
`FILTERED_PASS` and exits nonzero; only exactly 430 unfiltered passing cases can report `BIT_EXACT`.
The repeat audit rebuilt, reran both filtered and full gates, reproduced the retained CSV and
validator, inspected fresh AIR, and returned `COMMIT`. It authorizes Stage B while explicitly
endorsing no speed, physical-DRAM, full-family, or graph-integration claim. A forced CPU-only object
build still fails in older unguarded Metal diagnostics elsewhere in `ds4.c`; the new wrapper is
guarded and this is recorded as a pre-existing nonblocking issue.
