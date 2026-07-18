# Lead 08 iteration 27: V15 exact small-M executable protocol

Date: 2026-07-18  
Hypothesis: V15 protocol preflight  
Verdict: **REDESIGN, then GO as two research-only capability stages**

## Preflight decision

One iteration cannot rigorously implement both kernel formats, exhaust M=2..8 and all seven shapes,
factor the output-B seam, capture production inputs, and produce decision-grade 43-layer statistics.
That scope would make a numerical failure hard to localize and a timing result hard to audit.

V15 therefore has two stages. Both formats are mandatory at every stage; an F16-only or Q8-only
path is not retained or integrated. Stage A is the only next implementation. Stage B remains
unauthorized until Stage A passes and is audited. Neither stage changes graph callsites.
This supersedes artifact 34's monolithic-prototype wording, but preserves its final all-seven,
M=2..8, nonredundant-seam, and 7.5 ms admission gates in Stage B.

## Stage A: cross-quant M=2 capability

Add a self-contained `metal_graph_test_exact_smallm_dense_family` beside the retained M2 fidelity
unit in `ds4.c`, exposed from `ds4-spec-bench` by `DS4_LEAD08_EXACT_SMALLM_GATE=1`. Add one
research-only host operation and one shared M-token structural Metal template, initially for M=2.
The template owns the independent token accumulators/reductions and single logical traversal;
Q8_0 and F16 supply literal-M1 load/step specializations. Two unrelated bespoke kernels behind one
host API do not pass Stage A.

Use one representative of each arithmetic body at every real layer offset:

| Format | Site | Shape `(M,N,K)` | Real model offset |
|---|---|---:|---|
| Q8_0 | attention Q-a | `(2,1024,4096)` | each layer's `attn_q_a->abs_offset` |
| F16 | FFN router | `(2,256,4096)` | each layer's `ffn_gate_inp->abs_offset` |

Reference output is two independent calls to the unchanged production `n_tok=1` primitive, writing
separate row views. Candidate output is one dispatch with one threadgroup per output tile and two
independent copies of the literal M1 accumulator and reduction path.

The Q8 specialization retains M1 `NR0=2`, NSG 4, block loop, lane grouping, per-token
`sumq += qs[i]*yl[i]`, then per-token `sumf += sumq*d`, threadgroup reduction, and store order.
Only raw `qs` and `d` loads are shared; sharing a dequantized `q*d` value would change the M1 DAG.
The F16 specialization retains M1 `NR0=2`, NSG 8,
half4/float4 loop, lane grouping, threadgroup reduction, and store order. In Q8, only the raw
`qs`/`d` load and address recurrence sit outside the token loop; per-token `sumq` and scale multiply
remain inside. In F16, the half4 weight load/conversion sits outside the token loop. Token
accumulators never combine. A source map plus compiled AIR must show one output-tile dispatch, no
token grid dimension, no cloned Q8 raw-load recurrence, and no cloned F16 weight-load loop. This
proves one logical traversal, not physical DRAM traffic;
the unavailable M5 counters cannot prove the latter.

Run all 43 model offsets with immutable deterministic finite inputs. Constructors operate on F32
words and use `j=t*K+i` unless stated otherwise:

- `C0`: word `0x80000000` when `j` is odd, else `0x00000000`.
- `C1`: initialize `+0`; for base-list indices `{0,3,31,32,K-1}`, set `(base+t)%K` to
  `0x3f800000` when `(base_list_position+t)` is even, else `0xbf800000`. A later list entry wins if
  shifted indices collide.
- `C2`: `sign*2^e`, where `e=-12+((i+7*t)%17)` and sign is negative iff `(i+t)` is odd; construct
  word `(((i+t)&1)<<31) | ((e+127)<<23)` rather than using `pow`.
- `C3`: assign site IDs by the Stage-B table order `0..6` (HC attention through router), seed
  `s=0x4681500 ^ (layer<<16) ^ (site<<8) ^ M`; before each element set
  `s=1664525*s+1013904223` modulo `2^32`, compute magnitude
  `float((s>>8)&0xffff)/65536`, reinterpret its word, and replace bit 31 with `s>>31`. The division
  is exact binary; a zero magnitude starts as `+0` before the selected sign bit is applied.
- `C4`: cycle by `(i+3*t)%10` through exact words `{0x00000001,0x80000001,0x00800000,
  0x80800000,0x33800000,0xb3800000,0x3f7fffff,0xbf7fffff,0x3f800001,0xbf800001}`.
  All are finite; NaN and infinity are excluded.

Retain per-case input hash, candidate dispatch record, output bit-difference count, first differing
index/bits, and max absolute difference. Stage A passes only if both formats have one logical
weight traversal and zero differing F32 bits for every case. Any format asymmetry, independent
token traversal, accumulator-topology failure, or persistent mismatch is `STOP`. Commit only the
research API/harness and evidence; do not alter production graph callsites.

Stage A retains `36_iter28_exact_smallm_pair.md`, a machine-readable per-case CSV, a compact AIR
source-map/check summary, and a deterministic validator. Generated Metal source/AIR remain
reproducible scratch outputs rather than committed bulk artifacts.

## Stage B: full family and economic gate

Only after an audited Stage-A pass, extend both specializations to M=2..8. M=6..8 must still use
one candidate token group per output tile rather than the current ext path's multiple token tiles.
Exercise every real layer offset for every M and every exposed shape:

| Format | Site | Shape `(M,N,K)` |
|---|---|---:|
| F16 | HC attention mixer | `(M,24,16384)` |
| Q8_0 | attention Q-a | `(M,1024,4096)` |
| Q8_0 | attention KV | `(M,512,4096)` |
| Q8_0 | attention Q-b | `(M,32768,1024)` |
| Q8_0 | attention output-B | `(M,4096,8192)` |
| F16 | HC FFN mixer | `(M,24,16384)` |
| F16 | FFN router | `(M,256,4096)` |

Reuse C0-C4 and add `C5`, real K4 boundaries captured from the retained `code_sort_pairs` suffix at
position 104. Use row prefixes for M=2/3 and all rows for M=4; C5 does not apply to M=5..8 unless a
separate real wider-batch capture is preflighted. There are six external dense call boundaries;
output-B is internal to combined output. Capture `batch_heads` immediately before that operation
and the resulting `batch_attn_low` after it. Capture binaries stay external/gitignored; retain a
compact manifest, byte counts, SHA-256 hashes, and exact reproduction command.
For every case, candidate output must be bit-identical, including signed zeros, to M independent
production M1 calls consuming the same immutable input.

### Nonredundant output-B seam

Factor the combined attention-output encoder into an internal low-only encoder plus B projection,
and expose a research-only batch-low wrapper. Splitting the API is not mandatory; a suppress-B flag
or equivalent extension is acceptable. The ext arm uses the unchanged combined low+B operation.
The candidate arm uses the identical low-only encoder followed by exact-small-M B. Both timing arms
consume the same captured `heads`; direct B bit tests consume the same captured/replayed `low`. Require low
buffers to be bit-identical and sentinel-check that low-only does not write B.

The timing carrier includes low projection and B in both arms. This prevents the current diagnostic
mistake of computing ext B and overwriting it. GPU time includes both candidate dispatches; retain
host encode-plus-wait wall time separately because Metal GPU timestamps exclude host work.

### Fixed-work timing carrier

Each arm is explicitly M=4 and encodes all seven sites in production order for layers 0..42 in one
command buffer, using real offsets and captured C5 K4 inputs. Outputs are intentionally independent rather than fed
downstream, keeping work and routing fixed. `ds4_gpu_end_commands()` synchronizes with
`waitUntilCompleted`. Primary timing is command-buffer `GPUStartTime/GPUEndTime`; driver
`kernelStartTime/kernelEndTime` is not a shader interval and is not decision evidence.

- `E`: current production K4 ext kernels; output-B uses combined low+B.
- `C`: exact-small-M family; output-B uses identical low-only plus exact B.
- Warmup: four excluded blocks, alternating `E-C-C-E` and `C-E-E-C`.
- Measurement: 40 blocks with the same alternating orders, two observations per arm per block.
- Block delta: mean candidate GPU ms minus mean ext GPU ms within the block.
- Statistic: median of 40 paired block deltas.
- Interval: deterministic 100,000-resample paired bootstrap, seed `0x46815`, one-sided 95% upper
  percentile.

Retain every raw GPU/wall interval, order position, command-buffer status, device/OS/Xcode/Metal
identity, model hash, input hash, and deterministic parsed CSV. All GPU intervals must be positive
and finite. Validity requires all of the following:

- for each arm and block, label its observations by that arm's execution order `A1,A2` and compute
  `100*(A2-A1)/A1` plus `A2-A1` ms. Across 40 blocks, the relative-median two-sided 95% interval
  uses 100,000 paired block resamples with seed `0x46816 + arm_id`, must lie in `[-1%,+1%]`, and
  `abs(median(A2-A1))` must be at most 0.10 ms;
- the difference between median block deltas under `E-C-C-E` and `C-E-E-C` has a two-sided
  stratified two-sample bootstrap interval from 100,000 within-order resamples with seed `0x46818`,
  contained in `[-0.50,+0.50]` ms, and absolute point difference at most 0.25 ms;
- Theil-Sen slope versus block number for each arm mean and the block delta is at most 0.01 ms/block
  in absolute value; and
- every arm has 20 blocks in each order, with no missing/zero interval or command error.

A timing-validity failure is `AMBIGUOUS`; repeat once with 80 otherwise identical measured blocks
and the same numeric thresholds. Persistent ambiguity is `STOP`, not GO. Source/AIR topology,
bitwise correctness, or seam failure is a direct `STOP`, never an ambiguous timing result.

### Outcome tree

- `PASS`: both formats and all direct cases are bit-exact, every candidate uses one logical weight
  traversal, output-B is nonredundant, and the all-43 delta upper bound is at most 7.5 ms.
- `STOP`: any bit/topology/format/seam failure, or an economically valid upper bound
  above 7.5 ms.
- `AMBIGUOUS`: only a measurement-validity failure; repeat once as specified, then stop if it remains.

Passing Stage B admits graph rollout. It does not establish full-path exactness,
`verify_ms(4)<=50.5 ms`, or the project throughput goal.

## Reproduction contract

Stage A and Stage B use the same retained entry point with a phase selector. Stage A:

```sh
make -j4 ds4-spec-bench
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf

DS4_LEAD08_EXACT_SMALLM_GATE=1 \
DS4_LEAD08_EXACT_SMALLM_PHASE=pair \
DS4_METAL_DUMP_SOURCE=/tmp/lead08_exact_smallm.metal \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_exact_smallm.jsonl \
  2>/tmp/lead08_exact_smallm.err

xcrun metal -S -gline-tables-only \
  -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 \
  -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_exact_smallm.metal \
  -o /tmp/lead08_exact_smallm.air.s
```

Before Stage B is authorized, its implementation must make these two commands exact. The capture
command creates external C5 data plus the retained manifest; the full command consumes it:

```sh
CAPTURE=/tmp/lead08_exact_smallm_c5

DS4_LEAD08_EXACT_SMALLM_GATE=1 \
DS4_LEAD08_EXACT_SMALLM_PHASE=capture \
DS4_LEAD08_EXACT_SMALLM_CAPTURE_DIR="$CAPTURE" \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_exact_smallm_capture.jsonl \
  2>/tmp/lead08_exact_smallm_capture.err

DS4_LEAD08_EXACT_SMALLM_GATE=1 \
DS4_LEAD08_EXACT_SMALLM_PHASE=full \
DS4_LEAD08_EXACT_SMALLM_CAPTURE_DIR="$CAPTURE" \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_exact_smallm_full.jsonl \
  2>/tmp/lead08_exact_smallm_full.err
```

Stage A is estimated at one iteration (roughly 6-10 engineering hours). Stage B is a separate
iteration (roughly 8-16 hours). No production path is retained if either format fails.

## Audit record

The first independent red-team returned `FIX`. It validated the M1 topology constants but required:
honest K4-only C5 scope; separate captured `heads` and replayed `low` at output-B; an explicit M=4
timing carrier; bit-exact C0-C4 constructors; Q8 raw-load sharing without changing its per-token
`sumq`/scale order; a shared structural template rather than two bespoke kernels; a hard STOP for
topology failure; and numeric timing-validity thresholds. The first repeat returned `FIX` on ledger
C5 scope, format-specific AIR wording, Stage-B reproduction, validity-bootstrap details, and site
IDs. Those are also corrected above. The final repeat verified current files and returned `COMMIT`.
Its only nonblocking obligation is that the eventual Stage-B reproduction also dump and compile the
expanded M=2..8 Metal source for its already-required AIR gate.
