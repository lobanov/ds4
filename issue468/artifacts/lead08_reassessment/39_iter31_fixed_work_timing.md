# Lead 08 iteration 31: exact small-M fixed-work timing

Date: 2026-07-18
Hypothesis: V15 Stage B2b
Verdict: **STOP: persistent timing-validity ambiguity**

## Preflight correction

Independent preflight returned `REDESIGN, then proceed`. The existing derived-duration API did not
retain raw Metal timestamps/status, and the within-arm A2-minus-A1 control pooled both schedules,
allowing opposite order effects to cancel. Iteration 31 therefore exposes raw `GPUStartTime`,
`GPUEndTime`, and command-buffer status and evaluates four stability strata: E/C separately within
`ECCE` and `CEEC`. Each stratum keeps the predeclared `[-1%,+1%]` relative-median interval and
`abs(median absolute delta) <= 0.10 ms` limits.

## Carrier

The gate uploads the fixed position-103 C5 inputs once and allocates disjoint E/C outputs for every
layer and site. Layer-major work follows production order. E runs six standalone ext projections
plus unchanged combined output-A low/ext-B. C runs six exact-small-M projections plus the shared
low-only operation and exact B; it never calls combined output or ext B. Both arms issue eight
source-derived compute dispatches/layer, 344 per observation.

The seven exposed matrices cover 77.875 MiB/layer and common output-A adds 34 MiB/layer: 5,044,305,920
unique mapped weight bytes across 43 layers. This is a logical mapped-weight census, not physical
DRAM traffic. Two excluded prime observations populate pipelines/model views, followed by four
excluded alternating warmup blocks. NaN sentinels, all-output finiteness/overwrite checks, complete
per-arm hashes, and captured-low equality pass after warmup, measurement, and untimed replay. No
read, upload, dump, profile hook, or capture I/O occurs in measured command buffers.
The stable all-output hashes are E `45511b7150b45b22` and C `260cc33618b51f1b` in both runs.

Every observation owns one command buffer. Retained rows contain raw GPU start/end, derived GPU and
supporting kernel intervals, encode/commit-wait/wall intervals, raw status, schedule/slot/occurrence,
and the fixed-work census. All 180 rows in the first run and 340 in the repeat report Metal status 4
(`Completed`), positive finite intervals, 344 dispatches, and `PASS` execution.

## Results

The 40-block run is economically favorable but invalid:

- median C-minus-E block delta: `-3.459979 ms`
- one-sided 95% upper bound: `-3.442208 ms`
- validity: `AMBIGUOUS`; failed stability strata: `ECCE/E`, `ECCE/C`, `CEEC/E`

The one allowed 80-block repeat remains economically favorable but invalid:

- median C-minus-E block delta: `-3.475594 ms`
- one-sided 95% upper bound: `-3.455875 ms`
- order-effect interval: `[-0.037698,+0.035479] ms`; pass
- Theil-Sen slopes E/C/delta: `+0.000162`, `+0.000033`, `-0.000069 ms/block`; pass
- validity: `AMBIGUOUS`; all four arm/schedule stability strata fail

The repeat intervals cross a relative bound by 0.082 to 0.211 percentage points; absolute medians
all pass. The favorable primary signal does not authorize waiving the predeclared validity gate.
Persistent ambiguity is `STOP`.

## Environment

- device: Apple M5 Max, 128 GiB
- macOS: 26.5.2 (25F84), Darwin 25.5.0 arm64
- Xcode: 26.6 (17F113)
- Metal: Apple metal 32023.883, MobileAsset MetalToolchain 17.6.109.0
- base commit: `7d42ed6006335914b8619cab753722537b917a7c`
- model SHA-256: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- C5 manifest SHA-256: `4c504768422378cdebdb20f1a5dced0ec2c893e0d28f8d08a196dd820f46c4e7`

Retained hashes:

- timing metadata: `ed712f4de42e8b7197ab5c66dd4d02441ba86175d4b548e83a960456114ca51f`
- 40-block run log: `e7ae724ed18813f9d0b2f97e1b94ed0da44ba369ff29ba88fdc1fb365c2e796b`
- 80-block run log: `8e464df6a00e577d53a25f0f44ef3d7882b189084654a7802669278df87df13d`
- 40-block raw: `be68b92e20637c8208f39f0f937fd3827b71a7ee33f02ac9eee59dd5f4598643`
- 40-block summary: `8e7bbfe4277646f94e62390a561a5729cd2341c2ebabe9e29f4e6b7733a4c0c9`
- 80-block raw: `1c5c48b5889b5899bde147de93e25f204c438404c065bdc64c6d071b90cce96f`
- 80-block summary: `a7fe6507f5240b7a55e017688c31d42536b2adbc4efff27e5f729703ae0962b9`
- validator: `7b411b30cdfccdddc17b6214ed4fbbfbde174b2d0f5b99c88fc7497b65c93338`

## Reproduction

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
CAPTURE=/tmp/lead08_iter30_c5_v4
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl

DS4_LEAD08_EXACT_SMALLM_GATE=1 DS4_LEAD08_EXACT_SMALLM_PHASE=timing \
DS4_LEAD08_EXACT_SMALLM_CAPTURE_DIR="$CAPTURE" \
DS4_LEAD08_EXACT_SMALLM_TIMING_CSV=/tmp/lead08_iter31_timing_raw.csv \
DS4_LEAD08_EXACT_SMALLM_TIMING_METADATA=/tmp/lead08_iter31_timing_metadata.csv \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config "$ONE" --jsonl-out /tmp/lead08_iter31_unused.jsonl

python3 issue468/artifacts/lead08_reassessment/39_iter31_timing_validate.py \
  /tmp/lead08_iter31_timing_raw.csv /tmp/lead08_iter31_timing_summary.json

DS4_LEAD08_EXACT_SMALLM_GATE=1 DS4_LEAD08_EXACT_SMALLM_PHASE=timing \
DS4_LEAD08_EXACT_SMALLM_CAPTURE_DIR="$CAPTURE" \
DS4_LEAD08_EXACT_SMALLM_TIMING_CSV=/tmp/lead08_iter31_timing_raw_80.csv \
DS4_LEAD08_EXACT_SMALLM_TIMING_METADATA=/tmp/lead08_iter31_timing_metadata_80.csv \
DS4_LEAD08_EXACT_SMALLM_TIMING_BLOCKS=80 \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config "$ONE" --jsonl-out /tmp/lead08_iter31_unused80.jsonl

python3 issue468/artifacts/lead08_reassessment/39_iter31_timing_validate.py \
  /tmp/lead08_iter31_timing_raw_80.csv /tmp/lead08_iter31_timing_summary_80.json \
  --prior-summary /tmp/lead08_iter31_timing_summary.json
```

## Decision

V15 Stage B2b closes `STOP`; no further timing repeat and no graph integration are authorized under
this protocol. The negative nominal delta is useful engineering evidence that shared exact-small-M
dense work may be faster in isolation, but it is not a valid admission result and does not establish
verifier timing, full-path economics, bounded-divergence behavior, or project throughput.

## Audit record

The first independent red team reproduced the build, work census, metadata, prior correctness/AIR
gates, and both timing summaries, but returned `FIX`: C5 input finiteness still used `isfinite`
under `-ffast-math`. The loader now rejects IEEE exponent-all-ones words directly. An independent
scan finds zero nonfinite words across all 301 retained capture files, the live C5/seam gate still
passes 903/903 and 43/43 with byte-identical evidence, and an injected `+Inf` exits nonzero before
case execution.

The required repeat red team returned `COMMIT`. It independently confirmed completed raw Metal
timestamps/status, eight-dispatch layer parity and disjoint outputs, 301 metadata rows and the
5,044,305,920-byte logical census, all prior capability/seam gates, byte-identical 40/80 summaries,
the persistent-ambiguity `STOP`, and the absence of production graph changes or broader speed,
traffic, divergence, throughput, or rollout claims.
