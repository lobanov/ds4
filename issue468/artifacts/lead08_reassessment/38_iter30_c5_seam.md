# Lead 08 iteration 30: C5 replay and nonredundant output-B seam

Date: 2026-07-18
Hypothesis: V15 Stage B2a
Verdict: **PASS correctness and seam; timing remains open**

## Preflight corrections

Independent preflight returned `REDESIGN` twice. The protocol's proposed capture command set
`DS4_LEAD08_EXACT_SMALLM_GATE`, which exits after model load and never executes `code_sort_pairs`.
Capture must instead use the normal graph path, with the gate unset. A position probe then showed
that pinning `DS4_DSPARK_VERIFY_K=4` changes the diagnostic-batch starts: historical position 104
had only two rows, while the current trajectory has a full four-row boundary at position 103.
Position 103 is now fixed; absence, duplication, or a non-M4 reproduction is a capture failure.

Preflight split Stage B2 into auditable parts. Iteration 30 is B2a capture/replay/seam correctness
only. Decision-grade fixed-work timing is B2b and remains P0.

## Capture

Seven neutral `Lead08C5*` aliases expose the inputs to all seven dense sites without matching any
existing debug-path predicate. The lowercase `b_` distribution-probe arm at position 103 produced
exactly 43 layers x seven tensors = 301 external F32 files and 57,065,472 bytes. The manifest rejects
extra, missing, wrong-sized, nonfinite, non-`b_`, or wrong-position files and records per-file SHA-256,
model/config/prompt/capture-run hashes, dimensions, tag, and the actual expanded environment/argv.

`Lead08C5AttnNorm` intentionally supplies both Q-a and KV. The other inputs are HC-attention flat,
Q-lora normalized, attention heads, output-A low, HC-FFN flat, and FFN normalized.

## C5 replay

For M=2 and M=3 the replay consumes row prefixes; M=4 consumes all captured rows. Every candidate
uses one exact-small-M dispatch and is compared with M unchanged production M1 calls on the same
immutable input. All 43 layers x seven sites x three M values = 903 cases are word-exact, including
signed zero: 516 Q8 and 387 F16 cases, 129 per site, and 301 per M.

## Output-B seam

The small-batch direct output-A encoder was factored into one shared host helper. The unchanged
combined output API and the new research-only batch-low wrapper call that same helper with the same
pipeline, arguments, grid, scratch, and output-A weight range. No production graph callsite changed.

At M=4 and every layer, three separate command-buffer arms run:

- `E`: combined output-A low plus current ext B.
- `S`: shared batch-low wrapper plus current ext B.
- `C`: shared batch-low wrapper plus exact-small-M B, with four independent M1 B references.

Across 1,409,024 low words, `E.low == S.low == C.low == captured low`. Across 704,512 output words,
`E.out == S.out`; low-only leaves every sentinel output word unchanged; and `C.out` equals the four
M1 references. The retained source-derived census is `E=(combined 1, low 1, ext B 1)`,
`S=(0,1,1)`, and `C=(0,1,0, exact B 1)`, so C never computes ext B before overwriting it.

## Reproduction

```sh
make -j4 ds4-spec-bench
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl
CAPTURE=/tmp/lead08_iter30_c5
mkdir -p "$CAPTURE"

DS4_DSPARK_VERIFY_DIST_PROBE=1 DS4_DSPARK_VERIFY_K=4 \
DS4_DSPARK_VERIFY_BATCHED=0 DS4_DSPARK_ANCHOR_REUSE=1 \
DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 DS4_DSPARK_DRAFT_METAL=1 \
DS4_DSPARK_DRAFT_METAL_STS=1 DS4_METAL_GRAPH_DUMP_PREFIX="$CAPTURE/c5" \
DS4_METAL_GRAPH_DUMP_NAME=Lead08C5HCAttnFlat,Lead08C5AttnNorm,Lead08C5QLoraNorm,Lead08C5Heads,Lead08C5Low,Lead08C5HCFFNFlat,Lead08C5FFNNorm \
DS4_METAL_GRAPH_DUMP_LAYER=all DS4_METAL_GRAPH_DUMP_POS=103 \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config "$ONE" --jsonl-out /tmp/lead08_iter30_capture.jsonl

python3 issue468/artifacts/lead08_reassessment/38_iter30_c5_manifest.py "$CAPTURE" \
  --model "$MODEL" --config "$ONE" \
  --prompt issue468/prompts/exactness_small_corpus/code_sort_pairs.txt \
  --capture-json /tmp/lead08_iter30_capture.jsonl \
  --output /tmp/lead08_iter30_c5_manifest.json

DS4_LEAD08_EXACT_SMALLM_GATE=1 DS4_LEAD08_EXACT_SMALLM_PHASE=c5 \
DS4_LEAD08_EXACT_SMALLM_CAPTURE_DIR="$CAPTURE" \
DS4_LEAD08_EXACT_SMALLM_CSV=/tmp/lead08_iter30_c5_cases.csv \
DS4_LEAD08_EXACT_SMALLM_SEAM_CSV=/tmp/lead08_iter30_seam.csv \
./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" \
  --bulk-config "$ONE" --jsonl-out /tmp/lead08_iter30_unused.jsonl

python3 issue468/artifacts/lead08_reassessment/36_iter28_exact_smallm_pair_validate.py \
  --phase c5 --capture-dir "$CAPTURE" --seam /tmp/lead08_iter30_seam.csv \
  /tmp/lead08_iter30_c5_cases.csv
```

Retained hashes:

- C5 cases: `ff27e0c5bd58aaef0b90ff357af1f501f9583c443416ca5dde105e2fb34780eb`
- seam cases: `706e9a9336e0a32c13711b3f2bf1cca9d79f989c001d55c6315893ff4ca56c8d`
- capture manifest: `4c504768422378cdebdb20f1a5dced0ec2c893e0d28f8d08a196dd820f46c4e7`
- capture run: `fc4c209f2daea5d9d574411aaaa75e6a4dc0e322a8cf5b198dc26702db6c29ec`
- target model: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`

## Decision

B2a passes and authorizes B2b fixed-work timing. This is no speed result and does not admit graph
rollout. B2b is only a dense-carrier admission screen: it cannot establish full-verifier economics
because intervening graph work, cache pressure, scheduling, and possible register spill are absent.
The predeclared candidate-minus-ext one-sided 95% upper bound must still be at most 7.5 ms.

## Audit record

The first independent red team reproduced the forced build, Stage A/B1 regressions, C5 matrix,
seam, AIR map, and capture manifest, but returned `FIX`: the manifest hashed the actual `v4` capture
record while its command record retained template JSONL and dump-prefix paths. The generator now
records the supplied JSONL path and expanded capture prefix; the manifest was regenerated with the
corrected hash above. The seam census is explicitly labeled source-derived rather than runtime
instrumentation.

The repeat audit regenerated the corrected manifest byte-identically, repeated all 430 Stage-A,
10,535 Stage-B1, fourteen AIR, 903 C5, and 43 seam checks, and returned `COMMIT`. It confirmed the
nonredundant C arm from source, distinct-buffer, and sentinel evidence, and retained the no-timing,
no-economics, and no-rollout limits.
