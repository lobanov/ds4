# DSpark Imatrix Follow-up: Weighted Acceptance-Bundle Collector Validation

Date: 2026-06-30.

Purpose: record the first live validation of the corrected acceptance-bundle
collector plus the next collector refinement: draft-position weighting.

## 1. Starting point

After `42_acceptance_bundle_collector_and_kv_state_fix.md`, the DSpark
collector was already corrected to:

- consume an acceptance bundle directory via `--imatrix-dataset <dir>`
- reconstruct frontier anchors from captured `main_hidden`
- preserve the same growing drafter KV trajectory used by the B2 runtime path

The next question was whether the collector should now treat the 5 drafted
positions differently.

A fresh `gpt-5.5 xhigh` review said yes:

- the corrected collector had removed the larger state-mismatch issue
- the remaining main collector gap was that all 5 draft rows were still being
  averaged equally
- any weighting must preserve proper normalization, not just multiply raw sums

## 2. Code change made

Implemented in `ds4.c` and `ds4_cli.c`:

1. Added optional `--imatrix-draft-pos-weights`.
2. Changed the DSpark collector to maintain **5 per-position buckets**.
3. Kept the legacy `.dat` file format unchanged.
4. Merged buckets at save time using weighted normalization:

```text
merged = sum_b(weight[b] * sum2[b]) / sum_b(weight[b] * count[b])
```

Generic non-DSpark imatrix collection still uses one bucket, so its behavior is
unchanged.

Default DSpark weights are now:

```text
[1.0, 0.75, 0.5, 0.33, 0.2]
```

They remain CLI-overridable for A/B collection runs.

## 3. Live Metal validation

### 3.1 Real minimal acceptance bundle was generated

Built a live bundle in:

```text
/tmp/dspark_bundle_live
```

using:

- `target_topk.json`
- `target_greedy.json`
- captured `hc_dspark_main_hc-{40,41,42}_pos*.bin`

Observed:

- `prompt_tokens = 14`
- enough captures for positions `14..23`

### 3.2 Weighted acceptance-bundle collector ran successfully

Command:

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  --imatrix-dataset /tmp/dspark_bundle_live \
  --imatrix-out /tmp/dspark_bundle_live/out.weighted.imatrix.dat \
  --imatrix-max-tokens 3 \
  --ctx 4096
```

Observed:

- acceptance-bundle mode detected and used
- DSpark draft-position weights reported as
  `[1.000 0.750 0.500 0.330 0.200]`
- collector completed successfully
- wrote:
  `/tmp/dspark_bundle_live/out.weighted.imatrix.dat`
- `3` anchors
- `270` routed expert observations

Position diagnostics were emitted:

```text
pos0 merge_weight=1.000 tokens=9
pos1 merge_weight=0.750 tokens=9
pos2 merge_weight=0.500 tokens=9
pos3 merge_weight=0.330 tokens=9
pos4 merge_weight=0.200 tokens=9
```

### 3.3 Weighted vs uniform collection produces different artifacts

Uniform A/B run:

```sh
./ds4 --metal ... \
  --imatrix-draft-pos-weights 1,1,1,1,1 \
  --imatrix-out /tmp/dspark_bundle_live/out.uniform.imatrix.dat
```

Observed SHA-1:

```text
706599ce89e7d2df1815a7aea8763b1643de6849  out.weighted.imatrix.dat
a417fb5acaae6824024cd23ff60a842e5678c3a0  out.uniform.imatrix.dat
```

This proves the default weighted collector changes the resulting DSpark imatrix
relative to uniform weighting on the same frontier bundle.

## 4. End-to-end artifact path validation

The weighted imatrix was then used to build a pilot drafter GGUF:

```sh
gguf-tools/deepseek4-quantize \
  --hf /Users/lobanov/Projects/ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /tmp/dspark_bundle_live/dspark_weighted.gguf \
  --overwrite \
  --imatrix /tmp/dspark_bundle_live/out.weighted.imatrix.dat
```

Observed:

- imatrix loaded successfully
- output written:
  `/tmp/dspark_bundle_live/dspark_weighted.gguf`
- `81` tensors
- `type_changes: 0`

The pilot GGUF also loaded and ran as a DSpark drafter:

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /tmp/dspark_bundle_live/dspark_weighted.gguf \
  -p 'Briefly say hello.' -n 8 --temp 0
```

Observed:

- weighted pilot GGUF loaded successfully
- generation completed successfully
- no tensor-layout or runtime-format errors

## 5. What this proves

The DSpark acceptance-targeted artifact path is now live on Metal for a real
bundle:

```text
acceptance bundle -> corrected persistent-KV collector
                 -> weighted DSpark imatrix
                 -> Q4_K DSpark GGUF
                 -> ds4 runtime load
```

## 6. What is still missing

This is still only a **minimal-bundle validation**, not the assignment result.

Still required:

1. collect weighted imatrix data from real acceptance sweep bundles over the
   intended context set
2. build the real candidate GGUF from that larger weighted collection
3. measure acceptance uplift across:
   `8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k`
4. check whether average uplift reaches the `>= +5%` assignment gate

## 7. Next step

The next practical move should be:

- generate at least one real sweep bundle under `issue468/baseline/`
- collect a larger weighted acceptance imatrix from it
- build a candidate GGUF
- run the existing acceptance sweep harness baseline vs candidate

At this point the remaining work is primarily measurement scale-up, not missing
collector plumbing.
