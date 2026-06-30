# DSpark Imatrix Follow-up: Sweep-Root Collector + 2-Context Weighted Signal

Date: 2026-06-30.

Purpose: record the next scale-up step after single-bundle weighting validation:

1. teach the DSpark collector to consume a **sweep root** containing multiple
   `ctx_*` acceptance bundles
2. build a first multi-bundle weighted candidate
3. measure whether it shows any positive acceptance signal

## 1. Problem addressed

After `43_weighted_acceptance_bundle_imatrix_validation.md`, the collector and
weighting logic were only proven on one bundle at a time.

That was still insufficient for the assignment gate, which is based on average
acceptance across:

- `8k`
- `16k`
- `24k`
- `32k`
- `40k`
- `48k`
- `56k`
- `64k`

So the next missing primitive was **cross-bundle aggregation** without inventing
a lossy post-hoc `.dat` merge format.

## 2. Code change made

Implemented in `ds4.c`:

- directory mode now distinguishes:
  - a **single acceptance bundle**
  - a **sweep root** containing multiple `ctx_*` bundle directories
- sweep-root mode:
  - scans immediate child dirs
  - selects only dirs that contain:
    - `target_topk.json`
    - `target_greedy.json`
  - sorts them by path
  - reuses one in-memory `ds4_imatrix_collector`
  - resets session/KV per bundle
  - accumulates all selected bundles into one final weighted DSpark imatrix

This preserves the existing legacy imatrix output path and avoids lossy
merge-after-save tooling.

## 3. Live sweep-root validation

### 3.1 Real root created

Used the existing sweep harness to create a small real root:

```text
/tmp/dspark_sweep2
```

with:

- `ctx_08192`
- `ctx_16384`

Each directory contained:

- `prompt_rendered.txt`
- `target_topk.json`
- `target_greedy.json`
- captured `hc_dspark_main_hc-*`

### 3.2 Sweep-root collector ran successfully

Command:

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  --imatrix-dataset /tmp/dspark_sweep2 \
  --imatrix-out /tmp/dspark_sweep2/root_weighted.imatrix.dat \
  --imatrix-max-tokens 3 \
  --ctx 4096
```

Observed:

- collector consumed both:
  - `/tmp/dspark_sweep2/ctx_08192`
  - `/tmp/dspark_sweep2/ctx_16384`
- wrote:
  `/tmp/dspark_sweep2/root_weighted.imatrix.dat`
- `2` bundles
- `6` anchors
- `540` routed expert observations

## 4. End-to-end pilot build from the sweep root

Built:

```text
/tmp/dspark_sweep2/dspark_root_weighted.gguf
```

from:

```text
/tmp/dspark_sweep2/root_weighted.imatrix.dat
```

Observed:

- imatrix loaded successfully
- output GGUF written successfully
- `81` tensors
- `type_changes: 0`

## 5. First measured acceptance signal

Reused the already-captured `ctx_*` bundles directly for a faster direct probe
comparison:

- baseline: `../ds4/gguf/dspark.gguf`
- candidate: `/tmp/dspark_sweep2/dspark_root_weighted.gguf`

Used:

- existing captured `target_topk.json`
- existing captured `target_greedy.json`
- fresh DSpark probe q-dumps
- `issue468/baseline/dspark_capture/measure_metal_b2.py`

### 5.1 `8192`

Observed:

- baseline accepted: `3.56875`
- candidate accepted: `3.61875`
- delta: **`+1.40%`**

- baseline committed: `4.1625`
- candidate committed: `4.25625`
- delta: **`+2.25%`**

### 5.2 `16384`

Observed:

- baseline accepted: `3.39375`
- candidate accepted: `3.50000`
- delta: **`+3.13%`**

- baseline committed: `3.9625`
- candidate committed: `4.0500`
- delta: **`+2.21%`**

### 5.3 2-context average

Simple mean of the two measured acceptance deltas:

```text
(1.401% + 3.131%) / 2 = 2.266%
```

Interpretation:

- this is **positive signal**
- it is stronger than the earlier negative / flat generic-imatrix pilots
- but it is still **well below the assignment gate**
- and it was measured only on:
  - `2` contexts
  - a very small collection budget
  - `16` MC trials per context for the direct reprobe

So this is encouraging, but not decisive.

## 6. What this means

The weighted acceptance-targeted path now has all required plumbing for a real
assignment attempt:

```text
multi-context sweep root
-> weighted DSpark frontier collector
-> one aggregated DSpark imatrix
-> one Q4_K candidate GGUF
-> direct acceptance comparison
```

This is the first measured evidence in the current branch that the
acceptance-targeted weighted path may be directionally better than the old
generic path.

## 7. Next step

The next real experiment should be:

1. build a larger sweep root covering all target contexts
2. collect a larger weighted DSpark imatrix from that full root
3. quantize the real evaluation candidate
4. run the full `8k..64k` acceptance sweep
5. check whether the average acceptance uplift reaches `>= +5%`

At this point the remaining uncertainty is mostly experimental scale, not
missing implementation seams.
