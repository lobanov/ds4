# CUDA Imatrix Collection Probe Still Blocked in Practice

Date: 2026-07-01

## Purpose

Record the first direct post-`ecb8773` probe of DSpark imatrix collection on
DGX/CUDA, and tighten the execution plan for `59` based on what actually
happens at runtime.

This note follows:

- `issue468/62_split_execution_plan_after_metal_only_imatrix_blocker.md`
- `issue468/63_independent_review_after_oracle_pivot.md`
- `issue468/64_stronger_recoverable_gap_overlay_modes.md`

## What changed before this probe

Commit `ecb8773` removed the explicit Metal-only gate:

- `ds4_cli.c` no longer forces `--imatrix-out` to Metal
- `ds4.c` now accepts any graph backend for imatrix collection

So the obvious next question was:

- does DSpark acceptance-bundle imatrix collection actually complete on CUDA?

## Probe shape

DGX host:

- repo: `~/ds4`
- backend: `cuda`
- target model:
  - `~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- drafter:
  - `/tmp/dspark_baseline_full.gguf`
- dataset:
  - `/tmp/dspark_sweep8/.recoverablegap1ctx.overlay`
  - currently only `ctx_08192` staged

Minimal direct collector probe:

```sh
timeout 600 ./ds4 \
  --backend cuda \
  -m ~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /tmp/dspark_baseline_full.gguf \
  --imatrix-dataset /tmp/dspark_sweep8/.recoverablegap1ctx.overlay \
  --imatrix-out /tmp/dspark_sweep8/recoverablegap1ctx_probe1.imatrix.dat \
  --imatrix-draft-pos-weights 1,0.75,0.5,0.33,0.2 \
  --imatrix-max-tokens 1 \
  --ctx 4096
```

This is stronger evidence than the earlier full wrapped run because it reduces
the question to:

- one staged context
- one anchor step
- direct collector invocation

## Observed behavior

Startup succeeds:

- CUDA backend initializes
- model tensors load into device cache
- DSpark graph diagnostics initialize
- context buffers allocate

But collection does **not** complete in reasonable time even for one anchor.

Observed after about `90` seconds:

- no `recoverablegap1ctx_probe1.imatrix.dat` produced
- process remained in running state
- CPU remained high (`~92%`)
- GPU utilization stayed at `0%`
- only persistent dataset file visible in `lsof` was:
  - `ctx_08192/target_greedy.json`

Observed GPU memory reservation:

- about `109950 MiB`

So the current failure mode is no longer:

- immediate CLI/backend rejection

It is now:

- practical non-completion / host-side stall on the CUDA collector path

## Interpretation

This means `ecb8773` fixed the explicit backend gate but did **not** establish
that the DSpark acceptance-bundle collector is operational on CUDA.

Current evidence supports the narrower statement:

- CUDA is accepted as a collector backend by the CLI and engine
- but the live acceptance-bundle imatrix collection path is still not usable on
  DGX for the oracle-only branch as currently implemented

## Operational consequence for `59`

Until this runtime behavior changes, the branch should again treat split
execution as the practical plan:

1. build recoverable-gap overlays anywhere
2. collect imatrix on a working Metal path
3. quantize and reprobe on DGX/CUDA

This is still aligned with `59` because the research priority remains:

- Step 1. recoverable-gap dataset
- Step 2. routed `Q4_K` re-quantization first

The only revision is execution shape, not research ranking.

## Immediate next step

Do **not** spend more DGX time waiting on CUDA imatrix collection under the
current collector path.

Proceed instead by:

1. keeping the stronger recoverable-gap overlays from `64`
2. obtaining the corresponding imatrix from a working Metal host
3. using `--imatrix-in` on DGX for the first soft-vs-boosted routed-`Q4_K`
   comparison
