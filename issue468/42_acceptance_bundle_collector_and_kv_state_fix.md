# DSpark Imatrix Follow-up: Acceptance Bundle Collector + KV State Fix

Date: 2026-06-30.

Purpose: record the next collector correction after re-reading
`34_dspark_imatrix_q4k_assignment.md`, checking the live code, and consulting a
fresh `gpt-5.5` review pass.

## 1. New finding

The next highest-leverage issue was **not** draft-position weighting first.

The existing DSpark imatrix collector was sampling the drafter from a
single-anchor-reset KV state on every anchor:

- capture `main_hidden`
- run DSpark input stage
- zero + reseed drafter KV
- collect one block

That does not match the actual B2 speculative path, which uses a
**persistent, growing DSpark KV window** across cycles.

So before any weighting change, the collector needed to move onto the same KV
trajectory that the production drafter actually uses.

## 2. Code change made

Implemented in `ds4.c`:

1. Added a DSpark KV reset helper used only at sequence start.
2. Changed DSpark imatrix collection to reuse the persistent drafter KV window
   across successive anchors instead of re-seeding from scratch for every
   sample.
3. Added a DSpark **acceptance-bundle collector mode**:
   - if `--dspark` is loaded and `--imatrix-dataset PATH` points to a
     directory, treat it as an acceptance bundle
   - load:
     - `target_greedy.json`
     - `target_topk.json` (for `prompt_tokens`)
     - `hc_dspark_main_hc-{40,41,42}_pos*.bin`
   - reconstruct the frontier state directly from captured target hidden states
   - run the DSpark collector on those frontier anchors with the same growing KV
     behavior as the B2 path

This keeps the existing text-dataset collector working, but makes its DSpark
sampling more faithful too.

## 3. Why this is the right next correction

The acceptance assignment is about **measured acceptance uplift**, not generic
activation preservation.

If the collector samples a drafter state distribution that the runtime never
uses, then any later weighting, corpus choice, or anchor-budget work is being
optimized on the wrong surface.

So the order should be:

1. fix collector state mismatch
2. collect from acceptance-frontier bundles
3. then add position-aware or hard-case-aware weighting if needed

## 4. Validation status

Validated now:

- `make ds4.o ds4_cli.o` passes after the collector change

Not validated in this Codex environment:

- live Metal runtime execution

Blocker:

- local execution environment reports `Metal device not available`

So the bundle-mode runtime smoke still needs to be run on the real Metal host.

## 5. Expected invocation on the real machine

Example smoke command against one sweep bundle:

```sh
./ds4 --metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  --imatrix-dataset issue468/baseline/dspark_imatrix_sweep/ctx_08192 \
  --imatrix-out /tmp/dspark-acceptance-ctx8k.imatrix.dat \
  --imatrix-max-tokens 8 \
  --ctx 4096
```

Expected behavior:

- detect directory input
- load acceptance bundle artifacts
- collect DSpark `mtp.*` imatrix entries from frontier anchors
- preserve growing drafter KV across successive anchors

## 6. Next step

If the real-machine smoke works, the next implementation step should be:

- add explicit position-aware buckets / merge weights on top of this corrected
  collector

That keeps weighting work downstream of the now-fixed frontier-state mismatch.
