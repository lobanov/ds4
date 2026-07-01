# FP8 Reference Harness Revival Status on DGX

Date: 2026-07-01

## Purpose

Record the first concrete execution pass on the `issue468/52_fp8_headroom_plan.md`
branch using the DGX, and state clearly what was achieved vs what is still
blocked.

This note exists because the new goal in
`issue468/53_frozen_runtime_quantization_goal.md` depends on keeping three
ceilings distinct:

1. live Metal `Q4_K` baseline
2. `Q4_K` F32-oracle ceiling
3. source-side FP8 ceiling

The work below was the first attempt to re-establish item 3 on the DGX without
running the heavy path on the local machine.

## Scope of this pass

Attempted only the first part of the plan:

- revive the source-side reference harness on one real capture bundle
- use the older baseline capture first
- prove that the official/source path can at least execute a real two-step
  prefill+decode scenario on the DGX

This pass did **not** yet try to score the 4-context sweep root.

## What changed in the branch

### 1. New goal note is now committed

Committed:

- `issue468/53_frozen_runtime_quantization_goal.md`

This is the current governing goal for the branch.

### 2. Reference harness now supports real two-step validation inputs

Updated:

- `issue468/ref/dspark_ref_harness.py`

The harness no longer assumes validation is a single `main_hidden + input_ids`
pair reused for both prefill and decode.

It now supports a preferred NPZ format with:

- `main_hidden_prefill`
- `input_ids_prefill`
- `main_hidden_decode`
- `input_ids_decode`

Legacy single-step NPZs still load, but the real intended source-ceiling path
is now explicit.

### 3. Added a capture-bundle adapter

Added:

- `issue468/ref/build_capture_case_npz.py`

Purpose:

- read a capture bundle containing
  `hc_dspark_main_hc-{40,41,42}_pos*.bin`
- read `greedy25_tokens.json` or `target_greedy.json`
- build the two-step NPZ expected by the updated reference harness

This removes the old hand-built / undocumented validation input path.

### 4. Extended the TVM/tilelang shim

Updated:

- `issue468/ref/_tvm_ffi_shim.py`

Purpose:

- keep the original `__dict__` registration workaround
- add an attempted guard for the `derived_object` wrapper issue seen later in
  tilelang/TVM Python passes on this DGX stack

## DGX execution performed

All heavy execution in this note was done remotely through:

- `ssh dgx-direct`

### DGX environment findings

Confirmed present on the DGX:

- `~/ds4/hf-dspark`
- `~/ds4/ref-ckpt`
- `~/dref-venv`
- baseline capture bundle:
  `~/ds4/issue468/baseline/dspark_capture`

Not present on the DGX in the previously assumed location:

- `/tmp/dspark_sweep8/ctx_08192`

So the first revival pass used the older baseline capture bundle rather than
the newer 4-context sweep root.

### Baseline validation case built successfully

Built on DGX:

- `/tmp/dspark_ref_case_152_153.npz`

using:

```sh
source ~/dref-venv/bin/activate
cd ~/ds4
python issue468/ref/build_capture_case_npz.py \
  --capture-dir issue468/baseline/dspark_capture \
  --prefill-pos 152 \
  --decode-pos 153 \
  --out /tmp/dspark_ref_case_152_153.npz
```

Observed metadata:

- `pos0=152`
- `prefill_tok=2581`
- `decode_tok=1309`

That proves the new adapter works on a real capture bundle.

## Failure chain observed on the official/source path

### 1. First failure: missing dependency

Initial harness run failed immediately because `~/dref-venv` was missing:

- `safetensors`

This was an environment issue, not a model issue.

Installed on DGX:

```sh
source ~/dref-venv/bin/activate
pip install safetensors
```

After that, the harness progressed into actual checkpoint load and JIT.

### 2. Second failure: tilelang / TVM Python wrapper stack

Once the harness reached `model.forward_spec`, it did **not** fail on DSpark
shape or checkpoint mismatch first.

Instead, it failed inside the tilelang/TVM Python wrapper machinery during JIT
lowering.

Observed failure sequence:

1. `__dict__` registration bug
   - original known shim target
2. `NestedLoopChecker`
   - `_NestedLoopCheckVisitor` missing `_inst`
3. `FragmentLoopChecker`
   - `_FragmentLoopCheckVisitor` missing `_inst`
4. `DecoupleTypeCastMutator`
   - `DecoupleTypeCastMutator` missing `_inst`

Representative error family:

- `AttributeError: '<wrapped class>' object has no attribute '_inst'`

Key point:

- the official/source stack did **not** reach a successful FP8 forward on this
  DGX Python 3.12 environment
- the blocker is a broad tilelang/TVM Python interop problem, not one isolated
  semantic-check pass

### 3. Why this matters

The branch now has concrete evidence that:

- the current DGX reference environment is insufficient for straightforward
  source-side FP8 ceiling measurement through the official tilelang path
- reviving that path will likely require either:
  - a cleaner environment rebuild on a compatible Python / package stack
  - or an alternative source-side execution path that bypasses the brittle
    tilelang JIT wrapper layer

## What this pass achieved

This pass **did** achieve the following:

1. the frozen-runtime quantization goal note is committed
2. the reference harness input format is corrected for real two-step DSpark use
3. the capture-bundle-to-NPZ adapter exists and works on a real capture
4. the DGX environment gap is narrowed from "unknown" to a concrete failure
   chain inside tilelang/TVM Python wrappers

## What this pass did not achieve

This pass did **not** yet achieve:

1. a successful source-side FP8 forward on DGX
2. a scored source-FP8 ceiling on even one real capture bundle
3. the 4-context ceiling comparison planned in `52_fp8_headroom_plan.md`

## Interpretation

The original "revive the official reference harness first" avenue has not yet
produced the intended FP8 measurement.

However, it is no longer blocked by vague setup uncertainty. It is now blocked
by a specific and repeated class of failures:

- tilelang/TVM Python wrapper incompatibility on the current DGX environment

That is a materially stronger result than the prior state, because it separates:

- missing plumbing and input formatting, which are now addressed
- from the remaining environment/runtime incompatibility, which is now the
  dominant obstacle

## Immediate next options

The next move should be one of:

1. build a clean source-reference environment on the DGX that avoids the
   current tilelang/TVM wrapper failures
2. bypass the tilelang path and recover the source-side FP8 ceiling through a
   simpler HF/torch execution path
3. if source-side FP8 remains blocked, move to the next in-scope avenue from
   `53_frozen_runtime_quantization_goal.md` and treat this tilelang route as an
   exhausted branch pending a cleaner environment

Per Step 6 of the new goal note, an independent `gpt-5.5 xhigh` review should
be used here to help choose between those next avenues rather than assuming the
current interpretation is complete.
