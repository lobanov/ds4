# Phase 4 (preflight) — CUDA reference harness on DGX Spark: findings

Date: 2026-06-28. Node: DGX Spark `dgx-direct` (NVIDIA GB10 Grace Blackwell,
sm_120, aarch64, 128 GB). Purpose: stand up the official `inference/model.py.
forward_spec` as the gold-standard independent reference for the Phase-4 drafter
port (user-chosen validation strategy: CUDA box).

## What works on the DGX

- **Reachable + roomy:** ssh `dgx-direct`; 117 GB free; 3.4 TB disk. No critical
  workloads (user cleared for use).
- **Software installs:** `uv` + Python 3.11.15 venv; `tilelang==0.1.8` + `torch
  2.12.1` (CUDA 13) + `triton 3.7.1` + `ninja` install from PyPI.
- **Weights present + convertible:** drafter shards 46-48 + shared `embed`
  (shard 1) + `head` (shard 45) downloaded (~13 GB). `inference/convert.py` ran
  cleanly → `model0-mp1.safetensors` (FP4 view-cast to `float4_e2m1fn_x2`).
- **Model instantiates:** drafter-only `Transformer(n_layers=0,
  compress_ratios=(0,0,0))` builds (3 mtp stages, shared embed/head bound,
  correct shapes), and loads the converted checkpoint (strict=False).
- **Shipped `generate.py` confirmed:** it never calls `forward_spec` — only plain
  autoregressive `forward`. So there is no turnkey speculative reference even on
  a working CUDA box; the harness had to be written from scratch (done:
  `issue468/ref/dspark_ref_harness.py`).

## The blocker: tilelang 0.1.8 cannot JIT kernels on Blackwell (sm_120)

The official DSpark kernels (`fp4_gemm`, `fp8_gemm`, `sparse_attn`,
`hc_split_sinkhorn`, `act_quant`, `hc_split_sinkhorn`) are `@tilelang.jit`
CUDA-DSL definitions. On first invocation they must be compiled to PTX. This
codegen fails on the GB10, in a **cascade** of tvm_ffi / bundled-TVM-runtime
version-skew bugs (each patch exposes the next):

| # | failure | fix tried | result |
|---|---|---|---|
| 1 | `tvm_ffi` import: `setattr '__dict__' on type` (DictAttrs field) | downgrade `apache-tvm-ffi` 0.1.12 → 0.1.2 | cleared |
| 2 | harness: `n_layers=0` mis-indexed `compress_ratios` (mtp.2 → ratio 4, made spurious Compressor/Indexer) | pass `compress_ratios=(0,0,0)` | cleared |
| 3 | tilelang JIT needs `ninja` | `uv pip install ninja` | cleared |
| 4 | `Allocate` check: `tir.const(True)` not bool-dtype | patch `tir.const(True, dtype="bool")` | cleared |
| 5 | `_NestedLoopCheckVisitor` `__setattr__('_inst')` refused by `TVMDerivedObject` | — | **blocked** |

#5 is the same class of skew as #1/#4 (tilelang's bundled `3rdparty/tvm` runtime
uses attribute patterns the installed tvm_ffi forbids). Patching #5 would
reveal #6, etc. — unbounded whack-a-mole on a fundamental tilelang-0.1.8-bundled-
TVM vs installed-runtime incompatibility on brand-new (sm_120) hardware.

## Version space exhausted

- `tilelang==0.1.8` (DSpark's pinned version): blocked above.
- `tilelang==0.1.10` (latest 0.1.x on aarch64): worse — core-dumps on
  `TypeAttr __ffi_repr__ already registered` during import, before any kernel.
- `tilelang>=0.2.0` on aarch64 PyPI: **does not exist** (`uv` reports
  "unsatisfiable" for 0.2.0/0.2.5/0.3.0/0.4.0).

So there is no tilelang version on aarch64 that both (a) matches the
`model.py` API and (b) can compile on the GB10. The official-kernel CUDA
reference is unreachable on available hardware without patching tilelang
internals (deep TVM IR work, unbounded, uncertain) or building tilelang from
source against a matched TVM (multi-hour C++/CUDA build, no Blackwell guarantee).

## Implication for the Phase-4 gate

The phase4-refcheck verification contract is "drafter draft tokens match a
Python reference rollout of `inference/model.py.forward_spec`." The chosen CUDA
backend for that reference is blocked. Two viable paths forward:

- **(A) numpy/CPU reference** (originally option 0): re-implement
  `forward_spec` in numpy using the converter's already-dequantized F32 drafter
  weights (from `dspark.gguf`), fed `main_hidden` captured from ds4's validated
  target. Runs locally on the M5 Max; bounded effort; certain to run. Less
  independent (I read the same `model.py` for both the port and the reference),
  BUT the block internals (MLA attn, MoE FFN, HC) reuse ds4's already-end-to-end-
  validated target kernels, so a shared misreading is only possible in the small
  DSpark-unique surface (main_proj/main_norm, forward_embed/noise-block, Markov
  head, hc_head, forward_head) — and even there, control flow, shapes, dequant
  values, and the Markov-head rank-256 math are still cross-checked.
- **(B) different CUDA GPU** (older arch where tilelang 0.1.8 is known to work,
  e.g. H100/A100 sm_80/90): not available here; would need a second box.

## Artifacts preserved (for a possible later revisit)

- `~/ds4/ref/dspark_ref_harness.py`, `~/ds4/ref/_tvm_ffi_shim.py` (on DGX +
  `issue468/ref/` locally, version-controlled)
- `~/ds4/ref-ckpt/model0-mp1.safetensors` (13 GB converted drafter+embed+head)
- `~/ds4/hf-dspark/` (shards 1/45/46/47/48 + index)
- `issue468/ref/inference/{model.py,kernel.py,convert.py,config.json,generate.py}`
  (full official reference code, fetched from the HF repo)

Pending user decision on which path to take for the Phase-4 regression gate.
