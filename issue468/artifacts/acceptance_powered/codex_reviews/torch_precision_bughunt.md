## Per-Op Verdict

- RoPE: matches. Synthetic max `0.0`; cached cos/sin table vs `precompute_rope(64,4096)` max `0.0`.
- MLA attention: matches within CPU matmul order. `sparse_attn` max `5.96e-08`; real L0 `_attn` max `1.53e-4`, mean `2.05e-5`.
- HC split/pre: matches. Random `hc_split` max `5.96e-08`; real L0 `hc_pre` max `7.45e-09`.
- HC post: diverges. Synthetic current torch vs numpy max `14.86`, mean `1.98`; fixed broadcast max `0.0`.
- MoE: matches. Gate idx equal, weights max `0.0`; cached expert0 vs lazy GGUF dequant max `0.0`; routed/shared SwiGLU max `1.86e-09`.
- main_proj/RMSNorm: matches. Real `main_proj+rmsnorm` max `0.0`; synthetic RMSNorm max `2.38e-07`.
- Step loop/KV window: matches for checked steps `1,2,127,128,129`.

## Root Cause

Fixable porting bug in [drafter_body.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:67):

```python
residual.unsqueeze(-3)
```

The trusted numpy oracle at [hc_primitives.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/hc_primitives.py:82) uses:

```python
residual_hc[..., None, :]
```

That is equivalent to torch `residual.unsqueeze(-2)`, not `unsqueeze(-3)`.

Stage 2 mis-attributed this. The bug is tiny at L0 because the initial HC residual copies are identical, then nonlinear layers amplify it:

```text
layer 0: max 0.0002289, mean 0.0000102
layer 1: max 4.5876,    mean 0.3071
layer 2: max 112.9193,  mean 2.3691
```

That was reproduced in pure numpy by changing only `hc_post` broadcast semantics, so it is not inherent BLAS/MPS chaos.

## Fix

```python
def hc_post(x, residual, post, comb):
    return post.unsqueeze(-1) * x.unsqueeze(-2) + (
        comb.unsqueeze(-1) * residual.unsqueeze(-2)
    ).sum(-2)
```

## Decisive Test

Run one real all-3-layer step with oracle ops and lazy expert dequant, swapping only `hc_post` between oracle broadcast and current torch broadcast. It reproduces the catastrophic layer growth above without torch body, MPS, or full expert-cache loading.