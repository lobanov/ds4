## The Bug

[ drafter_body.py:50](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:50) is wrong:

```python
comb = comb / (comb.sum(-1, keepdim=True) + eps)
```

Oracle [hc_primitives.py:46](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/hc_primitives.py:46):

```python
comb = comb / comb.sum(axis=-1, keepdims=True) + eps
```

One-line fix:

```python
comb = comb / comb.sum(-1, keepdim=True) + eps
```

The `eps` is added after the initial row-softmax, not inside its denominator. Without it, saturated comb rows keep near-zero entries through Sinkhorn, changing HC residual mixing.

## Evidence

Actual `grounded_observatory` step 1, layer 0 attention sub-block:

```text
hc_pre yd max 0.0
hc_pre post max 4.66e-10
hc_pre comb max 0.672851 mean 0.139179
attn out max 0.000153
after hc_post max 0.059216 mean 0.003325
patched hc_pre comb max 0.0
patched after hc_post max 0.000080 mean 0.00000284
```

Pure numpy replay, using oracle attention+MoE, with only this torch-style no-eps Sinkhorn change:

```text
after attn max 0.059215 mean 0.003325
after full layer0 max 0.143405 mean 0.005014
```

That reproduces the reported layer-0 `~0.14` divergence without torch BLAS involved.

## Confidence

High. This is the systematic layer-0 source.

Secondary candidates: RoPE/attention are not the source here; with fixed `comb`, attention-path residual diff drops to `~8e-5`. `hc_post` axis semantics are consistent with oracle. MoE is only amplifying the already-wrong HC residual state.