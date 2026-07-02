# `attn_kv` Local Perturbation Screen on `ctx_08192`

Date: 2026-07-02

## Purpose

Follow the recommendation from `86` and `90` to test whether nearby attention
still has local sensitivity around the baseline, rather than only asking
whether the full source tensor copy is good.

The earlier result in `85` was:

- full-source `attn_kv` splice is slightly negative on mean

But `88` showed that `attn_kv` still wins a couple of recoverable steps inside
the impossible all-artifact chooser.

So the question here was narrower:

- does small movement along the baseline→source `attn_kv` direction show any
  useful local derivative near the baseline?

## Method

Starting from the exported source arrays:

- `/tmp/mtp.0.attn_kv.weight.ref.npy`
- `/tmp/mtp.1.attn_kv.weight.ref.npy`
- `/tmp/mtp.2.attn_kv.weight.ref.npy`

and the baseline GGUF tensors, build simple mixed arrays:

```text
mixed = baseline + alpha * (source - baseline)
```

Tested alphas on all three stages together:

- `alpha = -0.25`
- `alpha = +0.25`
- `alpha = +0.50`

The existing oracle scorer in
`issue468/ref/measure_spliced_ref_oracle_b2.py` already supports
`--replace-npy`, so no new scorer was needed.

## Context measured

Measured context:

- `ctx_08192`

This is the same context where:

- the full-source `attn_kv` splice had the clearest recoverable-step wins in
  `88`

## Reference points

On `ctx_08192`:

- baseline:
  - accepted `4.072368421052632`
  - committed `4.473684210526316`
- full-source `attn_kv` splice:
  - accepted `3.9979440789473686`
  - committed `4.442023026315789`

## Results

### `alpha = -0.25`

- accepted `3.9788240131578947`
- committed `4.427014802631579`

### `alpha = +0.25`

- accepted `3.979440789473684`
- committed `4.427014802631579`

### `alpha = +0.50`

- accepted `3.979440789473684`
- committed `4.427014802631579`

## Read

This is a negative local-sensitivity result.

What it shows:

- small signed movement along the baseline→source `attn_kv` direction is
  essentially flat
- and already slightly worse than the full-source endpoint on committed score

So there is no clear sign here of:

- a beneficial local derivative near the baseline
- or an obvious asymmetric `+/-` response that would justify a larger local
  attention search along this direction

## Decision impact

This does **not** fully prove that nearby attention is exhausted as a
mechanism.

But it does materially weaken:

- source-direction local `attn_kv` perturbation as a lead branch

The current read is:

- `attn_kv` can still win a few recoverable steps inside an impossible chooser
- but along the actual source direction, both the endpoint and nearby local
  perturbations remain below baseline on `ctx_08192`

So if nearby attention is revisited, it should probably require:

- a different perturbation basis than simple baseline→source interpolation
- or a stronger reason than the current recoverable-step chooser wins alone
