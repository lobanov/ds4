# Candidate Complementarity Screen on the 2ctx Ref-Oracle Recoverable Set

Date: 2026-07-02

## Purpose

Follow the recommendation from `86` to check whether the current flat / weakly
positive candidate families are cancelling each other across states.

Question:

- if we choose the best already-measured candidate **per anchor step**, how much
  headroom exists on the current 2-context same-checkpoint recoverable set?

This is an oracle-style branch discriminator, not a deployable candidate path.

## New helper

Added:

- `issue468/analyze_candidate_complementarity.py`

Purpose:

- load several existing `*.b2.json` artifacts
- compare `per_step_accepted` at shared context/step granularity
- compute a best-available per-step envelope
- report both:
  - full-context mean uplift
  - recoverable-step-only uplift using an existing recoverable-gap manifest

So this generalizes the older q-dump envelope logic from `49` to the current
mixed candidate family.

## Inputs

Sweep root:

- `/private/tmp/dspark_sweep2ctx`

Recoverable-step definition:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_reforacle2ctx_test.anchor_weights.json`

That same-checkpoint ref-oracle manifest marks these recoverable steps:

- `ctx_08192`: `2, 4, 5, 6, 11`
- `ctx_16384`: `1, 6, 7`

## Screen 1: runtime-side artifacts only

Compared bundle-local candidates:

- `baseline`
- `layer2gateup`
- `layer2up`
- `mainproj_q8imat`
- `dense_combo_q8imat`

Command:

```sh
python3 issue468/analyze_candidate_complementarity.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --bundle-label baseline=baseline \
  --bundle-label layer2gateup=candidate-recoverablegap_boosted2ctx_layer2gateup \
  --bundle-label layer2up=candidate-recoverablegap_boosted2ctx_layer2up \
  --bundle-label mainproj_q8imat=candidate-recoverablegap_boosted2ctx_mainproj_q8imat \
  --bundle-label dense_combo_q8imat=candidate-recoverablegap_boosted2ctx_dense_combo_q8imat \
  --recoverable-details-json /private/tmp/dspark_sweep2ctx/recoverablegap_reforacle2ctx_test.anchor_weights.json \
  --out-json /private/tmp/dspark_sweep2ctx/runtime_complementarity_2ctx.json
```

Result:

- mean best-available uplift across the two full contexts:
  - `+3.1927192037347396%`
- mean best-available uplift on recoverable steps only:
  - `+13.189463797814204%`

Per-context full-mean best-of:

| context | baseline accepted | runtime best-of | delta |
|---|---:|---:|---:|
| `8192` | `4.072368421052632` | `4.300986842105263` | `+5.613893376413559%` |
| `16384` | `4.2368421052631575` | `4.26953125` | `+0.7715450310559202%` |

Recoverable-step winner counts:

- `layer2gateup`: `4`
- `mainproj_q8imat`: `2`
- `layer2up`: `1`
- `baseline`: `1`

Recoverable-step winners:

- `ctx_08192`
  - step `2`: `layer2up`
  - step `4`: `mainproj_q8imat`
  - step `5`: `baseline`
  - step `6`: `layer2gateup`
  - step `11`: `layer2gateup`
- `ctx_16384`
  - step `1`: `layer2gateup`
  - step `6`: `layer2gateup`
  - step `7`: `mainproj_q8imat`

Read:

- there is real step-level complementarity inside the current runtime-side
  family
- `layer2gateup` remains the strongest routed-local winner on recoverable steps
- `mainproj_q8imat` complements it materially rather than redundantly

But:

- even this impossible runtime-only chooser still averages only `+3.19%`
  across the two contexts
- so the current measured runtime family still falls short of the `+5%` gate

## Screen 2: all currently available local artifacts

Added oracle-only artifacts where available:

- `ref_oracle`
- `splice_gateinp`
- `splice_attnkv`
- `splice_attnqa` on `ctx_08192` only

Command:

```sh
python3 issue468/analyze_candidate_complementarity.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --bundle-label baseline=baseline \
  --bundle-label layer2gateup=candidate-recoverablegap_boosted2ctx_layer2gateup \
  --bundle-label layer2up=candidate-recoverablegap_boosted2ctx_layer2up \
  --bundle-label mainproj_q8imat=candidate-recoverablegap_boosted2ctx_mainproj_q8imat \
  --bundle-label dense_combo_q8imat=candidate-recoverablegap_boosted2ctx_dense_combo_q8imat \
  --external-b2 ref_oracle=8192:/tmp/ref-oracle-fp8-ctx08192-19.b2.json \
  --external-b2 ref_oracle=16384:/tmp/ref-oracle-fp8-ctx16384-19.b2.json \
  --external-b2 splice_gateinp=8192:/tmp/splice-ref-gateinp-all3-ctx08192-19.b2.json \
  --external-b2 splice_gateinp=16384:/tmp/splice-ref-gateinp-all3-ctx16384-19.b2.json \
  --external-b2 splice_attnkv=8192:/tmp/splice-ref-attnkv-all3-ctx08192-19.b2.json \
  --external-b2 splice_attnkv=16384:/tmp/splice-ref-attnkv-all3-ctx16384-19.b2.json \
  --external-b2 splice_attnqa=8192:/tmp/splice-ref-attnqa-all3-ctx08192-19.b2.json \
  --recoverable-details-json /private/tmp/dspark_sweep2ctx/recoverablegap_reforacle2ctx_test.anchor_weights.json \
  --out-json /private/tmp/dspark_sweep2ctx/all_available_complementarity_2ctx.json
```

Result:

- mean best-available uplift across the two full contexts:
  - `+3.999917060927771%`
- mean best-available uplift on recoverable steps only:
  - `+21.04999146174863%`

Per-context full-mean best-of:

| context | baseline accepted | all-artifact best-of | delta |
|---|---:|---:|---:|
| `8192` | `4.072368421052632` | `4.329975328947368` | `+6.3257269789983805%` |
| `16384` | `4.2368421052631575` | `4.307771381578948` | `+1.674107142857162%` |

Recoverable-step winner counts:

- `ref_oracle`: `5`
- `splice_attnkv`: `2`
- `layer2gateup`: `1`

Recoverable-step winners:

- `ctx_08192`
  - step `2`: `splice_attnkv`
  - step `4`: `splice_attnkv`
  - step `5`: `ref_oracle`
  - step `6`: `layer2gateup`
  - step `11`: `ref_oracle`
- `ctx_16384`
  - step `1`: `ref_oracle`
  - step `6`: `ref_oracle`
  - step `7`: `ref_oracle`

Read:

- the branch still has meaningful **state-conditional headroom**
- negative source-splice means were partly hiding the fact that
  `attn_kv` wins a couple of the hardest currently identified recoverable steps
- but this is still a sparse effect, not a broad positive family

And critically:

- even the impossible all-artifact chooser remains below `+5%` mean on the two
  contexts overall

So the complementarity screen is positive in the narrow sense:

- there is nontrivial cancellation across states

But it is negative in the broader decision sense:

- cancellation alone is not enough to rescue the current measured family to the
  assignment gate

## Decision impact

This screen updates branch interpretation as follows:

1. `layer2gateup` still looks like the best runtime-side routed-local seed for
   acceptance-aware local search
2. `mainproj_q8imat` is the strongest complementary dense runtime-side control
3. nearby attention should not return as another source-copy lead branch, but
   `attn_kv` has enough recoverable-step wins to justify local perturbation
   testing before closing that mechanism completely
4. any future positive claim should be judged against the stronger ref-oracle
   recoverable set, not only context means

## Next move

The highest-signal next experiment remains:

- acceptance-aware local `Q4_K` search around `layer2gateup` / `layer2up`

with this complementarity read serving as the justification that:

- there is real but fragmented state-level signal there
- yet the currently measured family is still too weak to pass by selection
  alone
