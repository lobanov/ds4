# Lead 08 iteration 26: exact small-M dense-family inventory

Date: 2026-07-18  
Hypothesis: V13  
Verdict: **DESIGN COMPLETE; one cross-quant family capability gate is conditionally admissible**

## Question and preflight correction

After the bounded verifier track exhausted its concrete mechanisms, this iteration returned to the
primary exact-output contract. The initial question was too narrow: an exact batched-F16 mechanism
would move the captured frontier through the router, but it could not exactify the already exposed
Q8 projections. Independent preflight therefore redesigned V13 as a read-only inventory of one
small-M dense family with Q8 and F16 specializations.

This is a design result, not an implementation, exactness, or speed result. The existing batched
verifier remains about 62 ms and divergent. The modeled K4 verifier byte-floor range is 39-43 ms,
not a proved lower bound. Lead 08 uses 43 ms as a conservative admission-policy base, leaving 7.5
ms to the `verify_ms(4) <= 50.5 ms` gate.

## Complete exposed dense surface

The full exact-hybrid dense surface exposed by V6-V13 contains seven low-K sites. Shapes are
`(M, N, K)` at K4; weight bytes use the actual F16 or Q8_0 row representation.

| Site | Format | K4 shape | Weight/layer | Current K4 path | Literal M1 reference | Integration seam |
|---|---|---:|---:|---|---|---|
| HC attention mixer | F16 | `(4, 24, 16384)` | 0.750 MiB | `ds4_gpu_matmul_f16_tensor` ext kernel | F16 M1 matvec | clean call at `ds4.c:18015` |
| attention Q-a | Q8_0 | `(4, 1024, 4096)` | 4.250 MiB | Q8 ext kernel | `kernel_mul_mv_q8_0_f32` | clean call at `ds4.c:18099` |
| attention KV | Q8_0 | `(4, 512, 4096)` | 2.125 MiB | Q8 ext kernel | `kernel_mul_mv_q8_0_f32` | clean call at `ds4.c:18128` |
| attention Q-b | Q8_0 | `(4, 32768, 1024)` | 34.000 MiB | Q8 ext fallback | `kernel_mul_mv_q8_0_f32` | clean fallback at `ds4.c:18222`; advertised fused hook is a no-op |
| attention output-B | Q8_0 | `(4, 4096, 8192)` | 34.000 MiB | Q8 ext inside combined output API | `kernel_mul_mv_q8_0_f32` | nonredundant batch low-only/suppress-B seam required |
| HC FFN mixer | F16 | `(4, 24, 16384)` | 0.750 MiB | `ds4_gpu_matmul_f16_tensor` ext kernel | F16 M1 matvec | clean call at `ds4.c:19578` |
| FFN router | F16 | `(4, 256, 4096)` | 2.000 MiB | `ds4_gpu_matmul_f16_tensor` ext kernel | F16 M1 matvec | clean call at `ds4.c:19665`; first captured unresolved boundary |

The F16 host dispatcher selects the M1 matvec for one token and the `r1_4` ext path for K4
(`ds4_metal.m:13360`). The corresponding Metal bodies are the Q8 M1 kernel at
`metal/dense.metal:180`, F16 M1 kernel at `metal/dense.metal:555`, and low-K ext specializations at
`metal/dense.metal:907`. Q-b's fused host hook returns zero unconditionally
(`ds4_metal.m:14048`), so the Q8 fallback above is the live path. The attention-output batch API
encodes both low projection and output-B (`ds4_metal.m:16313`); the existing row-wise exactifier
then overwrites B. A singleton low-only API already exists; production needs an equivalent
nonredundant batch seam, which could be a low-only extension, a suppress-B flag, or a split API.

## Traffic bound and required structure

The exposed weights total 77.875 MiB/layer: 74.375 MiB Q8 plus 3.500 MiB F16. One traversal over
43 layers is 3,348.625 MiB (3.270 GiB). Four independent M1 row calls traverse the weights four
times, adding 9.810 GiB over a load-once K4 design. It also adds 903 dispatches across all seven
sites (`7 sites * 3 extra rows * 43 layers`); the three F16 sites alone add 387. This rules out M
independent dispatches or a grid-Y wrapper that merely launches M1 threadgroups. It does not prove
the proposed family will meet the timing gate.

The shared family must preserve the literal M1 arithmetic DAG independently for each token while
sharing only weight fetch/decode:

```text
for each output tile in literal M1 grid order:
    initialize acc[token][lane] independently
    for each inner block in literal M1 order:
        w = load/decode weight block once
        for token = 0 .. M-1:
            acc[token] = literal_m1_step(acc[token], w, x[token])
    for token = 0 .. M-1:
        y[token] = literal_m1_reduce_and_store(acc[token])
```

For each specialization this means the same M1 per-lane loop, grouping, NSG/SIMD/threadgroup
reduction, and store order. Weight reuse is allowed; cross-token accumulation, a different tree,
or the current ext reduction is not. The API should expose one `exact-row small-M dense` operation
with Q8/F16 specializations and M=2..8 variants, not seven bespoke correctness patches.

## Correctness and economic gates

The direct kernel gate is bit equality against M independent production M1 calls for every listed
shape, representative real model offsets, M=2..8, and adversarial plus random finite activations.
The graph gate first moves the captured layer-0 frontier through the router, then requires all-layer
and full-corpus zero output differences. Exactness at the router alone is not full-path exactness:
compressed attention, routed down/sum, shared FFN, later layer boundaries, and the output head
remain independently unresolved.

Keep proved lower bounds separate from the conservative policy screen:

```text
LB_exact = independently validated end-to-end lower bound
        or sum(proved non-overlapping unavoidable component lower bounds)

policy_projection = 43.0 ms conservative modeled base
                  + cumulative matched exactness-delta upper bound
```

The earlier row-wise diagnostic timing changes are confounded upper observations, not lower bounds,
and must not enter `LB_exact`. Only an independently validated `LB_exact > 50.5 ms` proves economic
impossibility. The family policy stops when its conservative projection exceeds 50.5 ms, without
claiming that mathematical lower-bound result.

A prototype passes admission only if it covers both formats, shares each weight traversal, provides
a nonredundant output-B seam, and passes direct bit tests. Its intended timing comparator is the
current production K4 ext kernels over the same seven sites, inputs, model offsets, and 43 layers;
the candidate-minus-ext paired delta's upper confidence bound must be at most 7.5 ms. Iteration 27
must predeclare the fixed-work harness, synchronization, statistic, rounds/order, retained outputs,
reproduction command, and whether batch-low seam/host overhead is inside the timed boundary before
any V15 code is written. This is only an admission screen: later rollout must remeasure a cumulative
matched delta rather than assume isolated upper deltas compose.
F16-only success or a Q8 path that requires K traversals is `REDESIGN`, not a partial production
patch.

## Decision

V13's inventory is complete and the cross-quant design is structurally plausible: literal M1
bodies exist for both formats, five sites have clean call seams, Q-b uses a visible fallback, and
output-B has a specific nonredundant-seam requirement. No source fact yet proves that the required M1 DAG
can share Q8/F16 weights within the 7.5 ms policy-delta budget.

Conditionally admit exactly one V15 family-level capability prototype after an iteration-27
protocol preflight. It must build the common Q8/F16 skeleton,
exercise all seven shapes in the direct bit gate, and measure cumulative all-layer delta before
any graph rollout. If it cannot preserve M1 arithmetic with one weight traversal, cannot provide a
nonredundant output-B seam, or exceeds the economic gate, stop the exact dense-family branch. Do not build a
router-only helper or infer a primary-goal speedup from this inventory.

## Audit record

The first independent red-team returned `FIX`. It validated all seven shapes, weight arithmetic,
the corrected 903-dispatch total, Q-b fallback, combined output behavior, and structural
plausibility. It required five corrections applied above: keep the modeled 39-43 ms floor distinct
from proved lower bounds; require an executable iteration-27 timing preflight; describe output-B's
need mechanism-neutrally; correct four callsite lines; and qualify old shared-family closures as the
existing batch family. The first repeat returned `FIX` only because independently proved lower bounds
were not explicitly required to be non-overlapping before addition. The formula and cumulative
matched-delta rule now encode that constraint. The final repeat verified the current files and
returned `COMMIT`; residual risk is explicitly design-only, with no implementation, exactness,
timing, or full-path speed result.
