# First Frontier Artifact Builds

Date: 2026-07-02

## Purpose

Record the first successful materialization of the routed-expert `Q2/Q4`
frontier endpoints defined in `96`.

This note now covers:

- buildability
- exact artifact size
- first 4-context acceptance measurements
- disk-space state

## Result

The first frontier candidates were built successfully into:

- `/private/tmp/dspark_pareto_q2q4`

Artifacts built so far:

| label | bytes | GiB |
|---|---:|---:|
| `all_q2` | `6054122176` | `5.638335` |
| `only_mtp0_q4` | `7866061504` | `7.325835` |
| `only_mtp1_q4` | `7866061504` | `7.325835` |
| `only_mtp2_q4` | `7866061504` | `7.325835` |
| `mtp01_q4` | `9678000832` | `9.013335` |
| `mtp02_q4` | `9678000832` | `9.013335` |
| `mtp12_q4` | `9678000832` | `9.013335` |
| `mtp0_full_mtp2_gateup_q4` | `9174684352` | `8.544585` |
| `mtp0_full_mtp2_gate_q4` | `8520372928` | `7.935210` |
| `mtp0_full_mtp2_up_q4` | `8520372928` | `7.935210` |
| `mtp0_full_mtp2_down_q4` | `8369377984` | `7.794585` |
| `mtp2_full_mtp0_gateup_q4` | `9174684352` | `8.544585` |
| `mtp2_full_mtp0_down_q4` | `8369377984` | `7.794585` |

These exactly match the size estimates from `95` and `96` to the byte-level
`approx_file_bytes` reported by the quantizer dry-run / build path.

## Build-path correction

The first build attempt exposed a small helper bug:

- the planner emitted overrides as `--tensor-type=NAME=TYPE`

but `deepseek4-quantize` expects:

- `--tensor-type NAME=TYPE`

The helper surface is now split correctly:

- `plan_q2_q4_frontier_candidates.py` emits bare `NAME=TYPE`
- `build_q2_q4_frontier_candidates.py` expands each one into a separate
  `--tensor-type` argv pair

After that correction, both endpoint artifacts built cleanly without template
or metadata issues.

## Why these two points first

These are the most informative first endpoints:

1. `all_q2`
2. `only_mtp2_q4`

They answer the first practical question of the new branch:

- what is the full routed-`Q2` size floor?
- what is the cost of keeping only the final drafter layer at baseline routed
  precision?

The memory frontier therefore has two concrete reduced-size anchor points below
the `Q4_K` baseline.

## 4-context measurement result

Measured artifacts were scored on the existing 4-context sweep root:

- `/private/tmp/dspark_sweep8`

using:

- `issue468/run_dspark_weighted_from_sweep_root.py`
- existing bundle contexts `8192`, `16384`, `24576`, `32768`
- `19` draft steps
- `128` B2 trials

Compatibility status used below:

- `drop-in`: already loadable and measurable in the frozen `ds4` runtime with
  existing kernels and metadata
- `can be made compatible`: expected to fit the frozen runtime with conversion
  or packaging work, but not yet a measured drop-in GGUF
- `requires new kernels`: would need runtime/kernel support beyond the current
  frozen `ds4` surface

Mean results:

| label | size GiB | mean accepted | accepted delta vs baseline | mean committed | committed delta vs baseline | runtime status |
|---|---:|---:|---:|---:|---:|---|
| baseline `Q4_K` | `10.700835` | `4.229749` | `0.000%` | `4.557052` | `0.000%` | `drop-in` |
| `mtp02_q4` | `9.013335` | `4.212788` | `-0.401%` | `4.536595` | `-0.449%` | `drop-in` |
| `mtp0_full_mtp2_gateup_q4` | `8.544585` | `4.202200` | `-0.651%` | `4.532689` | `-0.535%` | `drop-in` |
| `mtp0_full_mtp2_gate_q4` | `7.935210` | `0.638055` | `-84.915%` | `1.629626` | `-64.239%` | `drop-in` |
| `mtp0_full_mtp2_up_q4` | `7.935210` | `0.586143` | `-86.142%` | `1.584601` | `-65.227%` | `drop-in` |
| `mtp0_full_mtp2_down_q4` | `7.794585` | `4.196752` | `-0.780%` | `4.526933` | `-0.661%` | `drop-in` |
| `mtp2_full_mtp0_gateup_q4` | `8.544585` | `4.191612` | `-0.902%` | `4.521382` | `-0.783%` | `drop-in` |
| `mtp2_full_mtp0_down_q4` | `7.794585` | `4.175164` | `-1.290%` | `4.519840` | `-0.817%` | `drop-in` |
| `mtp01_q4` | `9.013335` | `4.202508` | `-0.644%` | `4.542044` | `-0.329%` | `drop-in` |
| `only_mtp0_q4` | `7.325835` | `4.185958` | `-1.035%` | `4.522718` | `-0.753%` | `drop-in` |
| `only_mtp2_q4` | `7.325835` | `4.155633` | `-1.752%` | `4.495785` | `-1.344%` | `drop-in` |
| `mtp12_q4` | `9.013335` | `4.135691` | `-2.224%` | `4.487973` | `-1.516%` | `drop-in` |
| `all_q2` | `5.638335` | `4.123869` | `-2.503%` | `4.476151` | `-1.775%` | `drop-in` |
| `only_mtp1_q4` | `7.325835` | `4.105674` | `-2.933%` | `4.468750` | `-1.938%` | `drop-in` |

Reference points outside this GGUF frontier should be read differently:

| reference | role | runtime status | reason |
|---|---|---|---|
| raw-HF DSpark oracle from `93` | ceiling/reference only | `can be made compatible` | source tensors are architecturally compatible with the frozen runtime, but they are not themselves a measured drop-in deployment artifact until converted and packaged into a GGUF |
| numpy oracle / ref-ckpt paths | analysis only | `can be made compatible` | useful for oracle-level scoring and tensor surgery, but not directly loadable by production `ds4` without conversion into the existing GGUF/runtime surface |
| any FP8-or-other new tensor-type experiment outside current GGUF types | hypothetical search space | `requires new kernels` | frozen runtime does not currently expose those expert/storage kernels |

Per-context accepted-token deltas vs baseline:

| context | `mtp02_q4` | `mtp0_full_mtp2_gateup_q4` | `mtp0_full_mtp2_gate_q4` | `mtp0_full_mtp2_up_q4` | `mtp0_full_mtp2_down_q4` | `mtp2_full_mtp0_gateup_q4` | `mtp2_full_mtp0_down_q4` | `mtp01_q4` | `only_mtp0_q4` | `only_mtp2_q4` | `mtp12_q4` | `only_mtp1_q4` | `all_q2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8192` | `-0.216%` | `-0.020%` | `-90.416%` | `-91.183%` | `-0.374%` | `-1.691%` | `-1.396%` | `-0.845%` | `-0.167%` | `-3.175%` | `-4.109%` | `-3.785%` | `-3.234%` |
| `16384` | `-0.720%` | `-0.486%` | `-77.823%` | `-82.862%` | `-1.041%` | `-0.739%` | `-1.002%` | `-0.204%` | `-1.138%` | `-1.031%` | `-0.788%` | `-2.091%` | `-2.033%` |
| `24576` | `-0.234%` | `-0.964%` | `-91.426%` | `-90.998%` | `-1.071%` | `-0.185%` | `-1.946%` | `-0.973%` | `-1.148%` | `-1.713%` | `-3.689%` | `-3.474%` | `-2.453%` |
| `32768` | `-0.432%` | `-1.123%` | `-80.121%` | `-79.670%` | `-0.634%` | `-0.998%` | `-0.825%` | `-0.557%` | `-1.670%` | `-1.113%` | `-0.355%` | `-2.400%` | `-2.304%` |

## Interpretation

The first reduced-size frontier is now visible.

The full single-layer-`Q4` tier at `7.325835 GiB` is now resolved:

1. `only_mtp0_q4`
2. `only_mtp2_q4`
3. `only_mtp1_q4`

The currently measured `9.013335 GiB` tier is now:

1. `mtp02_q4`
2. `mtp01_q4`
3. `mtp12_q4`

The first focused mixed-family point lands between the coarse tiers:

- `mtp0_full_mtp2_gateup_q4` at `8.544585 GiB`
- full `Q4` for all routed tensors in `mtp.0`
- `Q4` only for `ffn_gate_exps.weight` and `ffn_up_exps.weight` in `mtp.2`
- `Q2` retained for `mtp.2` `ffn_down_exps.weight`

The second focused mixed-family point pushes farther down the size curve:

- `mtp0_full_mtp2_gate_q4` at `7.935210 GiB`
- full `Q4` for all routed tensors in `mtp.0`
- `Q4` only for `mtp.2` `ffn_gate_exps.weight`
- `Q2` retained for `mtp.2` `ffn_up_exps.weight` and
  `ffn_down_exps.weight`

The third focused mixed-family point pushes farther down the size curve:

- `mtp0_full_mtp2_up_q4` at `7.935210 GiB`
- full `Q4` for all routed tensors in `mtp.0`
- `Q4` only for `mtp.2` `ffn_up_exps.weight`
- `Q2` retained for `mtp.2` `ffn_gate_exps.weight` and
  `ffn_down_exps.weight`

The fourth focused mixed-family point pushes farther down the size curve:

- `mtp0_full_mtp2_down_q4` at `7.794585 GiB`
- full `Q4` for all routed tensors in `mtp.0`
- `Q4` only for `mtp.2` `ffn_down_exps.weight`
- `Q2` retained for `mtp.2` `ffn_gate_exps.weight` and
  `ffn_up_exps.weight`

The fifth focused mixed-family point is the symmetry test at the earlier
middle tier:

- `mtp2_full_mtp0_gateup_q4` at `8.544585 GiB`
- full `Q4` for all routed tensors in `mtp.2`
- `Q4` only for `mtp.0` `ffn_gate_exps.weight` and
  `ffn_up_exps.weight`
- `Q2` retained for `mtp.0` `ffn_down_exps.weight`

The sixth focused mixed-family point closes the same symmetry test at the
smaller tier:

- `mtp2_full_mtp0_down_q4` at `7.794585 GiB`
- full `Q4` for all routed tensors in `mtp.2`
- `Q4` only for `mtp.0` `ffn_down_exps.weight`
- `Q2` retained for `mtp.0` `ffn_gate_exps.weight` and
  `ffn_up_exps.weight`

`all_q2` establishes the current size floor:

- about `47.3%` smaller than baseline
- with about `2.50%` lower mean accepted tokens

`only_mtp0_q4` is currently the best measured reduced-size knee:

- about `31.5%` smaller than baseline
- with only about `1.04%` lower mean accepted tokens

That point does **not** dominate baseline, but it clearly dominates `all_q2`
in quality at a still-material size reduction. The gain over `all_q2` is:

- `+0.062089` mean accepted
- for `+1.687500 GiB`

`only_mtp2_q4` remains better than `all_q2`, but it is now clearly inferior to
`only_mtp0_q4` at the same byte cost:

- `-0.030325` mean accepted vs `only_mtp0_q4`

`only_mtp1_q4` is the weakest same-size variant and does not justify further
attention as a deployment point:

- `-0.080284` mean accepted vs `only_mtp0_q4`
- `-0.049959` mean accepted vs `only_mtp2_q4`
- even `-0.018195` mean accepted vs `all_q2` despite costing `+1.687500 GiB`

The first `9.013335 GiB` point, `mtp01_q4`, does recover additional quality:

- mean accepted `4.202508`
- only `-0.644%` vs baseline
- `+0.016550` mean accepted vs `only_mtp0_q4`

But the recovery is still modest for the added size:

- `+1.687500 GiB` over `only_mtp0_q4`
- closes only about `37.8%` of the remaining accepted-token gap from
  `only_mtp0_q4` to baseline

`mtp02_q4` is better still:

- mean accepted `4.212788`
- only `-0.401%` vs baseline
- `+0.026830` mean accepted vs `only_mtp0_q4`
- `+0.010280` mean accepted vs `mtp01_q4`

Even so, the larger-tier improvement remains incremental rather than dramatic:

- `+1.687500 GiB` over `only_mtp0_q4`
- closes about `61.3%` of the remaining accepted-token gap from
  `only_mtp0_q4` to baseline

The first focused `mtp0/mtp2` family probe is a more interesting trade:

- `mtp0_full_mtp2_gateup_q4` mean accepted `4.202200`
- only `-0.651%` vs baseline
- `+0.016242` mean accepted vs `only_mtp0_q4`
- only `-0.010588` mean accepted vs `mtp02_q4`
- effectively tied with `mtp01_q4` at `-0.000308` mean accepted

That result matters because it buys most of the `mtp02_q4` quality recovery
while saving:

- `0.468750 GiB` vs `mtp02_q4`
- `0.468750 GiB` vs `mtp01_q4`

So the new evidence is that imatrix-style `Q2` still appears useful, but not
as a blunt all-layer choice. The promising regime is selective:

- keep `mtp.0` fully at `Q4`
- spend additional `Q4` budget in `mtp.2` primarily on gate/up tensors
- leave lower-value routed tensors such as `mtp.2` down experts at `Q2`

The second focused `mtp0/mtp2` family probe sharpens that conclusion rather
than weakening it:

- `mtp0_full_mtp2_down_q4` mean accepted `4.196752`
- only `-0.780%` vs baseline
- `+0.010794` mean accepted vs `only_mtp0_q4`
- but `-0.005448` mean accepted vs `mtp0_full_mtp2_gateup_q4`
- and `-0.016036` mean accepted vs `mtp02_q4`

That makes `mtp0_full_mtp2_down_q4` a real frontier point in the strict
size/quality sense, but not the preferred local direction inside this family.
Relative to `mtp0_full_mtp2_gateup_q4` it saves:

- `0.750000 GiB`

but gives back enough quality that the structural conclusion is now clearer:

- if limited extra `Q4` budget is spent in `mtp.2`, `gate/up` is the stronger
  destination than `down`
- `mtp.2 down` can be pushed back to `Q2` with only a modest loss, but it does
  not preserve quality as efficiently as `gate/up`
- the `mtp0`-anchored family still looks promising, but the local frontier now
  bends toward `gate/up`-first allocations

The new single-family `gate` split is not a frontier refinement at all. It is
a hard falsification:

- `mtp0_full_mtp2_gate_q4` mean accepted `0.638055`
- `-84.915%` vs baseline
- `-3.564145` mean accepted vs `mtp0_full_mtp2_gateup_q4`
- `-3.558697` mean accepted vs `mtp0_full_mtp2_down_q4`

This is strong enough to rule out the idea that the earlier `gate/up` win is
mostly driven by `mtp.2 gate` precision alone. The current branch read should
therefore be:

- `mtp.2 gate` by itself is catastrophically insufficient

The new single-family `up` split reaches the same conclusion:

- `mtp0_full_mtp2_up_q4` mean accepted `0.586143`
- `-86.142%` vs baseline
- `-3.616057` mean accepted vs `mtp0_full_mtp2_gateup_q4`
- `-3.610609` mean accepted vs `mtp0_full_mtp2_down_q4`
- even slightly worse than the already-failed `mtp0_full_mtp2_gate_q4`

So the single-family branch is now effectively closed:

- `mtp.2 gate` alone fails catastrophically
- `mtp.2 up` alone fails catastrophically
- the surviving interpretation is that the useful `mtp2 gate/up` spend is a
  coupled effect that depends on keeping both families at `Q4`
- `mtp0_full_mtp2_gateup_q4` remains the only viable focused point on that
  side of the branch

The symmetry test makes the layer-identity conclusion much harder to dismiss:

- `mtp2_full_mtp0_gateup_q4` mean accepted `4.191612`
- only `-0.902%` vs baseline
- `-0.010588` mean accepted vs `mtp0_full_mtp2_gateup_q4` at the same byte size
- `-0.005140` mean accepted vs `mtp0_full_mtp2_down_q4` despite costing
  `+0.750000 GiB`
- only `+0.035979` mean accepted vs `only_mtp2_q4` for `+1.218750 GiB`

So the best current read is no longer only "preserve gate/up where possible."
It is more specific:

- `mtp.0` is the strongest routed layer to keep fully at `Q4`
- additional `Q4` spend in `mtp.2` helps most when directed to `gate/up`
- the symmetric `mtp.2`-anchored allocation does not recover comparable
  quality at the same size

The smaller-tier symmetry close strengthens that conclusion further:

- `mtp2_full_mtp0_down_q4` mean accepted `4.175164`
- only `-1.290%` vs baseline
- `-0.021587` mean accepted vs `mtp0_full_mtp2_down_q4` at the same byte size
- `-0.016447` mean accepted vs `mtp2_full_mtp0_gateup_q4` despite identical
  full-`Q4` `mtp.2` anchoring
- worse even than `only_mtp0_q4` while costing `+0.468750 GiB`

That effectively closes the focused symmetry table:

- both `mtp0`-anchored variants beat their `mtp2`-anchored counterparts
- within the winning `mtp0`-anchored family, `mtp2 gate/up` beats `mtp2 down`
- within the losing `mtp2`-anchored family, even the better `gate/up` variant
  does not challenge the `mtp0`-anchored frontier

So the current local search conclusion is now strong enough to be operational:

- retire the `mtp2`-anchored branch from further frontier search
- keep future mixed `Q2/Q4` work concentrated on `mtp0`-anchored recipes
- treat `mtp0_full_mtp2_gateup_q4` as the best measured focused compromise
  below the `9.013335 GiB` tier so far

`mtp12_q4` falsifies the hope that any two-layer `Q4` recipe at this size is
automatically good:

- mean accepted `4.135691`
- `-0.076872` mean accepted vs `mtp02_q4`
- `-0.066592` mean accepted vs `mtp01_q4`
- only a small improvement over `all_q2` despite costing `+3.375000 GiB`

So the current evidence still supports the branch hypothesis that:

- the interesting frontier lies in mixed routed `Q2/Q4` recipes, not pure
  routed `Q2`

but it no longer supports the earlier assumption that `mtp.2` is the most
valuable single routed layer to keep at `Q4_K`.

Instead, the first same-size contrast suggests:

- layer identity matters materially even when total bytes are identical
- `mtp.0` is the strongest measured single-layer `Q4` keep on this metric
- `mtp.1` is a poor use of the `Q4` budget at this size
- `mtp.2` is helpful, but not the best single-layer keep
- `mtp02_q4` is the best measured larger-knee point so far
- the extra `Q4` layer beyond `only_mtp0_q4` helps, but by a modest amount
- `mtp12_q4` performs badly enough that the `9.013335 GiB` tier is probably
  characterized well enough for decision use

## Disk-space state

Current local state after build:

- `/private/tmp/dspark_pareto_q2q4`: about `103 GiB`
- `/private/tmp/dspark_sweep8`: about `6.6 GiB`
- free disk: about `267 GiB`

Transient reprobe artifacts for the measured 4-context runs were pruned after
summary extraction:

- deleted generated `metal_base_logits_19steps.bin`
- deleted generated `*.probe.stdout` / `*.probe.stderr`
- retained per-context `*.b2.json` plus top-level summary JSON/TSV

No further cleanup was applied here because:

- the seven frontier artifacts are active research outputs
- the remaining large files in `/private/tmp/dspark_sweep2ctx` are the three
  intentionally retained reference GGUFs plus the imatrix path used by this
  sweep

So disk pressure is currently acceptable, but future frontier expansion should
continue pruning superseded GGUFs as soon as they are no longer needed.

## Immediate next step

The next branch action should be more selective than the original full-grid
plan:

- decide whether the modest `mtp02_q4` gain over `only_mtp0_q4` is already
  enough to stop the larger-tier search entirely
- if any further search is justified, it should continue in the targeted
  `mtp0/mtp2` tensor-family space rather than returning to the coarse layer
  grid
- `mtp0_full_mtp2_down_q4` is now measured and underperforms the earlier
  `mtp0_full_mtp2_gateup_q4` quality recovery
- the symmetric `mtp2_full_mtp0_gateup_q4` probe is now measured and supports
  `mtp.0` dominance rather than a layer-agnostic gate/up story
- `mtp2_full_mtp0_down_q4` is now measured and confirms that the
  `mtp2`-anchored branch is not competitive
- `mtp0_full_mtp2_gate_q4` is now measured and catastrophically falsifies
  `gate` as a standalone `mtp.2` spend
- `mtp0_full_mtp2_up_q4` is now measured and catastrophically falsifies
  `up` as a standalone `mtp.2` spend
- the next highest-value follow-on should therefore move away from
  single-family routed splits and instead test either:
  - small dense add-backs on top of `mtp0_full_mtp2_gateup_q4`, or
  - coupled routed spends that preserve the successful `mtp2 gate/up` pair
