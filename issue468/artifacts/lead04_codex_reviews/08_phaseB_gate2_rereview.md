## Is the dtype refutation sound? (verify float32-Q2==F16-Q2 is real)

Mostly yes, for Q2 only.

Evidence:
- `--dtype` is wired: parsed at [run_lead03_torch_measure.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_torch_measure.py:105), mapped to `torch.float16/float32` at [line 110](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_torch_measure.py:110), passed to `build_body`/`build_head` at [lines 113-114](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_torch_measure.py:113).
- Cache risk exists: measure mode skips existing JSONs by output path only at [line 158](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_torch_measure.py:158). Same `--out` would silently reuse.
- Actual artifacts: `combined300_float32/per_prompt` has 240 files, separate from `combined300`, with sequential July 10 mtimes. All 240 common JSONs are exactly equal after parsing, including `drafts`; mean p1 `0.793041`.

Caveat: no run log or dtype metadata in those JSONs. Still, separate output tree plus mtimes makes cache reuse unlikely. M6 dtype confound is refuted for Q2 drafter arithmetic.

But this does not remove the bigger confound: FP is vLLM/native capture and Q2 is ds4/IQ2XXS capture. Different engine, kernels, hidden extraction path, and greedy trajectory. So the +5.3pp is dtype-consistent, not a clean causal “hidden precision only” comparison.

## F16 anomaly: C1 (capture error) or genuine F16 issue? Does it invalidate the float32 GO?

It does not mathematically prove C1, but C1 is the leading explanation.

The current FP analyzer now forces F16 at [analyze_phaseB_gap.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/analyze_phaseB_gap.py:121). The saved positive `/tmp/phaseB_results/analysis.log` is from the older float32 path: its banner lacks the dtype print. I did not find a retained full F16-on-FP JSON proving p1≈0.62, so I treat that anomaly as reported, not independently reproduced.

If true: F16-FP `0.62` vs F16-Q2 `0.79` is not a small numerical drift. A drafter distilled on native hiddens should not crater on correct native hiddens while staying sane on degraded Q2 hiddens. Possible alternatives are F16/MPS range sensitivity or a body/head implementation issue exposed by larger FP hidden ranges. But either way, it invalidates the float32 GO for deployment.

C1 remains unresolved: the hook calls `mhc_post_tilelang` on vLLM layer outputs in [dspark_hc_patch.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/dspark_hc_patch.py:82), then the converter HC-means and concats. Plausible, not algebraically proven equivalent to ds4 `after_ffn_hc`.

## Is the GO defensible, or should it be HOLD/STOP? (weigh F16-deployment vs float32)

Not GO.

Reasons:
- Deployment-relevant F16 reportedly reverses sign: FP hiddens hurt by ~17pp.
- Float32 result is positive but not deployment-relevant if F16 is the target.
- CI lower `+3.81pp` is basically the known ~4pp kernel systematic. Thin margin.
- Corpus is easy-side. Dolly is only `+2.73pp`; earlier exactness/code pilot was negative.
- Result is still cross-engine/cross-trajectory, not pure hidden-precision attribution.

I would not STOP the research claim yet, because a capture representation bug could be invalidating the F16 result. But I would STOP any action based on the current FP capture path at F16.

## Prompt types/lengths for balance

Use equal strata:
- Hard code: debugging, unit tests, refactors, algorithmic generation.
- Exactness/synthesis: deterministic transformations, timelines, structured extraction.
- JSON/schema: stricter and longer than current jsonex.
- Grounded long-context QA with citations or exact spans.
- Natural instruction/Dolly as a baseline, not dominant.

Length bins: short `<256`, medium `512-1k`, long `1k-2k`, and near configured max. Also include common-prefix/crossed-oracle subsets so FP/Q2 trajectory differences are measured, not hidden.

## Final verdict: APPROVE the GO, DOWNGRADE to HOLD (pending C1), or STOP?

**DOWNGRADE to HOLD pending C1.**

Current float32 GO is not defensible. The current F16 deployment path is effectively STOP, but the native-hidden hypothesis should not be killed until C1 is resolved with an algebraic vLLM-vs-ds4 hidden equivalence check and a retained F16 rerun artifact.