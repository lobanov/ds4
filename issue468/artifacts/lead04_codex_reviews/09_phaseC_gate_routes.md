## Part 1 gate: AMEND

**AMEND the Phase C verdict.** Key reason: FP float32 acceptance looks real, but “+15%” is not cleanly gated as a deployable IQ2XXS result.

- Capture fidelity: defensible as **likely**, not proven. Exact `v0.24.0` tag has the layer returning `(x, residual, post_mix, res_mix)`, and `mhc_post_tilelang` allocates `out = torch.empty_like(residual)` with `mutates_args=[]`. Good. But current vLLM checkout is not at `v0.24.0`, and this is still code-read fidelity, not vLLM-vs-ds4 algebraic equality.
- Offset: the fixed `drafts[s]` vs `target[s+2:]` cycle-jump alignment is right. The earlier `s+1` result was bogus; the fix is credible.
- `E[a|4]=2.741`: plausible as a strong FP float32 signal. But saved `/tmp/phaseC_fp_f32_full.json` says `n=296`, not 299.
- `+15%`: over-precise. `analyze_phaseC.py` computes `sk` then ignores it, using Q2 `S(4)=0.34`. FP `S(4)` must be measured. With `E=2.741`, baseline still likely clears even with much higher `S`, but the exact speedup should be amended.
- Cross-engine: does not invalidate “native vLLM target is easier for this drafter”; it **does** invalidate “IQ2XXS target can capture this gain” without crossed `H_fp/H_iq2 × Y_fp/Y_iq2`.

Verdict wording I’d accept: **“Native served-precision hiddens expose a large float32 acceptance ceiling; current F16 deployment cannot use it; recoverability on an IQ2XXS target is unproven.”**

## Part 2 routes

| Route | Sound? | Realistic upside | Fatal flaw |
|---|---:|---:|---|
| A. Body LoRA/adapt IQ2XXS drafter toward native teacher | **Partly, only if amended.** Use IQ2XXS labels as primary; native teacher as auxiliary. | Maybe **20-50% of the FP gap** if hidden-side signal exists; enough to beat baseline if it gets only a small slice. | Native teacher may teach the wrong argmax for IQ2XXS. Stage2 head-LoRA hurt does **not doom** body LoRA, but it warns that naive non-expert tuning overfits and that label/order drift dominated prior local tests. |
| B. Full `[4,4096]` HC residual per layer | **Speculative.** Probe first. | Could be large only if HC mean is the bottleneck; otherwise near zero. | The measured FP ceiling already used the **mean** representation, so full-HC is not obviously the missing lever. New `main_proj` is 4x input width; extra draft latency can eat the gain. Requires re-distill. |
| C. Float32 drafter | **Alone: unsound/useless.** | On IQ2XXS hiddens: **0%**; f32-Q2 == f16-Q2. With D: upper-bound helper. | F32 drafter likely doubles memory/bandwidth. If draft cost rises from 10 ms toward ~20 ms, most of the ceiling disappears. |
| D. IQ2XXS→native hidden dequant + f32 drafter | **Most direct scientifically; weak operationally.** | If common-prefix labels align and the map works: maybe **30-70%** of FP gap. Full capture unlikely. | Inverting IQ2XXS hidden loss is ill-posed; pairs may be cross-trajectory; still needs f32 or an F16-stable downstream. Label drift can turn “native-like” into wrong-for-IQ2XXS. |

**Most promising practical pick:** **A, amended**. F16-deployable adapter, trained on IQ2XXS exact labels with native-hidden/native-logit distillation as a regularizer. D is the better diagnostic ceiling, but f32 cost plus ill-posed hidden inversion makes it a bad first product route.

## Part 3 your leads

| Lead | Why it could work | Falsifier |
|---|---|---|
| Crossed FP/IQ2 oracle: `A(D,H_fp,Y_iq2)`, `A(D,H_iq2,Y_fp)`, `A(D,H_fp,Y_fp)` on common prefixes | Separates hidden gain from label drift before training the wrong thing. | `H_fp,Y_iq2` gives little/no lift, or label divergence explains most of the FP win. |
| F16 flip triage: f32 islands for main_proj/head/markov, or margin-triggered f32 rerun | If flips are near-tie final-score numerics, a small mixed-precision patch may recover FP behavior cheaper than full f32. | High-margin flips dominate, or f32 islands recover <90-95% of f32-FP drafts at low overhead. |
| Full-HC probe before Route B | Train a cheap locked-split reranker/probe from full HC vs mean HC. Tests whether mean reduction is actually losing target-token info. | Full-HC probe fails to beat mean probe by a meaningful p1 / `E[a|4]` margin. |
| Re-run K/scheduler with real FP `S(K)` and confidence | FP may change full-accept penalty and optimal K; confidence separation `+0.247` suggests scheduling may outperform fixed K. | Best scheduled replay adds <2-3 speed points over best fixed K, or overhead headroom remains <4 ms/cycle. |