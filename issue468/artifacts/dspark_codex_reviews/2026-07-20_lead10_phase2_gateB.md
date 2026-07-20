## Verdict per claim C1-C4

- **C1: questionable** — I re-derived `Δp1=+0.02295`, CI `[+0.01609,+0.03012]`; but it is conditional on one stochastic training run, no `torch.manual_seed`, and target/hparam selection already used lead3.
- **C2: likely-wrong** — phase2 baseline `0.8440` does **not** match `combined300`: full `p1_mean=0.79334`, pooled `S(1)=0.79002`, and same `0080-0099` IDs average only `~0.7945`.
- **C3: questionable** — F16 “PASS” is aggregate p1 equality, not row equality; retained F16 review reports F32 `0.850084` vs F16 `0.849523`, top-1 equal `1777/1781`, no live ds4 proof.
- **C4: questionable** — cached p1 arithmetic clears the rule, but the deployment verdict does not: baseline mismatch + F16 gap + no `E[a|4]` + head-only scope.

## The baseline question

Not just “lead3 is an easy subset.” Worse: the phase2 cache appears to be a **different eval regime/trajectory**. Same prompt IDs disagree hard against `combined300` (`codealpaca_0080`: `0.852` vs `0.762`; `dolly_0090`: `0.680` vs `0.852`). Mean phase2-minus-combined on the same 60 IDs is `+4.95pp`, abs mean `8.32pp`.

So `0.8440 vs 0.79` is not reconciled by “different corpus mean.” The available artifacts do not support that claim.

## The F16 + E[a|4] gaps

They undermine **deployment PROCEED**, not the existence of a p1 signal.

The cache has only first-target data, so `E[a|4]` / block acceptance is unmeasured. Since runtime value depends on block acceptance, p1 alone is insufficient for integration confidence.

## The verdict

**MARGINAL / run SC7 first.**

The in-cache p1 lift is real-looking: anchor-weighted delta is also positive, `+2.42pp`, CI `[+1.83,+3.06]`. But the decision-grade claim is over-scoped. This is a promising head-only ablation on a questionable cached eval, not a validated drafter re-distillation.

## Leading hypothesis + decisive test

Hypothesis: `head.hc_fn` learned a real first-token correction on the cached IQ2 trajectory, but the cached trajectory/baseline is not the deployed `0.79` regime.

Decisive test: bake/export the trained head LoRA into dspark, run live ds4/Metal on the locked `combined300` or fresh distill eval, and report row-matched p1 **plus `E[a|4]`**, F32/F16/ds4 agreement, per source.