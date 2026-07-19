## Verdict Per Claim C1-C4

- **C1: questionable.** Commit path is mostly sound: `first` is committed once via `ds4_session_eval()` on bypass, no stale DSpark metric is pushed, and `cycles - bypass_count == dspark_cycle_count`. But the claim overstates byte-exactness: `exclude_eos=true` means `first` may be EOS-masked while margin is unmasked top-logprobs at [ds4_spec_bench.c:1266](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1266) and [ds4_spec_bench.c:1287](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1287). Also lead3 off/on emitted different lengths on 11/60 prompts at θ=2.

- **C2: questionable.** Off baseline is really M3 by metrics: `bypass_count=0`, `scheduled_verify=true`, `verify_decode_ms=0`, and `drafted ~= rows+1` implies anchor reuse. Same prompts/seeds/stack match. But CI is mean per-prompt relative t/s, not aggregate throughput. Warm-weight use is not recorded. Lead3 is contaminated by output-length/EOS differences.

- **C3: sound.** Off-by-one diagnosis holds: current logits predict the anchor; first-draft target margin only exists after processing the anchor, i.e. inside verify. Anchor reuse folds the anchor into batched verify at [ds4.c:29671](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29671), so bypass has no verify to fold into.

- **C4: sound for current P1/P2, overclaimed as “fundamental.”** Recomputed: lead3 θ=2 mean +0.11% CI [-2.08,+2.28], aggregate -0.05%; baseline θ=2 mean +2.65% CI [-1.21,+6.52], aggregate +2.57%. No +3% decision-grade result.

## Premise Sensitivity

- **P1:** STOP survives if M3 baseline and +3% bar stand. If byte-exactness is strict, lead3 measurement is invalid, not positive. If the bar drops to +1%, baseline becomes “marginal,” but still CI includes 0.

- **P2:** STOP is only for the current-logits margin. If a cheap true first-draft target margin existed pre-draft, verdict could flip; code says it does not. Computing it requires target work, erasing the bypass economy.

## The Decisive Test

I ran the artifact-level flipper: paired output-length/EOS check. Result: θ=2 lead3 differs on 11/60 prompts; equal-length subset still only +0.17% mean, CI [-1.98,+2.32]. No flip.

The real flipper would be a clean rerun with token dumps + env/git metadata + θ sweep on the long corpus. Existing artifacts lack margins, so threshold optimization cannot be reconstructed.

## New Experiments To Try

1. **Clean θ sweep, long corpus, alternating off/on per prompt.** Run θ={0.5,1,1.5,2,2.5,3}, token dumps, warm weights logged. Expected: decide whether baseline θ≈2 hides a real >3%. Effort: medium.

2. **Fix/mask EOS consistently in margin.** Use top2 excluding EOS when `exclude_eos=true`. Expected small effect on retained lead3 capture: θ=2 decision diff only ~0.3%, but correctness improves. Effort: low.

3. **Log bypass-cycle timing and margin.** Add explicit bypass metrics instead of residual inference. Expected: confirm bypass cost ~30-33ms, not 26.2ms. Effort: low.

4. **Post-hoc oracle upper bound using true first-draft margin.** Not deployable, but bounds opportunity. Expected: separates “bad signal” from “bad economics.” Effort: medium.

5. **Scale baseline_corpus.** Current n=9 has P(>3%) ≈ 0.43 under bootstrap; too underpowered for “fundamental.” Effort: medium/high.

## Leading Hypothesis

The current bypass is correctly wired as a one-token plain decode, but it loses M3’s anchor folding and uses a weak/off-by-one signal. Measured bypass cost from residuals is ~29.7ms on lead3 θ=2 and ~33.0ms on baseline θ=2. The STOP verdict holds for this lever under P1/P2, but “scheduling axis exhausted” is too strong without a clean long-corpus θ sweep.