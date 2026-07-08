## Verdict Per Claim

1. **Precision gate: questionable.** `torch_precision_gate.json` has **6 prompts, all codealpaca**, not 15. I rederived all 240 stored checkpoints from `drafts + target stream`: **0 internal errors**, but absolute torch-vs-numpy scale is only weakly gated.

2. **`E[a|4]=2.366`, CI `[2.315, 2.419]`: arithmetically sound, headline questionable.** This is the **unweighted prompt mean**. The pooled sliding-cycle value is lower: **2.3448**.

3. **Prompt sd / power: sound.** Recomputed sd **0.40799**, bootstrap CI half-width **0.05205**, `N=256` for ±0.05 and `N=816` for ±0.028. Clustered bootstrap resamples prompts, correctly.

4. **Trajectory claim: likely wrong as stated.** The partition indexing is correct for adjacent sliding positions, but it is **not an actual speculative cycle trajectory**. Simulating K=4 jumps from the same drafts gives **E[a|4]=2.2019**, not 2.345/2.366.

5. **Per-source: sound, but undermines headline.** Recomputed prompt means: codealpaca **2.364**, dolly **2.222**, jsonex **2.512**. Dolly is not positive once full-accept decode penalty is included.

## Headline

**Overclaim.** “K=4 break-even signed positive” is only defensible for the sliding-position / equal Stage-2 mix estimator.

The model cost uses `S4` too, not just `E[a|4]`. With refreshed sliding pooled data: `E=2.3448`, `S4=0.385`, direct speedup is only **1.013x**. With actual K=4 cycle jumps: `E=2.2019`, `S4=0.344`, direct speedup is **0.982x**. That refutes the headline for trajectory-weighted decode.

## Corrected Picture

- Sliding prompt mean: `E[a|4]=2.366`, speed at prompt-mean `E/S4` ≈ **1.018x**.
- Sliding pooled cycles: `E[a|4]=2.345`, speed ≈ **1.013x**.
- K=4 cycle-jump trajectory: `E[a|4]=2.202`, speed ≈ **0.982x**.
- Start-offset sensitivity for cycle-jump: speed ≈ **0.975x to 0.982x**.
- Source pooled speeds: codealpaca **1.016x**, dolly **0.977x**, jsonex **1.045x**.

So the honest result is: **Stage-2 sliding acceptance is mildly positive; trajectory-weighted acceptance is break-even/negative; corpus mix dominates.**

## Capture Recommendation

Do **not** chase `N=816` on the same Stage-2 mix. It will shrink CI around a biased/easy-corpus estimator.

If staying with this corpus, capture to **~256-300** only to clear the ±0.05 floor. Better spend compute on **corpus diversity**: code/synthesis/grounded/deployment-weighted prompts, then report stratified and deployment-weighted cycle-jump speedup.

## Decisive Test

Make the first-class acceptance artifact a **K=4 cycle-jump simulation** from drafts:

`next_step += min(prefix, 4) + 1`, recompute `E[a|4]`, `S4`, and direct speedup, then bootstrap by prompt under deployment corpus weights.

Headline survives only if that lower CI is **> 1.0x**. Current stored data says it is not.