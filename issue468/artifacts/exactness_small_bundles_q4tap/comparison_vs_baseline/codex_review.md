## Verdict Per Claim

- **D1 sound.** Recomputed from raw bundles: `main_hidden` max abs diff `59.4888`, mean `1.09179`, not identical; target tokens differ `33/140`.
- **D2 mostly sound.** Manifests/top-k agree on same prompts, temp `0`, seed `2`, ctx `4096`, 14 tokens, prompt tokens `67..165`; `token_embd.weight` and `output.weight` raw SHA-256 are identical across GGUFs. Spot re-run with same explicit `dspark.gguf` reproduced stored baseline and Q4 rows exactly for `code_histogram`. Caveat: summaries do not persist model/drafter paths.
- **D3 questionable.** Observed p=1 improvement is absent: `65/80 -> 63/80`. But the study is underpowered; paired prompt-bootstrap CI for Q4-base p1 diff is `[-12.5pp, +6.25pp]`.
- **D4 questionable.** `+0.10 E[a|5]` is statistically indistinguishable from zero, but not proved noise. Bootstrap CI is `[-0.1625, +0.4125]`; deeper positions show one-sided small gains.
- **D5 likely-wrong as stated.** It falsifies “this Q4-tap expert-only variant materially helps,” not “tap-localized quant mismatch” in general. P1 leaves lower-layer Q2 corruption unresolved.
- **D6 likely-wrong as stated.** Fine-tuning is plausible, not proven “only viable.” Stage 0 top-2 coverage is current-rank headroom, not a strict fine-tune ceiling.

## Premise Sensitivity

- **P1:** Headline “Q4-tap did not help” survives. Broader falsification does not; if lower-layer Q2 is causal, this experiment would stay flat.
- **P2:** If Stage 0 shallow-rank result fails, the fine-tune argument and `~+10pp` headroom mostly collapse. Stage 1 flat result remains.
- **P3:** If speedup anchors fail, throughput recommendation changes; acceptance measurements do not.

## Statistical Re-Analysis

- **E[a|5block]:** deltas mean `+0.100`, sd `0.496`; paired t `t=0.638`, df `9`, `p=0.539`; Wilcoxon exact `p=0.781`; sign test `4 up / 3 down / 3 flat`, `p=1.0`; bootstrap 95% CI `[-0.1625, +0.4125]`.
- **p=1:** baseline `65/80 = 0.8125`, Q4 `63/80 = 0.7875`, diff `-2.5pp`. Discordants: base-only `7`, Q4-only `5`, exact paired sign/McNemar `p=0.774`. Wilson CIs: base `[0.713, 0.883]`, Q4 `[0.686, 0.863]`.
- **Per-position raw match Q4-base:** p1 `-2.5pp`, p2 `0`, p3 `+3.75pp`, p4 `+5.0pp`, p5 `+8.75pp`.
- **Prefix-survival deltas:** k1 `-2.5pp`, k2 `-1.25pp`, k3 `+2.5pp`, k4 `+5.0pp`, k5 `+6.25pp`. So the `+0.10` really is deeper-position movement; p5 raw signs are `6 up / 0 down / 4 flat`, but rollout-confounded and small-N.

## New Experiments To Try

1. **Full Q8/BF16 target on exact p1 miss set.** Check whether FP/Q8 agrees with drafter or Q2 target. Decisive for quant-vs-calibration. Effort high.
2. **Lower-layer precision sweep.** Build/capture Q4 ranges below 37, especially `1-36` or grouped slices. Signal: p1 jumps only when causal lower range is raised. Effort medium-high.
3. **Crossed hidden/label oracle from existing artifacts.** Q4 hidden + baseline target tokens, baseline hidden + Q4 tokens. Separates input-shift from target-token label shift. Effort low-medium.
4. **Run Stage 0 rank diagnostic on Q4-tap bundles.** Tests whether Q4 misses remain shallow/top-2. Effort medium, no ds4 recapture.
5. **Larger paired corpus and temp `0.5/1.0`.** Current 10 prompts cannot rule out small/moderate effects. Effort high.

## Leading Hypothesis

The data support: **not fixed by raising layers 37-42 expert precision; misses are shallow served-target ordering errors, possibly from lower-layer Q2 shift or drafter calibration.** The one decisive mechanism test is FP/Q8 target top-k on the same p1 miss set.

## Bottom Line

“No observed material Q4-tap effect” survives for this small temp-0 experiment. “Therefore tap-localized quant mismatch is falsified” is overstated. “Fine-tuning is the lever” survives only as a pragmatic next bet, not as an exclusive conclusion; the `~+10pp` number is a rough current top-2 headroom, not a hard fine-tune ceiling.