# ds4-eval Quality Gate — DSpark 64/92 (1 short of ≥65 gate)

Date: 2026-07-01. Seventeenth productionization handoff note. Records the
ds4-eval 92-case quality run with --dspark. Doc-only research record.

## 1. Result

Command (per the contract):
```
./ds4-eval -m MODEL --dspark dspark.gguf --plain --nothink -n 4096 --seed 1 --questions 92 --temp 1.0
```

Final report:
```
ds4-eval: 64/92 passed, 28 failed, runtime 01h:17m
```
Config: context auto-sized to 4873 tokens; DSpark drafter loaded (F32 KV default,
all fixes: Bug#1 + cap + Bug#3 + Bug#4 + FP8-off). Output was coherent throughout
(no garble, no crashes, no B2 exactness violations visible).

## 2. Gate status: MISS by 1 case

The quality gate is ≥65/92. DSpark scored **64/92 (69.6%)** — exactly ONE case
short. This is within temp=1.0 sampling noise (a different seed could flip 1-2
cases), but the contract specifies seed=1, so 64/92 is the measured result.

NOT marking quality-eval complete — it does not meet the ≥65 gate.

## 3. The key question: is 64 a DSpark regression or the model's baseline?

A plain-decode comparison run (same config, no --dspark) is in progress to
answer this. Two outcomes:
- **Plain also scores ~64-65**: then DSpark is quality-NEUTRAL (B2 doesn't
  degrade quality; the absolute gate is just set at/below the model's natural
  score, and 64 is a sampling-variance near-miss, not a DSpark defect). The
  quality gate could be re-run (different seed) or the threshold revisited.
- **Plain scores notably higher (e.g. 70+)**: then DSpark's B2 is introducing a
  quality regression (the argmax(p-q) correction + stochastic accept at temp=1.0
  diverging from the target distribution) — a real bug to fix.

## 4. Honest framing

- 64/92 is a near-miss on an absolute threshold. The PRACTICAL quality question
  (does DSpark degrade output?) is separate and answered by the plain comparison.
- B2 speculative sampling at temp=1.0 SHOULD be statistically equivalent to plain
  decode (that's the lossless guarantee of rejection sampling). A large plain-vs-
  dspark delta would indicate a B2-correctness bug; a small delta is expected
  sampling variance (different RNG paths produce different specific samples).
- This is the FIRST quality measurement of the productionized DSpark (all prior
  work was perf-focused). 64/92 with no garble/crash is itself evidence the
  productionization is functionally sound.

## 5. Decision pending

Awaiting the plain-decode comparison (~1h). If plain ≈ 64-65, the near-miss is
sampling variance, not a DSpark defect — will surface for a gate/seed decision.
If plain >> 64, it's a B2 quality bug to investigate.
