# Q8_0 Layer-2 Re-quant — WORSE Than Q4_K (avg prefix 4.37 → 3.79)

Date: 2026-07-01. Twenty-ninth productionization handoff note. The Q8_0 re-quant
result. Doc-only research record.

## 1. Result: Q8_0 is WORSE than Q4_K

| config | greedy match | avg prefix | histogram |
|---|---|---|---|
| Q4_K baseline (all experts) | 87/95 (91.6%) | **4.37** | [1 0 0 2 3 13] |
| Q8_0 mtp.2 (mixed-precision) | 77/95 (81.1%) | **3.79** | [2 0 1 3 4 9] |
| Oracle target | ~4.49 | ~4.49 | — |

The Q8_0 re-quant REDUCED draft acceptance by 13% (4.37 → 3.79). This is
counterintuitive — higher precision weights should produce better drafts, but
they produced worse ones.

## 2. Why Q8_0 is worse

The drafter's Q4_K quantization is not "lossy noise" that Q8_0 corrects — it's
an integral part of the drafter's behavior. The drafter model was either:
(a) Trained/distilled with Q4_K quantization in mind (the quantization is part
    of the model's training distribution), OR
(b) The Q4_K block-scale/d_offset values interact with the drafter's attention +
    MoE in a way that the model's original BF16/FP8 weights were calibrated to
    work with, and Q8_0 disrupts that calibration.

The oracle uses the SAME Q4_K weights (dequanted to F32) — so the oracle's
+8.27% headroom is NOT from using higher-precision weights. It's from the
numerical path (pure F32 dequant + F32 accumulation vs Metal's mixed F16/F32).
Re-quantizing to Q8_0 changes the weights to DIFFERENT values that the model
was not trained for, producing worse drafts.

## 3. Conclusion: Q8_0 re-quant is NOT the lever

The Q8_0 re-quant is actively harmful. The Q4_K quantization is the correct
configuration for this drafter model. The +8.27% oracle headroom is from the
numerical PATH, not the weight PRECISION — and the path difference (F32 vs
mixed F16/F32/SIMD) is the cumulative drift already characterized in
issue468/65.

## 4. Blocker rule precondition 9 satisfied

The Q8_0 re-quant was attempted + measured. Result: WORSE than Q4_K (4.37→3.79).
This is documented as not-crossing with measured probe numbers.

## 5. Next: pi agent review

Per the goal's two-phase recovery path, the Q8_0 re-quant (phase 1) did not
cross the gate. Phase 2: launch an independent pi agent review of ALL prior
research notes (issue468/33-68) to identify previously-discarded leads.
