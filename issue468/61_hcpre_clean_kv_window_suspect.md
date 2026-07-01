# Op-Level Bisection — hc_pre CLEAN; KV Window Accumulation Is the Prime Suspect

Date: 2026-07-01. Twenty-first productionization handoff note. Corrects the
methodological errors of issue468/60 and identifies the true divergence path.
Doc-only research record.

## 1. CORRECTION: hc_pre is CLEAN on identical input (cos 1.00000)

The IDENTICAL-INPUT test (feed Metal's actual layer-1 output into BOTH Metal's
and the oracle's hc_pre + attn_norm, compare outputs):

| step | layer 1 cos | layer 2 cos |
|---|---|---|
| 1 | 1.00000 | 1.00000 |
| 2 | 1.00000 | 1.00000 |
| 3 | 1.00000 | 1.00000 |

hc_pre + attn_norm is BIT-PERFECT on identical input for ALL layers including
layer 2. The 11.88% "at layer 2's hc_pre" (issue468/60) was a methodological
artifact: comparing Metal's output (on Metal's input) to the oracle's output
(on the oracle's OWN different input). TWO different inputs, not identical.

## 2. The attention sub-block DOES diverge — but KV window state differs

Identical-input test of the FULL attention sub-block (hc_pre + attn_norm + q/kv
matmuls + sparse_attn + output_proj + hc_post), feeding Metal's layer-0 output
into the oracle's layer-1 attention, comparing the mid-block output:

| step | L1 attn_mid rel% | L1 attn_mid cos | L2 attn_mid rel% | L2 attn_mid cos |
|---|---|---|---|---|
| 1 | 1.45% | 0.99966 | 9.98% | 0.98432 |
| 2 | 3.12% | 0.99955 | 12.54% | 0.97387 |
| 3 | 4.05% | 0.99822 | 19.34% | 0.98098 |

The attention sub-block diverges even on identical block-input — BUT this is NOT
a truly identical test because the **persistent KV window** (dspark_kv_cache)
differs between Metal and the oracle. The attention output depends on BOTH the
input AND the KV window state. The oracle accumulates its KV window from the
oracle's forward; Metal accumulates from Metal's forward. They diverge because
prior steps' KV computations differed slightly (F16 vs F32 in the attn_kv
Q8_0 matmul).

## 3. The growing divergence pattern confirms KV accumulation

The divergence GROWS with step number (L2 attn_mid: 10% → 12.5% → 19.3% from
step 1→3). This is consistent with **persistent KV window accumulation**: each
step, the anchor KV (computed from main_x via attn_kv Q8_0 matmul) diverges
slightly between Metal and the oracle. These small per-step divergences
accumulate in the window, amplifying through the attention softmax.

## 4. Prime suspect: attn_kv Q8_0 matmul precision

The anchor KV is computed by `ds4_gpu_matmul_q8_0_tensor(..., attn_kv->abs_offset,
..., dspark_main_x, 1)` (matvec, n_tok=1). This Q8_0 matmul uses F16-input tiles
(the same half-input MMA pattern as q_b). The oracle dequantizes attn_kv to F32
and matmuls in F32.

The anchor KV is PERSISTENT (stored in dspark_kv_cache at slot [n_real], grows
across steps). A small per-step KV divergence (from F16 dequant of attn_kv)
accumulates in the window, and by layer 2 at step 3, the accumulated divergence
amplifies through the softmax to 10-20%.

NOTE: the F32-input MMA test (issue468/60) only tested q_a + q_b, NOT attn_kv.
The attn_kv matmul was NEVER routed to F32-input. The hypothesis is testable:
route attn_kv to F32-input and see if the attention divergence drops.

## 5. What's been ruled out (definitively)

- hc_pre + attn_norm: CLEAN (identical-input, cos 1.0)
- q_a + q_b: CLEAN for layers 0/1 (F32-input test, cos 1.0; F16 path already correct)
- F32-input MMA for q_a/q_b: no change to divergence (q matmuls not the source)

## 6. Next test: KV window comparison

Dump Metal's dspark_kv_cache[1] (persistent window for layer 1) after step 1
and compare to the oracle's win_kv[1] state. If they differ → KV precision is
the root cause → route attn_kv to F32-input (batch path) + fix the matvec path
→ re-measure. If they match → the divergence is in sparse_attn or output_proj
itself (a non-KV attention bug).
