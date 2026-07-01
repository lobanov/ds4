# Drafter Drift Bisection — Layer-2 q_b (Q8_0) Localization

Date: 2026-07-01. Fourteenth productionization handoff note. Continues the
mtp.2-attention bisection (issue468/53, 54). Localizes the layer-2 divergence
to the q_b matmul / its Q8_0 weights. Doc-only research record.

## 0. Where we were

issue468/54 proved the mtp.2 divergence is a genuine layer-2-specific bug
(propagation refuted; same input → divergent output; cosine 0.975). This note
bisects WITHIN layer 2's attention to find the first diverging operation.

## 1. Level 6: MHSA 'o' (batch_heads) — diverges for layer 2 (NOT the output proj)

Added DS4_DSPARK_PROBE_DUMP_MHSA (dumps batch_heads post-sync in the probe
loop, avoiding the mid-command-buffer sync problem of the older DUMP_HEADS
hook). Compared to the oracle's pre-output-projection 'o' ([5,64,512]):

| step | lay0 o | lay1 o | lay2 o |
|---|---|---|---|
| 1 | 0.96% cos 0.99998 | 1.50% cos 0.99996 | 8.99% cos 0.99818 |
| 2 | 0.77% cos 0.99999 | 1.12% cos 0.99999 | 10.81% cos 0.99779 |
| 3 | 0.62% cos 1.00000 | 0.48% cos 1.00000 | 9.15% cos 0.99422 |

The MHSA output `o` itself diverges 9-11% for layer 2 (cosine 0.994-0.998).
Layers 0,1 essentially perfect. → the bug is INSIDE the MHSA, not in the
grouped-low-rank output projection (which is clean given identical input
cosine ~1.0 for layers 0,1).

## 2. Level 7: query q (batch_q) — diverges 4-6% for layer 2

Added DS4_DSPARK_PROBE_DUMP_QK (dumps batch_q post q_b+head_rms+rope, and
batch_kv). The kv dump is UNRELIABLE (batch_kv is overwritten by the anchor
KV after the draft KV; reads anchor KV, not draft KV — shows false divergence
for ALL layers). The q dump is reliable:

| step | lay0 q | lay1 q | lay2 q |
|---|---|---|---|
| 1 | 0.00% cos 1.0 | 0.48% cos 0.99999 | 4.14% cos 0.99858 |
| 2 | 0.00% cos 1.0 | 0.18% cos 1.0 | 4.22% cos 0.99863 |
| 3 | 0.00% cos 1.0 | 0.12% cos 1.0 | 5.84% cos 0.99543 |

(p99 diff on |q|~15: layers 0/1 < 0.02; layer 2 = 0.16-0.29.)

**The query q diverges 4-6% for layer 2 only.** Since q is the input to the
softmax (scores = q·kv), a 4-6% q divergence is consistent with — and likely
the cause of — the 9-11% `o` divergence (softmax amplifies: a small q
perturbation shifts which KV slots win attention).

## 3. Level 8: q_a-vs-q_b — divergence enters at q_b (the large Q8_0 matmul)

The q pipeline is q_a→q_a_norm→q_b→head_rms→rope. Dumped batch_qr (q_a output,
pre q_b) — but it's UNRELIABLE too (batch_qr is reused downstream; reads stale
data showing false divergence for all layers, 247-394% rel). The trustworthy
signal is batch_q (post q_b): PERFECT for layers 0,1 (cosine 1.0), diverges
4-6% only for layer 2.

Reasoning: q_b is the largest attention weight ([1024, 32768] = 32M elements,
Q8_0). All three stages share the identical matmul kernel + Q8_0 dequant.
Layers 0,1 produce bit-perfect q (cosine 1.0); only layer 2's q_b output
diverges. With identical code + identical quant types + identical block
scales (issue468/54 §3), the divergence must come from layer-2-SPECIFIC Q8_0
**int8 weight values** where the GPU's dequant/reduction-order diverges from
numpy's reference dequant — OR a per-layer buffer/state bug surfaced only for
the last stage.

## 4. Strong, testable hypothesis: mtp.2 q_b Q8_0 dequant divergence

The full localization chain (8 levels):
forward (not Markov) → blocks (not head) → layer 2 mtp.2 (not 0/1) → attention
(not MoE) → MHSA (not output proj) → query q (not kv: kv dump unreliable) →
q_b (q_a dump unreliable; q_b output is the first reliable diverging tensor).

→ **mtp.2's q_b Q8_0 dequant (or the q_b matmul accumulation) diverges from the
numpy reference.** Recovering this is the gate-crossing fix (issue468/42).

## 5. Next decisive test (no new C code needed)

The metal q_b weights are the Q8_0 bytes at the mtp.2 offset (issue468/53 §7:
7.6 GB). Compare Metal's dequantized q_b to the oracle's `_dequant_q8_0` output
directly:
- If they differ → GPU Q8_0 dequant kernel diverges for mtp.2's specific int8
  values (a dequant-kernel bug or an accumulation-order issue) → fix the dequant.
- If they match → the bug is the q_b matmul accumulation itself (Q8_0 × F32 on
  GPU using a different reduction than numpy's @) → either re-quant q_b to F16/Q8
  (out of scope: drafter frozen) OR fix the matmul kernel's accumulation.

Both are fixable in-scope (code-only, drafter byte-identical) since they target
the GPU KERNEL, not the drafter weights.

## 6. Significance

8 levels deep, the bug is localized to mtp.2's q_b — a single Q8_0 matmul on a
single tensor. This is as tight as a bisection gets. The remaining work is one
direct dequant comparison + a kernel/dequant fix, then re-measure the gate.
