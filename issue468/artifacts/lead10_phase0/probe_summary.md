# Lead 10 Phase-0 — gradient probe result (2026-07-19)

The gradient-magnitude ranking of the dense LoRA candidates, to pick the targets empirically.

## Setup

- **Capture:** Lead-11 unified capture (`lead3_h.bin`, 60 lead3 prompts). H_iq2 + Y_iq2 (sel =
  the `tok` field). **Soft labels missing** (dropped in Lead-11's Exp-0a consolidation) → the
  probe uses **hard-label CE vs sel** (a valid proxy for the gradient *ranking*; both CE and KL
  push toward the IQ2 distribution). KL training (Phase 1) needs the soft labels re-captured.
- **Precision / memory:** **F32** (F16 backward NaN'd through the MoE/rmsnorm). **max_step=3**
  per prompt — the MoE's expert-gather activation retention is ~9 GB/step (F32); max_step=3 keeps
  the peak ≈ 104 GB (experts 77 GB + retention 27 GB) within the 182 GB MPS limit. 60 prompts ×
  3 steps = 180 anchors; the `.grad` is the mean over prompts.
- **Experts frozen** (verified: `exp_gate.grad is None`). Embed/cos/sin frozen. Grad enabled on
  all dense body + head weights.

## Result (`gradient_probe_result.json`)

The ‖∇W‖/‖W‖ ratio **inflates the tiny HC-scale weights** (‖W‖≈0.06 → ratios of 70–320) — a
known artifact; those are scalar/vector weights (train directly or freeze, **not** LoRA targets).
The meaningful ranking is by **raw ‖∇W‖** (the LoRA-potential metric — loss reduction ∝ ‖∇W‖²),
**matrices only**:

| rank | target | raw ‖∇W‖ | notes |
|---|---|---|---|
| 1 | **head.hc_fn** | **17.36** | the head's HC-reduction [4,16384]; 65k params — train directly, not LoRA |
| 2 | main_proj | 1.31 | the IQ2-hidden input projection |
| 3 | head.lm_head | 0.58 | the output projection (530M params) |
| 4 | attn (q_a/q_b/kv/output_a/output_b) | 0.55 mean | the sparse-MLA attention |
| 5 | shared_expert | 0.39 | the always-active shared expert |
| 6 | router (ffn_gate_inp) | 0.21 | the MoE router |
| 7–8 | hc_attn_fn / hc_ffn_fn (body) | 0.19 / 0.18 | the body's HC-mixing matrices — LOW |

## The finding

**The IQ2-mismatch gradient concentrates at the OUTPUT side (head.hc_fn, lm_head) + the INPUT
projection (main_proj) + the attention — NOT in the body's HC mixing.** head.hc_fn dominates by
10×. This is genuinely informative (not obvious a priori — Lead 07 showed the drafter is
hidden-input-invariant at the *acceptance* level; the *distribution* mismatch localizing to the
output + input projections is a new signal).

## Caveats (for codex gate A)

1. **head.hc_fn's dominance may be partly a Sinkhorn-iteration gradient-amplification artifact**
   (the hc_head's iterative normalization can amplify the gradient through the mixing weights).
   The ablation (Phase 1) must confirm head.hc_fn actually generalizes, not just that its
   gradient is numerically large.
2. **CE-vs-KL + max_step=3 (early steps, cold KV window):** the ranking is a probe proxy; the
   codex gate should assess whether the ordering is robust to the loss choice + the step range.
3. **The tiny high-ratio weights (hc_*_scale, norms):** excluded from the LoRA candidate set —
   train directly or freeze.

## Suggested LoRA-target set (subject to the ablation)

- **Train directly (small, high-grad):** head.hc_fn.
- **LoRA (matrices):** main_proj, the attention projections (q_a/q_b/kv/output_a/output_b),
  head.lm_head, the shared expert. (Body hc_attn_fn/hc_ffn_fn — low priority.)

## Deferred

The **per-target LoRA ablation** (Phase-0 part 2) requires the training setup (body LoRA — not
yet built; head LoRA exists from Stage 2) + the labels (soft labels missing). Folded into Phase
1's first training runs, which ablate the targets as their first step. The probe (this result)
is the empirical target ranking; the ablation confirms which generalize.
