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

## Caveats — CORRECTED after codex gate A (2026-07-19, verified)

1. **The Sinkhorn concern was REFUTED** (my original caveat was wrong). The head's `hc_head`
   is a **sigmoid reduce** (`pre = sigmoid(mixes·hc_scale + hc_base)`, then weighted sum —
   `drafter_head.py:78`); the iterative Sinkhorn is body-only (`hc_primitives.hc_split_sinkhorn`).
   So **head.hc_fn's dominance is a real local CE sensitivity** (the 65k-param matrix gates the
   final 4×4096 HC state right before the output norm + lm_head), not a gradient-amplification
   artifact. Independently re-verified.
2. **The metric matters — raw ‖∇W‖ is size-confounded.** RMS grad (‖∇W‖/√n) re-ranks:
   head.hc_fn stays #1 by ~100× (RMS 6.78e-2), but **lm_head is demoted** (RMS 2.51e-5 — it is
   530M params, inflated under raw ‖∇W‖). For LoRA specifically the right metric is the **top-r
   singular energy of ∇W** (the low-rank LoRA captures the top-r directions); RMS is a fallback.
   (Not yet computed — a Phase-1 pre-training refinement.)
3. **lm_head + embed_w are target-loaded** (from the IQ2 target GGUF, shared with the verifier —
   `drafter_body.py:206`, `drafter_head.build_head`), not the drafter's own. Training lm_head has
   a deployment problem (it's the shared output head) → **dropped from the LoRA target set**.
4. **max_step=3 = early/cold-KV steps + CE-vs-KL unvalidated.** The ranking is a probe proxy;
   could shift under soft-label KL + later (warm-KV) steps. The codex recommends re-running with
   KL (once soft labels are re-captured) + a step-bucket comparison (steps 1–3 vs 16/32/64).

## Suggested LoRA-target set (corrected after codex gate A — RMS metric, lm_head dropped)

- **Train directly (small, dominant by raw AND RMS):** `head.hc_fn` (65k params, RMS 6.78e-2 —
  ~100× the next; a real CE sensitivity, confirmed not a Sinkhorn artifact).
- **LoRA (body dense matrices — meaningful RMS grad):** `main_proj` (1.85e-4), the sparse-MLA
  attention projections (`q_a/q_b/kv/output_a/output_b`), the router (`ffn_gate_inp` 2.36e-4),
  the body HC-mixing matrices (`hc_attn_fn`, `hc_ffn_fn`), the shared expert.
- **DROPPED:** `head.lm_head` (target-loaded/shared with the verifier — deployment problem; RMS
  2.51e-5, near-bottom per-param).

**Decisive next step (codex gate A):** run a **head.hc_fn-only ablation now** (the head LoRA
exists from Stage 2) — if held-out p1 improves under soft labels, keep it #1; if it only reduces
training loss or harms held-out, demote the gradient spike and select targets by KL + the
top-r/RMS metric.

## Deferred

The **per-target LoRA ablation** (Phase-0 part 2) requires the training setup (body LoRA — not
yet built; head LoRA exists from Stage 2) + the labels (soft labels missing). Folded into Phase
1's first training runs, which ablate the targets as their first step. The probe (this result)
is the empirical target ranking; the ablation confirms which generalize.
