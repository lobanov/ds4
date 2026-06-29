# DSpark research — review note: revisiting Outcome B, and the next three goals

Date: 2026-06-29. Author: review pass over Phases 0–6 (issue468/00–23) plus the
HuggingFace source checkpoint and the ds4 Metal kernels.

Purpose: assess whether the terminal "UNACHIEVABLE (Outcome B)" verdict
(issue468/23) is actually terminal, record the observations that emerged from a
fresh read of the artifacts and code, and define three concrete, sequenced goals
for the next research cycle.

**Bottom line up front:** the verdict disproved *one configuration* — the
BF16-accumulating fused Q4_K routed kernel, scored on greedy-match acceptance —
not the design space. At least three links in the chain from "Q4_K is the ceiling"
to "UNACHIEVABLE" were asserted rather than measured, and the most important one
(higher drafter precision) was the verdict's *own* nominated fix, filed
out-of-scope on a memory premise that turns out to be wrong. The project should
**not** be closed at Outcome B without the validation work in Goal 1.

---

## 1. Progress made (validated, reusable)

The negative verdict notwithstanding, the project produced substantial, correct
infrastructure:

- **Conversion (Phase 3).** `dspark.gguf` — 10.71 GiB, 81 tensors, 47/47 byte-exact
  crosscheck against the source. Residency planned and measured.
- **Metal drafter forward (Phase 4).** Complete: input stage (corr 1.0 vs oracle),
  attention sub-block (corr 0.99997), 3 DSparkBlocks, output head, sequential
  Markov head — runs end-to-end and emits draft tokens. Several real
  converter/kernel bugs found and fixed (F16 hc_fn, F32 norms, inverse-rope,
  Metal map registration, swiglu_limit clamp). A non-causal batched attention
  primitive was added.
- **numpy oracle.** A validated, algorithm-faithful reference forward
  (`issue468/dspark_oracle/`). This is now load-bearing: see Goal 3.
- **Measurement harness.** Env-gated probes for backbone timing, input/attn/block/
  token correctness, and a persistent-KV greedy-acceptance sweep. All reusable.
- **Cost curves (all measured on M5 Max / Metal).** Plain decode 31.3→35.5 ms/tok
  over 7k→64k (+11%, SWA-flat); batch verify(L=5) ≈ 75 ms (context-flat); drafter
  backbone (3 layers, 5 tok) 5.0 ms; total draft ≈ 7.2 ms.
- **Headline acceptance numbers.** Oracle greedy: 2.79 avg accepted prefix (57.9%).
  Metal greedy (19-step persistent-KV sweep): 1.53 (33.7%). The gap between these
  two is the entire basis of Outcome B.

## 2. Important observations (from this review)

These are corrections and gaps surfaced by re-reading the artifacts against the
code and the source checkpoint.

1. **"Q4_K is the precision ceiling" is true only for the *fused routed* kernel —
   not for the drafter.** `ds4_gpu_routed_mm_pipeline` (ds4_metal.m:20079) is
   genuinely capped at IQ2_XXS/Q2_K/Q4_K. But ds4 has dense F16 and Q8_0 matmul
   kernels (`ds4_gpu_matmul_f16_tensor` 13098, `ds4_gpu_matmul_q8_0_tensor` 12906)
   that are *already used and validated* in the drafter's own attn/input stages and
   in the main model's shared experts. The drafter is tiny (≈90 active expert ops
   per cycle), so its MoE can run through a gathered dense path instead of the
   fused routed kernel.

2. **The 2.79→1.53 collapse is attributed to *accumulation* precision, not weight
   quantization.** The oracle used the *same Q4_K weights*, dequantized to F32, and
   got 2.79; the Metal fused kernel uses BF16 accumulation and got 1.53 (FFN corr
   0.979/block, error/signal corr 0.15 = noise). So the fix is **memory-neutral**:
   an F32-accumulating gathered dense matmul over the *unchanged* Q4_K weights
   should reproduce the oracle's 2.79. No fatter weights required. (Two distinct
   error sources: **Source A** = Q4_K weight quant, already inside the oracle's
   2.79; **Source B** = BF16 accumulation, the dominant Metal-vs-oracle gap.)

3. **B2 rejection sampling — the protocol the project actually selected (Phase 2a)
   — was never measured.** Every acceptance number is greedy-argmax-match; the
   verdict's `committed ≈ greedy_prefix + ~1 (B2 bonus)` is a projection. B2 can
   accept non-argmax tokens, so true B2 accepted length is a different, plausibly
   higher quantity. The terminal gate was decided on a number that was never run.

4. **The single-step vs multi-step acceptance gap was never isolated.** Doc 21
   reports single-step (n_real=1) Metal acceptance ≈ "prefix 2–3, comparable to the
   oracle's 3," yet the 19-step persistent-KV sweep collapses to 1.53. Doc 16
   explicitly flags the window-KV ring handling as unvalidated for growing n_real.
   The collapse may be partly a KV-threading artifact, not pure precision — this
   was never cross-checked against a multi-step oracle on identical state.

5. **The 64k memory picture is internally contradictory, and the pessimistic number
   is the wrong one.** Doc 19 says "~7 GB free at 64k" (derived as observed-18 −
   10.71, not itemized). The Phase 3 residency check
   (`baseline/dspark_residency_check.md`) itemizes ~87.8 GiB fixed + 10.71 drafter
   = 98.5 GiB → **~29.5 GiB headroom**. SWA keeps 64k context at only 1394 MiB, so
   the ~22 GiB discrepancy cannot be context growth; doc 19's figure likely counts
   mmap page cache / other processes as "used." **Real headroom is ~29 GiB.**
   Consequence: Q8_0 (+9 GiB) and Q6_K (+3.8 GiB) drafter storage *fit
   comfortably*; only F16 (+26 GiB) breaks the budget. (This corrects an earlier
   claim, made on doc 19's bad number, that Q8_0 storage wouldn't fit.)

6. **Source precision is FP8, not fp4 (verified against the checkpoint).** HF
   `deepseek-ai/DeepSeek-V4-Flash-DSpark/config.json`:
   `quant_method=fp8, fmt=e4m3, scale_fmt=ue8m0, weight_block_size=[128,128]`,
   torch_dtype bfloat16. So the routed experts are natively **FP8 e4m3
   (~8 bits/weight, coarse 128×128 block scales)**. This corrects doc 09's
   "FP8 e4m3 / fp4." Implications: (a) there *is* real headroom above Q4_K (Q4_K
   discards FP8 mantissa); (b) Q8_0 (8.5 bpw, finer per-32 scales) ≈ reproduces the
   FP8 ceiling losslessly, so "Q8_0 + imatrix" ≈ ceiling and imatrix adds ~nothing
   at Q8; (c) imatrix's value is concentrated at Q4_K (its sweet spot). The real
   contest is **Q4_K+imatrix vs the FP8 ceiling**, with Q6_K+imatrix as the middle
   point — and the outcome is not predictable a priori.

7. **The drafter is plain Q4_K; the target is imatrix-quantized.** A re-quant of the
   drafter experts with an importance matrix is memory- and kernel-neutral at Q4_K
   (same tensors, same routed kernel reads them). But the imatrix must be collected
   from the drafter's *real* activation distribution (target L40/41/42 hidden
   states → main_proj → DSpark backbone, autoregressive within the noise block) —
   **not** by running text through a standalone LM, and **not** by reusing the
   target's imatrix. MoE sparsity (top-6 of 256) means stable per-expert statistics
   need on the order of millions of decode-step activations.

8. **ds4 has a real SSD expert-streaming layer** (`ds4_ssd.h`, mmap +
   `g_stream_expert_cache_*` with pread accounting). The "streaming" in the cost
   model is unified-memory bandwidth (~800 GB/s), and the measured per-layer costs
   confirm the hot experts are RAM-resident. But this is a cliff: any change that
   pushes the working set past the resident cache budget falls to SSD pread
   (~100× slower) — another reason to prefer the memory-neutral precision path and
   to confirm the true 64k residency (Goal 1c).

## 3. Lever analysis (where the headroom is)

Per-cycle economics: `ms/tok = (draft + verify) / committed`, draft 7.2,
verify 75, plain 35.5 @64k. Speculative wins iff `(draft+verify) < 35.5 ×
committed`. From this:

- **Acceptance is the #1 lever** (elasticity 1.0 — it amortizes the entire fixed
  draft+verify cost — and higher acceptance *unlocks larger γ*, which is cheaper
  per token via verify's sub-linearity). Levers, cheapest first:
  - **F32 accumulation over Q4_K** — closes Source B, memory-neutral, no GGUF
    rebuild. Expected 1.53 → ~2.79.
  - **imatrix at Q4_K** — shrinks Source A, memory/kernel-neutral. Lifts the
    ceiling; may punch above its weight because the sequential head amplifies small
    per-step gains.
  - **higher quant (Q6_K/Q8_0)** — only if the Pareto sweep (Goal 3) shows the gain
    clears a pre-set threshold; needs a new routed kernel (Tier 1). Q8_0 ≈ ceiling.
  - **target-distillation fine-tuning** — highest ceiling (corrects the
    FP-target→2-bit-target train/serve mismatch), exactness-safe (drafter is only a
    proposal; B2 preserves correctness), but costliest (needs a differentiable
    drafter forward + GGUF round-trip). Stacks on all of the above.
- **Verifier speed is the #2 lever — a margin-widener, not a standalone rescue.**
  At acceptance 1.53 it would need a ~2× cut to clear the gate, implausible for a
  bandwidth-bound MoE; realistic 10–30% gains matter only *after* acceptance is
  restored (then they turn the borderline 32k/55k cases into comfortable passes,
  and 64k from +21% to +50%+). Do it after acceptance, not before.
- **Non-sequential draft head** — the architectural fallback. Pursue only if
  precision + fine-tuning fail to recover acceptance, which would itself prove the
  sequential head (not quantization) is the limiter.

Key enabler for cheap iteration: **under the F32-accumulation premise the oracle
*is* the production model**, so the entire quant × imatrix Pareto frontier is
measurable offline in numpy with zero new Metal kernels. Build a Tier-1 kernel
only for the winner, only if the offline numbers justify it.

## 4. The next three goals

Sequenced; each has a deliverable and an exit condition. Goal 1 gates the rest.

### Goal 1 — Re-establish the true acceptance baseline (measurement integrity)

No kernel work. Before trusting "1.53," fix how it was measured:
- **1a. Measure true B2 rejection-sampling accepted length** on the existing Metal
  forward (replace greedy-argmax-match with the actual u < p_target/p_draft
  protocol at temp=1.0). This is the number the terminal gate needed.
- **1b. Reconcile single-step vs 19-step.** Run the identical persistent-KV sweep
  through the numpy oracle with the same KV bookkeeping, to isolate whether the
  1.53 collapse is precision (Source B) or the unvalidated window-KV ring path
  (doc 16). Per-step, per-position breakdown for both.
- **1c. Itemize real RAM at 64k** (RSS + mmap page cache + GPU buffers + any
  streaming-cache mlock budget) and reconcile doc 19 (7 GiB) vs Phase 3 (29.5 GiB).

*Deliverable:* corrected acceptance figures (greedy + B2) and a confirmed 64k
headroom number. *Exit:* if B2 and/or the KV fix already lift acceptance
materially, the verdict flips on measurement alone — proceed to Goal 2 to bank it;
if 1.53 survives all three checks, the precision story (Goal 2) is on firm ground.

### Goal 2 — Memory-neutral F32-accumulation drafter MoE

Implement a gathered dense expert matmul (reuse `ds4_gpu_matmul_f16_tensor` / an
F32-accumulating path) over the **existing Q4_K** drafter weights, for the ~90
active expert ops per cycle. Storage unchanged (+0 RAM).
- Cross-check the gathered-dense path reproduces the oracle (corr ~1.0) on captured
  positions (closes Source B by construction).
- Re-measure end-to-end acceptance + speedup at 32k / 55k / 64k, with the B2
  protocol from Goal 1.

*Deliverable:* measured end-to-end speedup with Source B eliminated. *Exit:* if
acceptance recovers toward ~2.79 and the gate passes at ≥55k, the project flips to
**Outcome A** — proceed to Goal 3 to find the cheapest precision recipe and to γ /
verifier tuning. If acceptance stays near 1.53 even with confirmed-correct F32
accumulation and KV, that is strong evidence the *sequential head*, not
quantization, is the limiter → escalate to the non-sequential-head redesign.

### Goal 3 — Capture pipeline + offline Pareto sweep (precision recipe + imatrix)

Build the target-anchor capture harness once (target L40/41/42 hidden states +
generated/teacher-forced continuations over `gguf-tools/imatrix/dataset`), then use
it for everything:
- **3a. Collect a drafter-correct imatrix** (hook the routed-expert inputs in the
  oracle/Metal draft forward over the captured anchors; weight the continuation
  region; verify per-expert coverage given top-6/256 sparsity).
- **3b. Run the offline oracle acceptance sweep** across
  {Q4_K, Q4_K+imat, Q6_K+imat, Q8_0, FP8-source-ceiling}, ranking on **acceptance /
  end-to-end speedup** (not on perplexity/correlation — the sequential head breaks
  that proxy). Map the frontier vs bits/weight, RAM, and *kernel-engineering effort*
  (Tier 0 = Q4_K±imat, no kernel; Tier 1 = Q5/Q6/Q8, new routed kernel). Optional:
  mixed gate/down precision — the routed kernel already dispatches `gate_type` and
  `down_type` separately (ds4_metal.m:25041).

*Deliverable:* the acceptance-vs-precision Pareto frontier and a go/no-go, against a
pre-set threshold (e.g. Δspeedup ≥ 8%), on whether any recipe beyond Q4_K+imatrix
justifies a new kernel. *Exit:* ship the cheapest recipe on the frontier knee; the
same capture pipeline then seeds target-distillation fine-tuning if the frontier
shows the FP8 ceiling itself is short of the gate.

---

## 5. What would change the verdict

Outcome B stands only if, after Goal 1 (correct B2 + KV + memory) and Goal 2
(confirmed F32 accumulation over Q4_K), acceptance remains far below the ~2.5–2.8
the long-context gate needs. Until those are measured, "UNACHIEVABLE" is premature:
it rests on greedy accounting of a single quantization+accumulation configuration,
on an unvalidated multi-step KV path, and on a memory constraint contradicted by
the project's own residency check. The economics have real margin — verify (75 ms)
dominates, so even a 2–3× more expensive drafter wins once acceptance is restored —
and every lever above is exactness-safe because the B2 verifier preserves the output
guarantee regardless of draft quality.
