# Phase 3+ Plan — DSpark official-tensor integration on Branch B (B2)

Status: **plan, not started.** Formed after the Phase-2a quality baseline
(`08_phase2_quality_results.md`) confirmed B2. Goal (per user): **prove DSpark
in ds4 using the official tensors from
`deepseek-ai/DeepSeek-V4-Flash-DSpark`, to empirically obtain acceptance rate
and speedup.**

This supersedes the old Phase 3 (generic loader recon) and Phase 4 (draft-only
prototype) with a concrete, tensor-grounded path.

## 0. Why this path is now justified (chain of evidence)

1. Phase 1 (`06`): only the **batch** verifier is sub-linear; every exact path
   is a loss. The batch verifier is the load-bearing curve.
2. Phase 2a (`08`): greedy-argmax on the batch verifier is a **reproducible
   −6-case quality regression** (61/92 vs 67/92; exact reproduces 67/92). The
   drift is real and harmful under greedy.
3. `07` §3 + DSpark `generation_config.json`: the paper's protocol is
   **rejection sampling at temp=1.0**, under which batch drift is benign by
   construction (smooth in the logits).
4. DSpark repo: the drafter ships, but **verification is not implemented** in
   the open-source `generate.py` — left to the integrator.

→ B2 (rejection sampling over the batch verifier) + the official DSpark drafter
is the path that both fixes the Phase-2a regression AND measures real DSpark
acceptance/speedup.

## 1. Source-of-truth: the official DSpark checkpoint

Repo `deepseek-ai/DeepSeek-V4-Flash-DSpark`, config distilled:

| field | value | meaning |
|---|---|---|
| `dspark_block_size` | **5** | γ = 5 draft tokens (paper DSpark-5) |
| `dspark_target_layer_ids` | **[40, 41, 42]** | drafter reads mean-hidden of last 3 target layers |
| `dspark_markov_rank` | **256** | Markov head rank r |
| `dspark_noise_token_id` | **128799** | pad token for non-anchor draft positions |
| `n_mtp_layers` | **3** | drafter depth (mtp.0/1/2) |
| `quantization_config` | FP8 e4m3, ue8m0 scales, block [128,128]; expert_dtype fp4 | source quant |

**Drafter tensors live in shards 46/47/48** (~10.9 GB total). Per-layer structure
(dedup'd over 256 experts — full list in `issue468/baseline/dspark_tensors.txt`,
to be captured):

- **attn** (MLA, same family as target): `wq_a/wq_b/wo_a/wo_b` (LoRA,
  scale+weight), `wkv` (scale+weight), `q_norm`, `kv_norm`, `attn_sink`.
- **ffn**: 256× routed experts (`w1/w2/w3` each scale+weight) + shared_experts
  (w1/w2/w3) + `gate` (weight+bias).
- **hc gating** (DSpark-specific, all layers): `hc_attn_*`, `hc_ffn_*`
  (base/fn/scale).
- mtp.0 extra: `main_norm`, `main_proj` (target-hidden → drafter projection).
- mtp.2 extra: `markov_head.markov_w1/w2`, `confidence_head.proj`,
  `hc_head_*`, `norm`.

Reference algorithm (`inference/model.py`): `Transformer.forward_spec` runs the
3-layer **parallel backbone** over a block of γ+1 positions (1 anchor + γ
noise-padded), then `DSparkBlock.forward_head` applies the **sequential Markov
head** per position (logits += markov_w2(markov_w1(prev_token)); rank-256,
cheap) and samples, producing γ draft tokens + per-position confidence. **One
parallel backbone pass + γ cheap Markov evals = the semi-autoregressive draft.**

## 2. Phased delivery

### Phase 3 — Drafter tensor extraction + GGUF conversion (loader/format)

**Goal:** a single `dspark.gguf` loadable by ds4, holding the 3 drafter layers.

- Download shards 46/47/48 only (NOT the full 165 GB). ~10.9 GB.
- Write `issue468/convert_dspark.py`: read the safetensors, extract `mtp.*`
  tensors, quantize FP8/fp4 → ds4's GGUF block types (Q4_K/Q8_0 to match the
  existing MTP GGUF; evaluate Q8_0 for the small HC/Markov/confidence tensors
  to preserve accuracy). Emit `dspark.gguf` + a metadata header capturing
  block_size/target_layer_ids/markov_rank/noise_token_id.
- Capture `issue468/baseline/dspark_tensors.txt` (the dedup'd inventory) as the
  conversion spec.

**Placement:** a new conversion tool, not engine code. Output goes to
`../ds4/gguf/` next to the existing models.

### Phase 4 — DSpark drafter forward in ds4 (Metal)

**Goal:** `ds4` can run the DSpark drafter and emit γ draft tokens + logits +
confidence, given the target's hidden states at layers [40,41,42].

- **Hidden capture:** add capture points in the Metal target graph to export
  `mean(hidden, over hc_mult)` at layers 40/41/42, concatenated →
  `main_hidden` (dim = 3·4096). Reuse the existing `spec_*` buffer pattern.
- **DSpark drafter graph:** port `DSparkBlock` (parallel backbone: MLA
  attention with DSpark windowed topk `get_dspark_topk_idxs`, MoE FFN, HC
  combine) + `forward_embed` (main_proj/main_norm + noise-block embed +
  HC-expand) + `forward_head` (hc_head + norm + head + **sequential Markov
  head** + confidence head). New Metal kernels where the shapes differ from the
  target's; reuse the target's MoE/attn kernels where shapes align.
- **Loader:** extend the MTP GGUF loader (`ds4.c:4436` region) to recognize the
  DSpark 3-layer structure + Markov/confidence tensors.

This is the largest single work item. Keep it **draft-only** first (no
verification) to validate the drafter produces sensible tokens against the
reference `model.py` (`__main__` smoke).

### Phase 5 — B2 rejection-sampling verification

**Goal:** make the batch verifier quality-neutral by construction and measure
acceptance.

- Implement the textbook rejection sampler over the **batch verifier's** logits
  (`metal_graph_verify_suffix_tops` already produces per-position target
  logits): accept draft token x (from drafter dist q) with prob
  `min(1, p(x)/q(x))`; on reject, resample from `norm(max(0, p−q))`. Accept the
  longest such prefix + one bonus token (Chen 2023 / Leviathan 2023).
- This **dissolves** the Phase-2a regression: the batch verifier's drift is
  smooth under the rejection rule, not a step function.
- Wire behind `--dspark` (sampling path; requires temp>0). Greedy-argmax stays
  the default for the existing `--mtp` path.

### Phase 6 — Empirical acceptance + speedup (the deliverable)

**Goal:** the numbers that justify (or kill) the effort.

- **Acceptance rate by position** (the paper's Figure 2 analogue): conditional
  acceptance per draft position, chat + code, at ctx {2k/4k/8k}. Promote the
  deferred counter from `02_gap_and_spec.md`. Compare to the current MTP-1
  drafter (Phase 0: pos-2 ≈ 0.41) and the paper's DSpark (≈0.63–0.72).
- **End-to-end speedup** vs target-only: tokens/sec under `--dspark` (B2) vs
  plain decode, at matched quality (rejection sampling is provably
  distribution-exact, so quality parity is by construction; confirm on the 92
  eval cases — expect 67/92 ± noise).
- **Break-even analysis** using Phase 1's batch `verify(L)` curve + Phase 0's
  draft cost: does `(draft + verify(γ)) / accepted < plain_decode`? The
  DSpark drafter is heavier than MTP-1 (3 layers, full MoE), so draft cost must
  be re-measured, not reused.

## 3. Stop/Go gates

- **Phase 3 → 4:** conversion produces a loadable GGUF with all tensors
  accounted for (the inventory in §1 is the checklist).
- **Phase 4 → 5:** the drafter's sampled tokens match the reference
  `model.py.forward_spec` on a fixed prompt (regression test against the Python
  reference before trusting Metal).
- **Phase 5 → 6:** B2 restores 67/92 (±nondeterminism) on the eval set — i.e.
  the Phase-2a regression is gone.
- **Phase 6 (overall):** acceptance ≥ current MTP-1 (0.41) and trending toward
  the paper (0.6+), AND end-to-end speedup > 0 (ideally toward the ≥20% gate).
  If the heavier DSpark drafter's cost cancels the acceptance gain, that is a
  real (publishable) negative result and we stop local work.

## 4. Risks specific to this path

- **Drafter port size:** 3 layers of full MoE (256 experts) is much heavier
  than MTP-1. Draft cost could dominate; Phase 6's break-even may be negative
  even with good acceptance. Mitigation: the parallel backbone is O(1) passes
  regardless of γ; the per-token Markov head is rank-256 (cheap). Measure early.
- **FP8→GGUF quantization fidelity:** DSpark experts are fp4/e4m3 with
  [128,128] block scales; Q4_K/Q8_0 are different schemes. The drafter's
  accuracy under re-quantization must be validated against the reference
  before any speedup claim. Mitigation: keep HC/Markov/confidence tensors at
  Q8_0 or F16 (they're small).
- **Target hidden capture cost:** exporting layers 40/41/42 every decode step
  adds 3 device→device copies per token. Must be measured; should be sub-ms
  (small, contiguous).
- **Reference is MP=4 multi-GPU:** the official inference assumes 4-way tensor
  parallel. ds4 is single-device Metal. The drafter must fit in unified memory
  (~11 GB — fine on 128 GB M5 Max) and the ParallelEmbedding/ParallelHead
  column/row splits must be merged to single-device.

## 5. Open questions to resolve before Phase 4 coding

1. **Quant scheme:** Q4_K for experts (matches target, smallest) — accept the
   fidelity risk and validate, or Q8_0 (safer, ~2× drafter size)?
2. **Markov head dtype:** keep F16/Q8_0 (it drives token selection — accuracy
   matters most here).
3. **Confidence head:** needed only for the Phase-6 *scheduler* study, not for
   B2 itself. Defer its wiring to Phase 6; Phase 5 can ignore confidence.
4. **`dspark_block_size` at runtime:** fix at γ=5 (paper config) or make it a
   knob to reuse Phase 1's `verify(L)` curve for the break-even?
