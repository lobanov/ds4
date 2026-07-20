---
name: pytorch-drafter-finetuning
description: Patterns, pitfalls, and lessons for fine-tuning the DSpark drafter with PyTorch on Apple Silicon (MPS). Use before any drafter training, gradient probe, LoRA experiment, or F16 fidelity check involving drafter_body.py / drafter_head.py.
---

# PyTorch drafter fine-tuning

Hard-won lessons from Lead 10 (drafter re-distillation) + the milestone-2/3 drafter work.
Every entry below was paid for in debugging time or a codex gate finding. Read before
touching `drafter_body.py`, `drafter_head.py`, or any drafter training loop.

## The drafter at a glance

- **Architecture**: 3-layer MTP MoE drafter (`mtp.0/1/2`). 19.85B params total, ~19.35B
  in the 256 routed experts (MXFP4, 6 active/token), ~0.5B dense. The output head
  (`hc_head_fn`) lives only at `mtp.2` (NOT `mtp.0/1`).
- **Weights**: `/Users/lobanov/Projects/ds4/gguf/dspark.gguf` (11 GB). The target's
  `token_embd.weight` + `output.weight` (`lm_head`) are loaded from the **target GGUF**
  (81 GB), not the dspark GGUF — they are shared with the verifier, not the drafter's own.
- **Torch port**: `issue468/dspark_train/{drafter_body.py, drafter_head.py}`. Built by
  `build_body(dspark_path, target_path, device, dtype)` + `build_head(...)`.
- **Captures**: the drafter's input is `main_hidden [N, 12288]` = concat(mean(hc_ffn_post[40/41/42]))
  — the target's layers-40/41/42 post-FFN HC residual, mean-reduced over HC=4. Captured via
  `DS4_DSPARK_DUMP_HIDDEN` (ds4-spec-bench `--dump-hidden-dir`) or the Lead-11 unified capture.

## Lesson 1: The experts are a memory wall — freeze them

**Pattern**: full fine-tune of the 19.85B MoE is infeasible at any sane corpus. At 30k anchors
with 6 active/token, each ~25M-param routed expert sees only ~700 gradient updates → per-expert
overfitting. Plus MXFP4 training needs dequant→train→requant.

**Always**: freeze the routed experts (`exp_gate/exp_up/exp_down`). Train only the dense parts
(attention, `main_proj`, `hc_head_fn`, the norms, the shared expert). Use LoRA (not full
fine-tune) on the dense matrices — they're ~0.5B params, manageable.

```python
# freeze the experts
body.exp_gate.requires_grad_(False)
body.exp_up.requires_grad_(False)
body.exp_down.requires_grad_(False)
# enable grad only on the dense targets
body.main_proj.requires_grad_(True)
for layer in body.layers:
    for k, v in layer.items():
        v.requires_grad_(True)
```

## Lesson 2: F16 backward NaNs through the MoE — use F32 for gradients

**Pattern**: the MoE's expert-gather activation retention is ~9 GB/step at F32 (the
`self.exp_gate[s][ids]` tensor is `[block, topk, 4096, 2048]`). At F16 it's ~4.5 GB/step but
the backward through the rmsnorm + the MoE's SwiGLU produces NaN gradients.

**Always**: compute gradients at F32. For the gradient probe (a one-shot backward), use
`dtype=torch.float32` + `max_step=3` (limits the activation retention to ~27 GB, fitting within
the 182 GB MPS limit). For LoRA training on precomputed features (the head only), the body runs
in `no_grad` so F16 is fine for the body forward; the head trains at F32.

**Anti-pattern**: loading the drafter at F16 + calling `.backward()`. The body's backward will
NaN. The head's backward will also NaN if the hc_head's rmsnorm overflows (see Lesson 4).

```python
# gradient probe: F32, small max_step
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
# process max_step=3 anchors per prompt (fits memory)
```

## Lesson 3: The head's LoRA is zero-init B (not both-zero)

**Pattern**: the drafter_head's LoRA init is `lora_hc_A ~ N(0, 0.02)`, `lora_hc_B = 0`.
The delta `B @ A` starts at 0 but `grad_B = grad @ A.T` is non-zero (A != 0) → B learns first,
then A. This is the correct init (the standard LoRA init).

**Anti-pattern**: initializing BOTH A and B to zero → a dead LoRA (zero gradient forever —
`grad_B = grad @ 0 = 0`, `grad_A = 0 @ grad = 0`). This was a bug in an earlier version of
drafter_head.py, documented in the code comments.

## Lesson 4: The hc_head's F16 overflow — the #1 numerical trap

**Pattern**: the drafter_body's output (the head's input) has values up to **1721** (abs).
The `hc_head` computes `rsqrt = 1/sqrt(mean(flat²))` where `flat = x.reshape(.., hc*d)`. At F16,
`flat²` for values > 255 overflows (`> 65504` → inf → the rsqrt → 0 → the hc reduction collapses
→ p1 drops from ~0.85 to ~0.55).

**Root cause**: the torch port's `hc_head` computed the rsqrt + the matmul in the input dtype
(F16). The ds4's C implementation (`dspark_hc_head_one`, ds4.c:28286) computes the RMS norm
(`rms_norm_no_weight`) + the matvec (`matvec_f16`) + the sigmoid **all in F32** (`float *`
signatures + F32 accumulators). The torch port wasn't faithful.

**Fix** (applied to drafter_head.py): make `hc_head` + all head norms F32-internal:
```python
def hc_head(self, x):
    out_dt = x.dtype
    xf = x.float()                          # cast to F32
    flat = xf.reshape(b, s, hc * d)
    rsqrt = 1.0 / torch.sqrt((flat * flat).mean(-1, keepdim=True) + NORM_EPS)
    mixes = (flat @ self._hc_fn().float().T) * rsqrt
    pre = torch.sigmoid(mixes * self.hc_scale[0] + self.hc_base) + HC_EPS
    return ((pre.unsqueeze(-1) * xf).sum(dim=2)).to(out_dt)  # cast back
```

Apply the same F32-internal pattern to **every** rmsnorm in the head (`p1_scores`, `k_scores`,
`forward`). The body's rmsnorms (in `drafter_body.py`) are less sensitive (the body's hidden
states are smaller) but should also be checked if the body is run at F16.

**The general lesson**: when porting a C/CUDA implementation to torch, check every normalization
+ matmul for dtype-sensitive intermediates. The C code often uses F32 accumulators internally
even when the stored weights are F16. The torch port must match.

## Lesson 5: Precompute the body features for head-only training

**Pattern**: training only the head (LoRA on `hc_head_fn`, `norm_w`, `markov_w1/w2`) doesn't
need the body's forward at all. Precompute the body's output (`xs[:, 0, :, :]` — the head's
input) once at F32 no_grad, cache to disk (`.npz`), then train the head on the cached features.

**Advantage**: the body forward (with its 19.85B experts + the MoE activation retention) is the
memory + time bottleneck. Precomputing side-steps it entirely. The head training is then light
(~2 GB for the LoRA params + the cached features).

```python
# precompute: body forward (no_grad, F32) → cache the features
with torch.no_grad():
    xs = body.forward_prompt(mh_seq, anchors)
    x0 = xs[:, 0, :, :].float().cpu().numpy()
np.savez(cache, x0=x0, anchor=anchor, target=target)
del body  # free the experts
# train: load the cache → head.p1_scores(x0, anchor) → loss → backward
```

**Anti-pattern**: running the body forward (with grad) inside the training loop. The MoE
activation retention (~9 GB/step at F32) will OOM on most hardware. If the body must be in the
loop (e.g., body LoRA), use `max_step=3` + gradient checkpointing.

## Lesson 6: The GGUF offset — `data_off + off`, not just `off`

**Pattern**: when baking a trained LoRA into the dspark GGUF (modifying a weight tensor in-place),
the tensor's offset from `index_gguf` is **relative to the data section**, not the absolute file
offset. The absolute offset is `data_off + off` where `data_off` is the GGUF's data section start.

**Anti-pattern**: using `off` as the absolute offset → writing the delta to the wrong location
(in the GGUF header/metadata, not the tensor data) → the ds4 loads the original (unchanged)
weight → no effect.

```python
_, infos, data_off = index_gguf(DSPARK)
ne, dt, off = infos['mtp.2.hc_head_fn.weight']
abs_off = data_off + off  # the ABSOLUTE file offset
with open(dspark_lora, 'r+b') as f:
    f.seek(abs_off)
    f.write(modified.astype(np.float16).tobytes())
```

**Also**: the GGUF's `ne` ordering (ne[0] is the fastest-varying). The `_gguf_ne_to_torch`
function does `arr.reshape(reversed(dims))` — a simple C-order reshape, NOT a transpose. When
writing back, use `modified.astype(F16).tobytes()` (the C-order flat), not `.T`.

## Lesson 7: The torch oracle ≠ the live ds4 — the regime discrepancy

**Pattern**: the torch port (`drafter_body` + `drafter_head`) and the live ds4 (the C/Metal
implementation) produce **different** p1/acceptance for the same prompts + the same weights. The
torch oracle's baseline (~0.844 on lead3) ≠ the combined300 (~0.795 for the same IDs). An
offline-trained LoRA that improves the torch oracle's p1 (+2.30pp) does NOT improve the live
ds4's acceptance (E[a|K] flat).

**Root cause** (unresolved): the two implementations differ somewhere in their forward —
possibly the body's F16 handling (the torch body at F16 overflows; the ds4 body doesn't), or a
subtle indexing/convention difference. The Lead 02 `compare_live_vs_oracle` tool was designed to
catch this but may not have covered all cases.

**Implication**: any offline-trained drafter adaptation (LoRA, fine-tune) must be validated on
the **live ds4** (bake into the GGUF + run the ds4-spec-bench), not just the torch oracle. The
torch oracle is a useful development tool (fast iteration, gradient probes) but NOT a deployment
fidelity proof.

## Lesson 8: The gradient probe — rank targets by ‖∇W‖/√n (RMS), not raw ‖∇W‖

**Pattern**: to select LoRA targets empirically (which dense weights to adapt), compute the CE
loss gradient at each weight + rank by the gradient signal. The raw `‖∇W‖` is confounded by the
weight's size (bigger weights → bigger gradients). Use `‖∇W‖ / √n_params` (the per-element RMS
gradient) for a fair cross-weight comparison.

**Anti-pattern**: ranking by `‖∇W‖ / ‖W‖` (the relative update). This inflates tiny weights (the
HC-scale weights, ‖W‖ ≈ 0.06 → ratios of 70–320), which are scalar/vector weights — not
meaningful LoRA targets (train them directly or freeze, not LoRA).

**Also**: exclude `lm_head` + `embed_w` from the LoRA candidates — they are loaded from the
**target GGUF** (shared with the verifier), not the drafter's own weights. Training them has a
deployment problem (you'd be modifying the shared head).

## Lesson 9: Soft labels (KL) vs hard labels (CE) — KL ≈ CE for peaked teachers

**Pattern**: the IQ2 target's distribution is highly peaked (the top token often has ~99.75%
probability). The KL divergence vs the top-128 ≈ the cross-entropy vs the greedy token (the top
token dominates both losses). The LoRA trained with KL (+2.58pp) vs CE (+2.41pp) — close, with
KL slightly higher.

**Implication**: for a quick probe/ablation, CE (hard labels, just the greedy token `sel`) is a
valid proxy for the KL (soft labels, the top-128 distribution). The full training should use KL
(the goal's specified loss), but the CE version gives a fast directional read.

**Capture**: the greedy tokens (`sel`) are available from the h.bin's `tok` field (the unified
capture). The soft labels (the top-128) require `--dump-logprobs-jsonl` (ds4-spec-bench). If the
soft labels are dropped (as happened in Lead-11's Exp-0a consolidation), the CE proxy keeps the
work unblocked.

## Lesson 10: Single-process memory discipline on Apple Silicon

**Pattern**: the target GGUF (81 GB) + the dspark GGUF (11 GB) = ~92 GB. On a 128 GB M5 Max,
this leaves ~36 GB for everything else (the OS, torch, the activations). The MPS backend has a
~182 GB limit (the unified memory).

**Rules**:
- **Never** run two heavy processes concurrently (e.g., the ds4-spec-bench + a torch training
  loop, or two ds4-spec-bench runs).
- **Monitor swap**: `sysctl vm.swapusage`. If swap grows unbounded, kill + reduce scope (fewer
  prompts, smaller max_step, smaller batch).
- **On a crash**: switch to a lighter path (reduce max_step, use precomputed features, move to
  CPU). Do NOT retry the same way.
- **The drafter load** (build_body + build_head): ~30s. The experts are mmap'd (~19 GB F32 on
  disk, read per-array). The dense parts are ~2 GB in GPU memory.
- **The ds4-spec-bench**: loads the target (~5 min warm) + the dspark (~1 min). Budget ~10 min
  per run for the load + the generation.

## Workflow: the drafter fine-tuning pipeline

1. **Capture** the training data: `ds4-spec-bench --dump-hidden-dir` (the H_iq2 main_hidden)
   + `--dump-logprobs-jsonl` (the Y_iq2 sel + the IQ2 top-128 soft labels). Or reuse the Lead-11
   unified capture.
2. **Precompute** the body features: `body.forward_prompt(H_iq2, anchors)` at F32 no_grad →
   cache `xs[:, 0, :, :]` to a `.npz`.
3. **Gradient probe** (optional, for target selection): F32, max_step=3, CE vs sel → rank the
   dense candidates by RMS gradient.
4. **Train** the head LoRA: on the cached features, KL vs the IQ2 top-128 (or CE vs sel as a
   proxy), F32, AdamW (lr=3e-4, wd=0.01).
5. **F16 fidelity check**: build the head at F16 (with the F32-internal hc_head fix) → verify
   F16 ≈ F32 (the dtype-invariance).
6. **Bake** the LoRA into the dspark GGUF: compute the delta (`B @ A`), modify the weight at
   `data_off + off` (F16 C-order).
7. **Live ds4 validation**: `ds4-spec-bench --dspark dspark_lora.gguf --mode speculative_argmax`
   → compare E[a|K] + t/s to the baseline. **This is the decisive test** — the torch oracle is
   NOT sufficient.

## Key file locations

- Drafter torch port: `issue468/dspark_train/{drafter_body.py, drafter_head.py}`
- GGUF loader: `issue468/dspark_oracle/gguf_loader.py` (`index_gguf`, `read_tensor`,
  `load_gguf_dense_only`, `_gguf_ne_to_torch`)
- Analysis oracle: `issue468/dspark_oracle/{forward.py, measure_acceptance_bundle.py,
  analyze_phaseB_gap.py}`
- Model inventory: `issue468/inventories/dsv4_flash_dspark_model.md`
- Lead 10 archive (the full investigation): `issue468/archive/leads/lead_10_drafter_redistillation.md`
- Phase 0 artifacts: `issue468/artifacts/lead10_phase0/`
- Phase 1 artifacts: `issue468/artifacts/lead10_phase1/` (features cache, training scripts,
  the F16 fix, the learning curve)
- Phase 2 artifacts: `issue468/artifacts/lead10_phase2/` (the validation, the baking, the live
  ds4 results)
