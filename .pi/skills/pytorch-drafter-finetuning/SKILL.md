---
name: pytorch-drafter-finetuning
description: Patterns, pitfalls, and lessons for fine-tuning the DSpark drafter with PyTorch (LoRA, gradient probes, GGUF baking). Use before any drafter training, LoRA experiment, or F16 fidelity check involving drafter_body.py / drafter_head.py.
---

# PyTorch drafter fine-tuning

Hard-won lessons from Lead 10 (drafter re-distillation) + the milestone-2/3 drafter work.
Every entry below was paid for in debugging time or a codex gate finding.

**Pair with:**
- **`pytorch-numerical-modelling`** — the general torch drafter patterns (fidelity-gating,
  the GGUF loader, the forward_prompt interface, the dtype choices, the offline ≠ live
  engine gap, the bootstrap CI, the indexing conventions). This skill covers the
  **fine-tuning-specific** layer on top of those.
- **`research-lead`** — the methodology (orient first, lock the decision rule, fidelity-
  gate every change, two codex gates, scope the verdict honestly, single-process memory
  discipline). This skill covers the **technical execution** of a drafter fine-tuning lead.

## The drafter at a glance

- **Architecture**: 3-layer MTP MoE drafter (`mtp.0/1/2`). 19.85B params total, ~19.35B
  in the 256 routed experts (MXFP4, 6 active/token), ~0.5B dense. The output head
  (`hc_head_fn`) lives only at `mtp.2` (NOT `mtp.0/1`).
- **Weights**: `dspark.gguf` (11 GB). The target's `token_embd.weight` + `output.weight`
  (`lm_head`) are loaded from the **target GGUF** (81 GB) — they are shared with the
  verifier, not the drafter's own.
- **Torch port**: `issue468/dspark_train/{drafter_body.py, drafter_head.py}`. Built by
  `build_body(dspark_path, target_path, device, dtype)` + `build_head(...)`.

## Lesson 1: The experts are a memory wall — freeze them

Full fine-tune of the 19.85B MoE is infeasible at any sane corpus. At 30k anchors with
6 active/token, each ~25M-param routed expert sees only ~700 gradient updates → per-expert
overfitting. Plus MXFP4 training needs dequant→train→requant.

**Always**: freeze the routed experts. Train only the dense parts via LoRA (~0.5B params,
manageable).

```python
body.exp_gate.requires_grad_(False); body.exp_up.requires_grad_(False); body.exp_down.requires_grad_(False)
body.main_proj.requires_grad_(True)
for layer in body.layers:
    for v in layer.values(): v.requires_grad_(True)
```

## Lesson 2: F16 backward NaNs through the MoE

At F16, the backward through the drafter body's rmsnorm + the MoE's SwiGLU produces NaN
gradients (the F16 overflow in the intermediate accumulation). This is separate from the
hc_head overflow (Lesson 4) — it's the body's own backward that fails.

**Always**: compute gradients at F32. For a gradient probe (one-shot backward), use
`dtype=torch.float32` + `max_step=3` (limits the MoE activation retention to ~27 GB on MPS).
For LoRA training on precomputed features (Lesson 5), the body runs in `no_grad` so F16 is
fine for the body forward; the head trains at F32.

*(For the general F32-vs-F16 dtype guidance — including the block-divergence, the
Lead-04 dtype-invariance, and the p1-vs-block sensitivity — see
`pytorch-numerical-modelling` principle 3.)*

## Lesson 3: The LoRA zero-init B pattern

The drafter_head's LoRA init: `lora_hc_A ~ N(0, 0.02)`, `lora_hc_B = 0`. The delta
`B @ A` starts at 0 but `grad_B = grad @ A.T` is non-zero (A != 0) → B learns first, then A.
This is the correct standard LoRA init.

**Anti-pattern**: initializing BOTH A and B to zero → a dead LoRA (zero gradient forever).
This was a bug in an earlier version of drafter_head.py.

## Lesson 4: The hc_head's F16 overflow — the #1 numerical trap

The drafter body's output (the head's input) has values up to **1721** (abs). The
`hc_head` computes `rsqrt = 1/sqrt(mean(flat²))` where `flat = x.reshape(.., hc*d)`. At
F16, `flat²` for values > 255 overflows (`> 65504` → inf → the rsqrt → 0 → the hc reduction
collapses → p1 drops from ~0.85 to ~0.55).

**Root cause**: the torch port's `hc_head` computed the rsqrt + the matmul in the input
dtype (F16). The ds4's C implementation (`dspark_hc_head_one`, ds4.c:28286) computes the
RMS norm (`rms_norm_no_weight`) + the matvec (`matvec_f16`) + the sigmoid **all in F32**
(`float *` signatures + F32 accumulators — `rms_norm_no_weight` uses a `double ss`
accumulator at ds4.c:4614; `dot_f16_row` uses `float32x4_t acc` at ds4.c:4650).

**Fix** (applied to drafter_head.py): make `hc_head` + all head norms F32-internal:
```python
def hc_head(self, x):
    out_dt = x.dtype
    xf = x.float()  # F32-internal (faithful to ds4's dspark_hc_head_one)
    flat = xf.reshape(b, s, hc * d)
    rsqrt = 1.0 / torch.sqrt((flat * flat).mean(-1, keepdim=True) + NORM_EPS)
    mixes = (flat @ self._hc_fn().float().T) * rsqrt
    pre = torch.sigmoid(mixes * self.hc_scale[0] + self.hc_base) + HC_EPS
    return ((pre.unsqueeze(-1) * xf).sum(dim=2)).to(out_dt)
```

Apply the same F32-internal pattern to **every** rmsnorm in the head (`p1_scores`,
`k_scores`, `forward`).

**The general lesson**: when porting a C/CUDA implementation to torch, check every
normalization + matmul for dtype-sensitive intermediates. The C code often uses F32
accumulators internally even when the stored weights are F16.

## Lesson 5: Precompute the body features for head-only training

Training only the head (LoRA on `hc_head_fn`, `norm_w`, `markov_w1/w2`) doesn't need the
body's forward at all. Precompute the body's output (`xs[:, 0, :, :]` — the head's input)
once at F32 `no_grad`, cache to disk (`.npz`), then train the head on the cached features.
The body forward (with its 19.85B experts + the MoE activation retention) is the memory +
time bottleneck — precomputing side-steps it entirely.

```python
with torch.no_grad():
    xs = body.forward_prompt(mh_seq, anchors)  # F32, no retention
    x0 = xs[:, 0, :, :].float().cpu().numpy()
np.savez(cache, x0=x0, anchor=anchor, target=target)
del body  # free the ~92 GB
# train: load the cache → head.p1_scores(x0, anchor) → loss → backward (light)
```

**Anti-pattern**: running the body forward (with grad) inside the training loop. The MoE
activation retention (~9 GB/step at F32) will OOM. If the body must be in the loop (body
LoRA), use `max_step=3` + gradient checkpointing.

## Lesson 6: Baking a LoRA into the GGUF — `data_off + off`, not just `off`

When baking a trained LoRA into the dspark GGUF (modifying a weight tensor in-place for the
live ds4), the tensor's offset from `index_gguf` is **relative to the data section**, not
the absolute file offset. The absolute offset is `data_off + off`.

**Anti-pattern** (a real bug in Lead 10): using `off` as the absolute offset → writing the
delta to the wrong location (in the GGUF header/metadata) → the ds4 loads the original
(unchanged) weight → no effect.

```python
_, infos, data_off = index_gguf(DSPARK)
ne, dt, off = infos['mtp.2.hc_head_fn.weight']
abs_off = data_off + off  # ABSOLUTE file offset
```

**Also**: the GGUF's `ne` ordering — `_gguf_ne_to_torch` does `arr.reshape(reversed(dims))`,
a simple C-order reshape (NOT a transpose). When writing back, use
`modified.astype(F16).tobytes()` (the C-order flat), not `.T.tobytes()`.

*(For the general GGUF loader patterns — `load_gguf_dense_only`, `index_gguf`,
`read_tensor`, the embed/lm_head loading from the target GGUF — see
`pytorch-numerical-modelling` principle 4.)*

## Lesson 7: The torch oracle ≠ the live ds4 — always validate on the real engine

An offline-trained LoRA that improves the torch oracle's p1 (+2.30 pp) does NOT necessarily
improve the live ds4's acceptance (E[a|K] flat). The torch port + the live ds4 drafter are
**different regimes** — the same baked weight produces different acceptance on the two.

**Any offline-trained drafter adaptation must be validated on the live ds4** (bake into the
GGUF + run the ds4-spec-bench `speculative_argmax`), not just the torch oracle. The torch
oracle is a development tool (fast iteration, gradient probes) but NOT a deployment proof.

*(This is the fine-tuning-specific manifestation of `pytorch-numerical-modelling`
principle 2: "the offline drafter ≠ the live engine." For the general offline-vs-live
guidance — including the block-divergence, the dtype gap, and the throughput discrepancy —
see there. For the research-lead's "separate deployable from diagnostic" principle, see
`research-lead` principle 11.)*

## Lesson 8: Gradient probe ranking — RMS, not raw ‖∇W‖

To select LoRA targets empirically, compute the CE loss gradient at each weight + rank by
the gradient signal. The raw `‖∇W‖` is confounded by weight size. Use `‖∇W‖ / √n_params`
(the per-element RMS gradient) for a fair cross-weight comparison.

**Anti-pattern**: `‖∇W‖ / ‖W‖` (the relative update) — this inflates tiny weights (the
HC-scale weights, ‖W‖ ≈ 0.06 → ratios of 70–320), which are scalar/vector, not LoRA targets.

**Exclude** `lm_head` + `embed_w` — they're loaded from the **target GGUF** (shared with the
verifier). Training them has a deployment problem.

## Lesson 9: Soft labels (KL) ≈ hard labels (CE) for peaked teachers

The IQ2 target's distribution is highly peaked (the top token often ~99.75%). The KL vs the
top-128 ≈ the CE vs the greedy token. The LoRA trained with KL (+2.58 pp) vs CE (+2.41 pp) —
close. For a quick probe/ablation, CE (hard labels) is a valid proxy. The full training
should use KL (the specified loss), but the CE version gives a fast directional read.

**If the soft labels are dropped** (as happened in Lead-11's Exp-0a consolidation), the CE
proxy (using the greedy `sel` from the h.bin `tok` field) keeps the work unblocked. The soft
labels require `--dump-logprobs-jsonl` (ds4-spec-bench) to re-capture.

## Workflow: the drafter fine-tuning pipeline

1. **Capture** the training data: `ds4-spec-bench --dump-hidden-dir` (H_iq2) +
   `--dump-logprobs-jsonl` (the soft labels). Or reuse the Lead-11 unified capture.
2. **Precompute** the body features (Lesson 5): F32 `no_grad` → cache `xs[:, 0]` to `.npz`.
3. **Gradient probe** (optional, Lesson 8): F32, max_step=3, CE vs sel → rank by RMS.
4. **Train** the head LoRA: on the cached features, KL vs the top-128 (or CE proxy), F32,
   AdamW (lr=3e-4, wd=0.01).
5. **F16 fidelity check** (Lesson 4): build the head at F16 (with the F32-internal fix) →
   verify F16 ≈ F32.
6. **Bake** the LoRA into the GGUF (Lesson 6): `data_off + off`, F16 C-order.
7. **Live ds4 validation** (Lesson 7): `ds4-spec-bench --dspark dspark_lora.gguf
   --mode speculative_argmax` → compare E[a|K] + t/s to the baseline. **This is the
   decisive test.**

*(For the fidelity-gate methodology — reproduce a known-good baseline before trusting any
variant — see `research-lead` principle 4. For the torch-specific fidelity gate vs the
numpy oracle, see `pytorch-numerical-modelling` principle 1. For the two-codex-gate
discipline — setup + verdict — see `research-lead` principle 5.)*

## Memory + process discipline

The drafter fine-tuning shares the ~128 GB Apple-Silicon box with the 87 GB target model.
**Never** run two heavy processes concurrently. Monitor swap (`sysctl vm.swapusage`). On a
crash, switch to a lighter path (reduce max_step, use precomputed features, move to CPU).
Do NOT retry the same way.

*(For the full single-process memory discipline — the tool profiles, the crash-recovery
patterns, the swap monitoring — see `research-lead` principle 9.)*

## Key file locations

- Drafter torch port: `issue468/dspark_train/{drafter_body.py, drafter_head.py}`
- GGUF loader: `issue468/dspark_oracle/gguf_loader.py`
- Model inventory: `issue468/inventories/dsv4_flash_dspark_model.md`
- Lead 10 archive (the full investigation): `issue468/archive/leads/lead_10_drafter_redistillation.md`
- Phase 0 artifacts: `issue468/artifacts/lead10_phase0/`
- Phase 1 artifacts: `issue468/artifacts/lead10_phase1/`
- Phase 2 artifacts: `issue468/artifacts/lead10_phase2/`
