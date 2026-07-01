# Drafter Drift Bisection — Propagation Refuted; mtp.2 Attention Bug Confirmed Real

Date: 2026-07-01. Thirteenth productionization handoff note. Continues
issue468/53 (which isolated the divergence to mtp.2 attention). This note
records the decisive propagation test + exhaustive hypothesis elimination.
Doc-only research record.

## 0. Question answered

issue468/53 showed mtp.2's attention diverges 11-20% from the oracle while
layers 0,1 are clean (<4.3%). But that comparison fed the oracle's OWN
layer-1 output into its layer 2 — conflating "layer-2 bug" with "propagation
of layer-1's small error." This note settles it.

## 1. DECISIVE: propagation REFUTED — it's a genuine layer-2-specific bug

Fed Metal's ACTUAL layer-1 output (metal_h_stepNN_lay1.bin) into the oracle's
layer-2 block, compared to Metal's layer-2 output. If the divergence were
propagation, feeding the correct input would shrink it dramatically. It didn't:

| step | oracle-from-METAL-input vs Metal-L2 | oracle-from-oracle-input vs Metal-L2 |
|---|---|---|
| 1 | 17.2% (max 77.3) | 20.5% (max 90.4) |
| 2 | 9.5% (max 59.1) | 10.0% (max 62.9) |
| 3 | 6.4% (max 40.6) | 9.7% (max 60.8) |
| 4 | 19.0% (max 75.5) | 18.1% (max 79.1) |
| 5 | 24.0% (max 97.9) | 25.5% (max 109.3) |

The two columns are essentially identical → the input source doesn't matter.
**Same input → divergent output for layer 2 only.** Genuine layer-2-specific bug.

## 2. It's DIRECTIONAL divergence, not scale (cosine 0.975)

Layer-2 output magnitudes MATCH the oracle (oracle max 440-628/mean 6.2 vs
Metal 438-589/mean 6.0 — within 5%). But per-element:
- diff p50=0.91, p99=8.2, max=90.4 (on |oracle| mean 6.2 → 23.5% mean rel)
- **cosine similarity mean 0.975, min 0.844, p1 0.864** — well below precision
  noise (>0.999). This is a SEMANTIC directional divergence: same magnitude,
  noticeably different direction. Consistent with the attention attending
  slightly differently (different score pattern → different weighted average),
  not a global scale error.

## 3. Exhaustive hypothesis elimination (10 ruled out)

Given identical code path, identical weight types, same input → divergent
output ONLY for layer 2:

1. ~~Propagation~~ — REFUTED (§1).
2. ~~Weight quant type mismatch~~ — all attn weights Q8_0/F32/F16 IDENTICAL
   across stages (issue468/53 §7).
3. ~~Q8_0 block-scale edge case~~ — ALL Q8_0 scales are uniform 0.00003 across
   every block and every layer → can't be layer-specific.
4. ~~NaN/Inf numerical blowup~~ — clean (0 nonfinite; layer-2 just larger
   magnitude, which is inherent — oracle has it too).
5. ~~Wrong magnitude (amplification)~~ — magnitudes match oracle (§2).
6. ~~Routing (ffn_gate_inp F16)~~ — expert selection matches ~93% (issue468/53).
7. ~~MoE~~ — post-ATTN intermediate already diverges 11-20% before MoE (issue468/53 §6).
8. ~~Per-layer allocation/size~~ — dspark_kv_cache[0..2] uniform alloc loop.
9. ~~Weight binding offset~~ — GGUF tensor offsets/dims consistent across stages.
10. ~~batch_attn_norm computed once (input reuse)~~ — recomputed per-layer inside
    encode_attention (hc_pre+rmsnorm at ds4.c:~17707).

## 4. What remains (the elusive root cause)

A genuine computational divergence isolated to mtp.2's attention, with all
obvious causes eliminated. Remaining candidates are subtle GPU-side:
- **GPU buffer aliasing/reuse**: a scratch tensor that layers 0,1 populate
  benignly but layer 2 reads with stale/wrong content. Needs a per-layer
  intermediate dump INSIDE the Metal attention kernel.
- **GPU Q8_0 dequant accumulation-order divergence** triggered by mtp.2's
  specific int8 weight VALUES (not type/scale — the actual values), where the
  GPU's reduction order differs from numpy's. Would need a direct Metal-vs-CPU
  dequant comparison on mtp.2's bytes.
- **Window-KV gather layout edge case** for the last stage.

## 5. The GPU dump obstacle (why the final isolation is hard)

The op-level isolation needs the multi-head-attention output (`batch_heads`)
per layer. The existing DS4_DSPARK_PROBE_DUMP_HEADS hook fires inside
encode_attention (within an active command buffer); it works for layer 0 but
the mid-command-buffer ds4_gpu_synchronize() disrupts subsequent layers
(layers 1,2 dumps don't appear). Moving the dump to AFTER the per-layer
encode_block sync (in the probe loop, where batch_heads still holds the
just-computed layer's MHSA before the next layer overwrites it) is the clean
fix and the concrete next step.

## 6. Significance

The bug is PROVEN real (propagation refuted, cosine 0.975 semantic divergence,
10 hypotheses eliminated). This means the +8.27% oracle acceptance headroom
(issue468/51) IS recoverable — it's a genuine implementation defect in one
layer, not inherent quantization noise. Per issue468/42's break-even, fixing
mtp.2's attention to layer-0/1 fidelity recovers the full gap → α 0.83→0.90 →
~41.5 t/s, crossing the gate.

The work is no longer "find out if there's a bug" (there is) but "isolate the
exact GPU operation" — a single targeted dump-compare, blocked only by the
mid-command-buffer sync timing.

## 7. Honest status / next

- FP8-off fix LANDED (issue468/52, +2.6% quality, free) — recovered ground.
- mtp.2 attention bug PROVEN, localization tight, root cause ~1 dump away.
- Final isolation needs the post-sync MHSA dump (§5) → compare to oracle's
  layer-2 `o` → bisect q/kv/scores/output-proj → fix → re-measure gate.

This is bounded but deep GPU debugging. The gate-crossing path is real and
in-scope (code-only, frozen drafter); the remaining work is isolating one
elusive GPU operation.
