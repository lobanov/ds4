# Phase 3 — DSpark Drafter Conversion: RESULTS (gate passed)

Date: 2026-06-28. Branch: `dspark`. Supersedes iter-1 (`10_converter_reuse.md`).

## Outcome: PASS. `dspark.gguf` produced and validated.

`../ds4/gguf/dspark.gguf` — **10.71 GiB** (11,501,064,576 bytes), 81 tensors, all
3 drafter layers (mtp.0/1/2). Loads without parse error; all F32/BF16 tensors
byte-match the HF source; fits the RAM budget with 29.5 GiB headroom.

## Conversion summary

- **Source:** `deepseek-ai/DeepSeek-V4-Flash-DSpark` shards 46/47/48 (~10.9 GB),
  integrity-verified (slack=0, 0 bad offsets; shard 47 sha256 matches HF LFS etag).
- **Tool:** adapted `gguf-tools/deepseek4-quantize` + synthesized template
  (`issue468/build_dspark_template.py`). ~90% of the converter reused as-is
  (safetensors loader, FP4/FP8 dequant, Q4_K/Q8_0 quantizers, GGUF writer,
  threaded expert fusion). Small DSpark-specific patch (mtp.* name mapping).
- **Recipe:** Q4_K routed experts (9 packed tensors, 256 experts each),
  Q8_0 attn/shared/main_proj, F32 norms/hc_*/sinks/gate, BF16 markov/confidence.
- **Runtime:** 78s (8 threads).

### Tensor inventory (81 = 26 + 24 + 31 across mtp.0/1/2)

| type | count | what |
|---|---|---|
| F32 | 42 | norms, hc_attn/hc_ffn base/fn/scale, attn_sinks, exp_probs_b, hc_head (mtp.2) |
| Q8_0 | 25 | attn projections (q_a/q_b/kv/output_a/b), shared experts, main_proj |
| Q4_K | 9 | routed experts: ffn_gate/up/down_exps × 3 layers (256 experts packed each) |
| BF16 | 5 | main_norm (mtp.0), norm (mtp.2), markov_w1/w2, confidence_head.proj (mtp.2) |

> **Note on the "87 tensors" contract figure.** The phase3-convert contract and
> the iter-1 template said "87 tensors." That count came from iter-1's template
> erroneously emitting `hc_head_*` on mtp.0/1 — tensors that **do not exist** in
> the HF checkpoint (`hc_head_*` is mtp.2-only, the drafter's output-stage HC
> head; verified against the HF inventory). The HF-faithful count is **81**.
> This is the complete drafter; nothing is missing.

## Bugs found & fixed during full conversion (beyond iter-1)

1. **Template: phantom `hc_head_*` on mtp.0/1.** iter-1 emitted
   `hc_head_base/fn/scale` for all 3 layers; HF only has them on mtp.2. Fixed in
   `build_dspark_template.py` (gated behind `L == 2`). Without this, conversion
   died at tensor 1 (`HF tensor not found: mtp.0.hc_head_base`).
2. **Converter: `check_reversed_shape` rejected HF `[1,N]` tensors.** The check
   strips trailing size-1 dims on the GGUF/template side (via `tensor_n_dims`)
   but compared against the HF side's raw rank. DSpark's
   `mtp.2.confidence_head.proj` is genuinely HF shape `[1, 4352]` (rank 2), so a
   1-D template could never match. Fixed: also strip leading-1 HF dims (the
   equivalent operation on the reversed-order HF shape). Normal `[out,in]`
   weights have a non-1 outer dim and are unaffected. iter-1 masked this because
   its crosscheck only read `markov_w1`, not the last tensor.
3. **Residency doc: drafter size was wrong.** iter-1 estimated "6.18 GiB" for the
   Q4_K drafter; the **measured** converted size is **10.71 GiB**. Q4_K still
   fits (29.5 GiB headroom, 3.7× the 8 GiB min), so the recipe decision is
   unchanged. Doc corrected.

## Validation (two independent methods agree)

### Method A — `--compare-tensor` (gguf-tools authoritative round-trip)
`issue468/crosscheck_dspark_compare_tensor.py` loops the converter's own
`--compare-tensor` over all 47 F32/BF16 non-expert tensors. For lossless tensors
the regeneration is identity, so `byte_compare: OK` ⟺ stored-GGUF == raw-HF-bytes.

```
mtp.0: 14 OK, 0 fail
mtp.1: 13 OK, 0 fail
mtp.2: 20 OK, 0 fail
total: 47 OK, 0 fail
```

Plus one FP4→Q4_K packed-expert spot check (`mtp.0.ffn_gate_exps.weight`,
1.2 GiB) round-trips OK (dequant+requant deterministic; HF-fidelity is the
Phase-4 forward gate, not byte-exact).

### Method B — direct HF↔GGUF byte-match (converter-independent)
`issue468/crosscheck_mtp2.py` reads both the HF safetensors and the GGUF directly
and compares bytes, with NO converter in the loop:

```
lossless byte-exact:   32 OK, 0 bad
BF16->F32 value-exact: 15 OK, 0 bad
```

- **32 byte-exact:** all F32↔F32 and BF16↔BF16 tensors (hc_attn/hc_ffn
  base/fn/scale all layers, attn_sinks all layers, exp_probs_b all layers,
  hc_head mtp.2, main_norm mtp.0, norm mtp.2, markov_w1/w2, confidence_head.proj).
- **15 value-exact BF16→F32 upcasts:** the norm/gate tensors that are BF16 in HF
  but stored F32 in the GGUF (attn_norm/ffn_norm/q_norm/kv_norm/gate_inp all
  layers) — lossless precision upcast, values identical.

### Out of scope for Phase 3 (Phase-4 forward gate)
- FP8_E4M3 → Q8_0 dequant+requant (attn projections, shared experts, main_proj):
  not byte-exact by construction; validated by drafter-forward token agreement.
- FP4 → Q4_K routed experts: same; validated by drafter-forward token agreement.

## Reusable gguf-tools assets (assessment)

- **`--compare-tensor` — YES, the right validator.** Authoritative GGUF reader
  (correct `general.alignment` + per-tensor padding), regenerates-from-HF and
  byte-compares. Used as Method A above.
- **`quality-testing/` (score_official.c + collect_official.py) — NOT applicable.**
  Model-level target-token NLL vs official API; needs a standalone runnable
  model. The drafter needs target hidden states, so it can't be scored this way.
  Redundant for Phase 6 anyway (`ds4-eval` has its own 92-case harness).
- **`mixed/splice_mixed_expert_layers_gguf.py` — not a validator**, but its GGUF
  offset handling is a correct reference.

## Gate status

| Phase 3 gate item | status |
|---|---|
| dspark.gguf from shards 46+47+48 | ✅ 10.71 GiB, 81 tensors |
| converter shows all mtp.0/1/2 tensors | ✅ 81 (contract's "87" was the buggy iter-1 template; 81 is HF-faithful and complete) |
| residency check recorded (recipe + headroom) | ✅ Q4_K, 29.5 GiB headroom (3.7× min) — `dspark_residency_check.md` |
| crosscheck exact byte-match on BF16/F32 per layer | ✅ 47/47 (Method A + B agree) |
| GGUF loads without parse error | ✅ parsed by 3 independent readers |

**Phase 3 → Phase 4 gate: PASSED.** Next: Phase 4 — DSpark drafter forward in ds4
(loader + target hidden capture at [40,41,42] + Metal forward + Python-ref
`model.py.forward_spec` token-agreement regression).
