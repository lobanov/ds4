# Phase 3 — Converter Reuse Investigation (findings)

Investigated `gguf-tools/` to adapt existing tooling for the DSpark drafter
conversion rather than reinventing it. **Conclusion: ~90% of the converter is
reusable as-is; a small patch + a synthesized template GGUF is the path.**

## 1. What exists

- **`deepseek4-quantize.c`** (1908 lines): plain-C HF-safetensors → GGUF
  quantizer, no GGML dependency. Self-contained: sharded-safetensors loader
  (`st_db`/`shard`/`db_read`), JSON tokenizer, GGUF reader/writer, FP4/FP8
  **dequantizers** (`dequant_fp4_weight:711`, `dequant_fp8_weight:683`,
  `e8m0_to_f32:631`, `e4m3fn_to_f32:638`), Q2/Q4/Q8/IQ2XXS **quantizers**
  (`quants.c`), imatrix, threaded expert workers.
- **Architecture (template-driven):** load template GGUF → its tensor list
  (names/shapes/types) + KV metadata define the *output*; for each tensor,
  derive the HF source name, read+dequant+requant, write. `policy_type` picks
  the output type (1-D tensors keep template type; families overridable via
  `--experts`/`--attention-proj`/etc.).
- **`--compare-tensor`**: regenerate one tensor, byte-compare vs reference GGUF.
- **`quality-testing/`**: `collect_official.py` + `score_official.c` +
  `compare_scores.py` — a harness comparing local GGUF vs official continuations
  (relevant for the Phase-4 drafter validation gate).

## 2. Reuse assessment for DSpark

| converter piece | reusable for DSpark? | notes |
|---|---|---|
| sharded safetensors loader | ✓ as-is | reads `model.safetensors.index.json`; works for shards 46-48 |
| FP4 dequant (`dequant_fp4_weight`) | ✓ as-is | DSpark experts are I8-packed-fp4 + E8M0 scales — identical to target |
| FP8 dequant (`dequant_fp8_weight`) | ✓ as-is | DSpark dense weights are F8_E4M3 + F8_E8M0 pairs — identical |
| `generate_regular` weight+scale pairing | ✓ as-is | auto-lookups `.scale` for `.weight` F8_E4M3 tensors |
| expert fusion (256 files → 1 tensor) | ✓ w/ patch | `generate_one_expert` hardcodes `layers.%d.` prefix → needs `mtp.%d.` |
| `parse_expert_tensor` | ✓ w/ patch | regex hardwired to `blk.%d.ffn_*_exps` → add `mtp.%d.` |
| `hf_name_for_regular` name-map | ✓ w/ patch | `layer_map` carries over; add DSpark branch + new entries |
| quantizers / GGUF writer / imatrix | ✓ as-is | unchanged |
| template-driven flow | ✓ w/ new template | no DSpark template exists → synthesize one |

## 3. Key iterative finding: DSpark ≠ old MTP-1 drafter

Comparing the **existing** `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (old MTP-1,
32 tensors, what `mtp_weights_bind` reads) against the **DSpark** HF inventory:

| | old MTP-1 | DSpark |
|---|---|---|
| drafter layers | 1 (`mtp.0`) | 3 (`mtp.0/1/2`) |
| target→drafter input | `e_proj`/`h_proj`/`enorm`/`hnorm` | `main_proj`/`main_norm` (mtp.0 only) |
| output stage | `hc_head` on mtp.0 | `markov_head` + `confidence_head` + `hc_head` + `norm` on **mtp.2 only** |
| block internals | MLA + MoE + HC | **same** (mini-DeepSeek-V4 block) |

So the old MTP GGUF is a *format reference* (KV conventions, quant recipe,
expert packing) but **not** a direct template, and Phase 4 needs a new DSpark
loader (the existing `mtp_weights_bind` is single-layer + wrong tensors). The
block-internal tensors reuse the `layer_map` nearly verbatim.

## 4. The plan (iteration 1 = plumbing proof)

**Converter patch (small):**
1. `expert_tensor`: add `bool is_mtp`.
2. `parse_expert_tensor`: also match `mtp.%d.ffn_%15[^_]_exps.weight`, set flag.
3. `generate_one_expert`: prefix `mtp.%d.ffn.experts.%d.%s` when `is_mtp`.
4. `hf_name_for_regular`: `mtp.N.<rest>` branch using a DSpark `mtp_layer_map`
   (mostly = `layer_map`) + DSpark-unique entries (`main_proj`, `main_norm`,
   `markov_head.markov_w1/w2`, `confidence_head.proj`, `norm`).

**Template GGUF (synthesized):** `build_dspark_template.py` writes GGUF header +
KV (`deepseek4.expert_count=256`, `deepseek4.mtp_layer_count=3`,
`dspark_block_size=5`, `dspark_target_layer_ids=[40,41,42]`,
`dspark_markov_rank=256`, `dspark_noise_token_id=128799`) + tensor metadata for
mtp.0/1/2 in ds4-renamed conventions with quant types mirroring the old MTP GGUF
(F32 norms/hc_*/sinks, Q8_0 attn proj, Q4_K experts, BF16 markov/confidence).
Zero data — the converter regenerates bytes from HF.

**Naming decision:** emit DSpark tensors in **ds4-renamed convention**
(`mtp.N.attn_q_a.weight`, `mtp.N.ffn_gate_exps.weight`) so the Phase-4 loader is
structurally parallel to `mtp_weights_bind`. This reuses `layer_map` and aids
debugging, at the cost of a small mapping table.

**Validation order (staged, avoids 11 GB download before plumbing is proven):**
1. Patch + build template → converter `--dry-run` against the DSpark template
   (plumbing proof, no HF data read).
2. Download shard 48 (mtp.2, 3.7 GB — the most distinct layer) → convert
   mtp.2-only template → check GGUF parses + tensor stats sane + a round-trip
   dequant cross-check vs a Python reference.
3. Full 3-shard download + conversion → `dspark.gguf`.

Step 2's cross-check (C dequant vs Python dequant on the same DSpark bytes) is
the real correctness gate for the conversion, since no DSpark reference GGUF
exists for `--compare-tensor`.

## 5. Phases 4-5 hooks this enables

- Phase 4: write `ds4_dspark_weights_bind` (parallel to `mtp_weights_bind`,
  3 layers + markov/confidence/main_proj), then the Metal drafter forward
  (port `DSparkBlock.forward`/`forward_embed`/`forward_head`). Validate against
  the Python `model.py` reference on a fixed prompt.
- Phase 5: rejection sampler over `metal_graph_verify_suffix_tops` logits.

The conversion output format is co-designed with the loader (iterative), so
Phase-4 findings feed back into Phase-3 naming/layout choices.
