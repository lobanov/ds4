# Inventory — upstream `deepseek-ai/DeepSeek-V4-Flash-DSpark` HF model

**Date:** 2026-07-09. **Source:** Hugging Face `deepseek-ai/DeepSeek-V4-Flash-DSpark`
(same architecture/config as the public `deepseek-ai/DeepSeek-V4-Flash`; the `-DSpark`
checkpoint additionally ships the DSpark MTP drafter). **Derived from:** the staged
safetensors in the Modal `huggingface-cache` volume — `config.json` +
`model.safetensors.index.json` + every shard's safetensors header (tensor → shape/dtype).
Probe: `run_lead04_modal` `probe_inventory`/`probe_inv2`.

This is the **native served-precision** target for Lead 04 (the "FP ceiling"). It is the
precision the DSpark drafter was distilled against — there is no BF16 release.

## Headline numbers

| | |
|---|---|
| Total tensors | **72,317** |
| Safetensors shards | **48** (`model-00000 … model-00047 .safetensors`) + `model.safetensors.index.json` |
| Total on-disk size | **166.88 GB** |
| Architecture | `DeepseekV4ForCausalLM`, `model_type = deepseek_v4` |
| Decoder layers | **43** |
| MTP/drafter layers | **3** (`mtp.0`, `mtp.1`, `mtp.2`) — the DSpark drafter |
| Hidden size | 4096 (`dim`) |
| HC (Manifold-constrained Hyper-Connections) | `hc_mult = 4` → HC residual `[4, 4096] = 16384` per token |
| Vocab | 129,280 |
| MoE | 256 routed experts, 1 shared expert, 6 activated/token, `moe_intermediate_size = 2048` |
| Attention | sparse MLA: `q_lora_rank=1024`, `kv_lora_rank=512`, `head_dim=512`, sliding window 128, c4a/c128a compression, DSA indexer |

**Memory-fit implication:** 166.88 GB does **not** fit a single H200 (141 GB) → TP=1 OOMs
(confirmed). Minimum is **TP=2 (2×H200 = 282 GB)**. Does not fit the DGX-Spark (~118 GB)
at all (native FP4/FP8 has no smaller release; ds4 only supports IQ2_XXS/Q2_K).

## Precision & encoding scheme

| safetensors dtype | count | what it is | where |
|---|---|---|---|
| `I8` (int8) | 35,328 | **FP4 weights, packed 2/byte** (MXFP4) | **routed experts** `w1/w2/w3.weight` (decoder + drafter) |
| `F8_E8M0` | 35,718 | **E8M0 microscale** (8-bit exponent, 0-bit mantissa) — per-block shared exponent | paired `.scale` for every FP4/FP8 weight |
| `F8_E4M3` | 390 | **FP8 E4M3** weights | attention (`wq_a/wq_b/wkv/wo_a/wo_b`), **shared experts**, drafter `main_proj`, indexer `wq_b` |
| `BF16` | 445 | full-precision bf16 | norms, `ffn.gate`, `embed`, `head` (lm_head), `norm`, markov_head, confidence_head, compressor/indexer (some) |
| `F32` | 433 | float32 | **HC-mixing weights** (`hc_attn_fn/hc_ffn_fn/hc_head_fn`, `[24/4, 16384]`), `attn_sink`, `gate.bias`, compressor `ape` |
| `I64` | 3 | int64 | `ffn.gate.tid2eid` `[129280, 6]` (token→expert routing table, 3 layers) |

### Expert quantization (the bulk of the weights)
- **Routed experts = MXFP4.** `layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight` is stored
  `I8` with logical shape = packed-2/byte FP4:
  - `w1.weight` `[2048, 2048] I8` → logical `[2048, 4096]` FP4 (gate: hidden 4096 → inter 2048)
  - `w3.weight` `[2048, 2048] I8` → logical `[2048, 4096]` FP4 (up)
  - `w2.weight` `[4096, 1024] I8` → logical `[4096, 2048]` FP4 (down: inter → hidden)
  - `.scale` `[2048, 128]` / `[4096, 64]` `F8_E8M0` → **32-element microblocks** with a shared E8M0 exponent (microscaling FP4). `n = 11,008 = 43 × 256` per pattern.
- **Shared experts = FP8 E4M3 + E8M0 scales.** `shared_experts.{w1,w2,w3}.weight` `F8_E4M3`,
  logical shapes match stored (1 byte/element); `.scale` ~128×128 blocks. `n = 43`.
- **Drafter (`mtp.*`) experts** mirror this exactly (256 routed FP4 + 1 shared FP8), `n = 768 = 3 × 256` per pattern.

### Non-expert precision
- **Attention (MLA):** `wq_a` `[1024,4096]`, `wq_b` `[32768,1024]`, `wkv` `[512,4096]`, `wo_a` `[8192,4096]`, `wo_b` `[4096,8192]` — all `F8_E4M3` + `F8_E8M0` scales; `q_norm`/`kv_norm` `BF16`; `attn_sink` `F32`. The sparse-MLA **compressor** + **indexer** sub-modules (`compressor.wgate/wkv`, `indexer.wq_b/weights_proj/compressor`) are BF16/F8 per layer (compressor on 41 layers, indexer on 21 layers).
- **HC mixing:** `hc_attn_fn`/`hc_ffn_fn` `[24, 16384]` F32, `hc_{attn,ffn}_scale` `[3]` F32, `hc_{attn,ffn}_base` `[24]` F32. (These are the Sinkhorn mixing parameters; `mix_hc = (2+hc_mult)*hc_mult = 24`.)
- **Final head reduction:** top-level `hc_head_fn` `[4, 16384]` F32, `hc_head_scale` `[1]`, `hc_head_base` `[4]` — reduces the 4 HC streams to 1 for the output hidden. `head.weight` (lm_head) `[129280, 4096]` BF16. `norm.weight` `[4096]` BF16 (final RMSNorm).
- **Embedding:** `embed.weight` `[129280, 4096]` BF16 (untied; `tie_word_embeddings=false`).

## Layout / tensor-name conventions

Names are `.`-nested. Decoder layers are `layers.0 … layers.42`; the drafter is `mtp.0/1/2`.

### Decoder layer `layers.{L}` (×43)
```
layers.{L}.attn_norm.weight            [4096]      BF16   (pre-attention RMSNorm)
layers.{L}.attn.                       (sparse MLA attention)
   wq_a / wq_b / wkv / wo_a / wo_b     weight F8_E4M3 + scale F8_E8M0
   q_norm / kv_norm                    [1024]/[512]  BF16
   attn_sink                           [64]          F32
   compressor.{wgate,wkv,norm,ape}     (c4a KV compression)           [41 layers]
   indexer.{wq_b,weights_proj,compressor} (DSA sparse index)          [21 layers]
layers.{L}.hc_attn_{fn,base,scale}     [24,16384]/[24]/[3]  F32  (HC mixing: attn)
layers.{L}.ffn_norm.weight             [4096]      BF16   (pre-FFN RMSNorm)
layers.{L}.ffn.gate.weight             [256,4096]  BF16   (MoE router)
layers.{L}.ffn.gate.bias               [256]       F32    (40 layers)
layers.{L}.ffn.gate.tid2eid            [129280,6]  I64    (routing table, 3 layers)
layers.{L}.ffn.shared_experts.{w1,w2,w3}.weight  F8_E4M3 + E8M0 scale
layers.{L}.ffn.experts.{E}.{w1,w2,w3}.{weight,scale}  FP4(packed I8)+E8M0  (E=0..255)
layers.{L}.hc_ffn_{fn,base,scale}      [24,16384]/[24]/[3]  F32  (HC mixing: ffn)
```
The per-token residual stream through a layer is the **HC residual `[hc_mult=4, 4096] = 16384`** — `hc_pre` reduces 4→1 (Sinkhorn-weighted) before attn/ffn, `hc_post` expands 1→4 after. The post-FFN HC residual at a layer = `after_ffn_hc` (ds4) / `_mtp_hidden_buffer` (vLLM) — the drafter's input source.

### DSpark drafter `mtp.{M}` (×3, the MTP/DSpark speculator)
```
mtp.{M}.main_proj.weight   [4096, 12288]  F8_E4M3   ← drafter INPUT projection (12288 = 3×4096)
mtp.{M}.main_norm.weight   [4096]         BF16      ← RMSNorm after main_proj
mtp.{M}.attn. + attn_norm               (same sparse-MLA attention as decoder)
mtp.{M}.hc_attn_{fn,base,scale}         F32        (HC mixing: attn)
mtp.{M}.ffn_norm + ffn.gate + ffn.shared_experts + ffn.experts.{E}  (MoE, same as decoder)
mtp.{M}.hc_ffn_{fn,base,scale}          F32        (HC mixing: ffn)
mtp.{M}.norm.weight         [4096]       BF16       (drafter final norm)
mtp.{M}.hc_head_{fn,base,scale}  [4,16384]/[4]/[1] F32  (drafter HC→1 head)
mtp.{M}.markov_head.markov_w{1,2}.weight [129280, 256] BF16  (DSpark Markov head, rank 256)
mtp.{M}.confidence_head.proj.weight      [1, 4352]  BF16  (DSpark confidence head)
```
**`main_proj = [4096, 12288]`** confirms the drafter consumes `main_hidden = [12288] =
concat(mean(hc_ffn_post[40]), mean(hc_ffn_post[41]), mean(hc_ffn_post[42]))` — i.e. the
mean over the 4 HC components, per layer, concatenated across the 3 target layers
(`dspark_target_layer_ids = [40, 41, 42]`). This is the load-bearing fact for the
representation mapping: **the drafter input is the mean of the per-layer HC residual
(16384 → 4096 per layer, ×3 → 12288)**, matching `dspark_oracle/build_main_hidden_from_captures.py`
(`arr.mean(axis=0)` over `[4,4096]`, concat 3 layers).

## Notes for the capture (Lead 04)

- The drafter reads layers **40/41/42** (last 3 decoder layers; n_layers=43) — `dspark_target_layer_ids=[40,41,42]` in `inference/config.json`, and `main_proj=[4096,12288]` confirms the 3-layer concat.
- The HC residual at a layer is `[4, 4096]` (4 DISTINCT components, confirmed empirically in the retained ds4 Q2 captures; `mean|hc−mean|/|mean| ≈ 0.71`). The drafter uses the **mean** over the 4.
- Precision is **native served precision** (FP4 routed experts + FP8 attention/shared + BF16 norms/head + F32 HC) — this IS the distillation-time precision; there is no higher-precision release. So "FP ceiling" = this checkpoint's precision.
- vLLM quant method = `deepseek_v4_fp8` (handles the MXFP4 experts via `expert_dtype=fp4` internally). Requires `kv_cache_dtype="fp8"` (fp8_ds_mla layout). Loads only where the 166.88 GB fits (≥TP=2 on H200).
- ds4 cannot run this checkpoint (ds4 supports only IQ2_XXS/Q2_K expert formats, not native FP4) — confirmed in `ds4.c` line 2364.

## Provenance

Derived 2026-07-09 by reading the staged checkpoint's `config.json` + all 48 safetensors
headers (tensor name/shape/dtype) via a Modal cpu probe (`probe_inventory.py` /
`probe_inv2.py` in `issue468/run_lead04_modal/`); aggregated by normalized tensor-name
pattern (`layers.{L}`, `experts.{E}`, `mtp.{M}`). Raw aggregation retained at
`/tmp/inv2.json`. Config cross-checked against the public `deepseek-ai/DeepSeek-V4-Flash`
`config.json` (identical architecture).
