# Phase 4 — DSpark drafter forward: oracle design + algorithm spec

Reference: `issue468/ref/inference/model.py` (DSparkBlock, forward_embed,
forward_head, Block, Attention, MoE) + `kernel.py` (hc_split_sinkhorn,
act_quant, sparse_attn). This doc is the faithful decomposition both the numpy
oracle (the regression oracle) and the ds4 Metal port implement.

## 1. Top-level: Transformer.forward_spec (model.py:922-932)

```
h, main_x = mtp[0].forward_embed(main_hidden, input_ids)   # [1, block_size, hc, dim], [1,1,dim]
for layer in mtp[0,1,2]:                                    # 3 stages, all compress_ratio==0
    h = layer(h, start_pos, input_ids, main_x)             # DSparkBlock.forward
output_ids, logits, confidence = mtp[2].forward_head(h, input_ids)
```
Decode path only (start_pos>0); prefill (start_pos==0) just caches the anchor KV
via `layer.attn(...)`. `block_size` (gamma) = 5. `hc_mult` = 4. `dim` = 4096.

## 2. forward_embed (mtp.0; model.py:851-857)
- `main_x = main_norm(main_proj(main_hidden))`   main_hidden [1,1,3*dim] -> [1,1,dim]
  (main_proj Q8_0 [dim, 3*dim]; main_norm BF16 RMSNorm)
- `draft_input_ids = [anchor, noise, noise, noise, noise]`  (block_size=5; idx0=anchor)
- `x = embed(draft_input_ids)` -> [1,5,dim]
- HC-expand: `x = x.unsqueeze(2).repeat(1,1,hc_mult,1)` -> [1,5,4,dim]
- Returns (x, main_x). The 3 blocks then each consume x and the (broadcast) main_x.

## 3. DSparkBlock.forward = Block.forward (model.py:725-736), attn cls = DSparkAttention
Two HC-wrapped sub-steps (attn, then ffn), identical structure:
```
residual = x
x, post, comb = hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)  # [1,5,dim]+state
x = attn_norm(x); x = attn(x, start_pos, main_x)                    # DSparkAttention
x = hc_post(x, residual, post, comb)                                 # back to [1,5,4,dim]
residual = x
x, post, comb = hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x = ffn_norm(x); x = ffn(x, input_ids)                               # MoE
x = hc_post(x, residual, post, comb)
```
- hc_pre/hc_post/hc_head: see §6 (hc_split_sinkhorn).
- attn_norm/ffn_norm/main_norm/norm: RMSNorm(eps=1e-6); BF16 weights.
- DSparkAttention: see §4. MoE: see §5.

## 4. DSparkAttention.forward (model.py:770-807) — decode (start_pos>0)
Window-only MLA (compress_ratio==0). NO indexer/compressor. Shapes: x [1,5,dim]
(the draft block), main_x [1,1,dim] (the anchor, one position).
```
# 1. Cache the ANCHOR's KV from main_x (target hidden), at window slot start_pos%win
main_kv = kv_norm(wkv(main_x))              # [1,1,head_dim=512]
apply_rotary(main_kv[..., -rope_dim:])      # rope on last 64 dims
self.kv_cache[start_pos % win] = main_kv    # window_size=128

# 2. Draft-block queries/KV from x (the 5 draft positions)
q = wq_b(q_norm(wq_a(x)))                   # [1,5,n_heads=64,head_dim=512]
q *= rsqrt(mean(q^2)+eps)                   # per-head RMSNorm
apply_rotary(q[..., -rope_dim:], freqs[start+1..start+6])
kv = kv_norm(wkv(x))                        # [1,5,512]
apply_rotary(kv[..., -rope_dim:])
kv = cat([kv_cache[0..win-1], kv], dim=1)   # [1, win+5, 512]

# 3. sparse attention over a fixed topk pattern (DSpark-specific)
topk_idxs = get_dspark_topk_idxs(win, block_size, start_pos)
#   = [0..min(win,start+1)-1]  ++  [win+0 .. win+block_size-1]
#   i.e. each of the 5 draft positions attends to: the full window (cached
#   anchors incl. the just-added one) + the 5 draft positions themselves.
o = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale=head_dim**-0.5)
apply_rotary(o[..., -rope_dim:], inverse=True)

# 4. low-rank grouped output projection (wo_a/wo_b, o_groups=8)
o = o.view(1,5,n_groups=8,-1)
o = einsum("bsgd,grd->bsgr", o, wo_a)      # wo_a BF16 [8,lora_o=1024,..]
x = wo_b(o.flatten(2))                      # -> [1,5,dim]
```
MLA specifics (Flash): q_a is a low-rank proj (dim->lora_q=1024), q_norm, q_b
(lora_q -> n_heads*head_dim). wo is grouped-low-rank (o_groups=8).
`attn_sink` is a per-head additive bias (n_heads F32 scalars).

> **Oracle decision: SKIP act_quant (FP8 activation sim).** The reference calls
> `act_quant(kv[..., :-rd], 64, ..., inplace=True)` (lossy FP8 round-trip on the
> non-rope dims). ds4's Metal MLA kernels do NOT FP8-sim activations. To be the
> clean ground truth closest to the Metal (BF16) port, the oracle uses pure F32
> throughout. Token agreement (greedy argmax) tolerates this; FP8-sim would make
> the oracle match the reference but diverge from the Metal port.

## 5. MoE (model.py:580-602) — gate + 256 routed experts + 1 shared
```
weights, indices = gate(x, input_ids)      # sqrtsoftplus, top-6, route_scale=1.5
  scores = softplus(linear(x, gate_inp)).sqrt()  # f32; + exp_probs_b bias for topk
  indices = topk(scores+bias, 6); weights = gather(scores,indices); norm to sum 1; *1.5
y = 0
for i in selected 6 experts: y += weights * expert_i(x)   # SwiGLU: w2(silu(w1(x))*w3(x))
y += shared_experts(x)                                     # same SwiGLU shape
```
Experts are the 256 packed Q4_K tensors; gate_inp/exp_probs_b F32; shared Q8_0.
Hash routing (first n_hash_layers=3) uses tid2eid — but drafter layers are 43+
(n_hash_layers=3 targets layers 0-2), so drafter uses score-based routing.

## 6. hc_pre / hc_post / hc_head + hc_split_sinkhorn (model.py:690-760, kernel.py:372)
hc_pre(x [b,s,hc,d], hc_fn, hc_scale, hc_base) -> (y [b,s,d], post [b,s,hc], comb [b,s,hc,hc]):
```
flat = x.flatten(hc,d)                                    # [b,s,hc*d]
mixes = linear(flat, hc_fn) * rsqrt(mean(flat^2)+eps)     # [b,s,(2+hc)*hc=24]
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=4, iters=20, eps=1e-6)
y = sum(pre[...,None] * x, dim=hc)                        # weighted reduce over hc copies
```
hc_split_sinkhorn (EXACT numpy port below): mixes [.,24], scale[3], base[24] ->
```
pre[j]      = sigmoid(mixes[j]      * scale[0] + base[j])      + eps        # j in [0,4)
post[j]     = 2*sigmoid(mixes[j+4]   * scale[1] + base[j+4])                # j in [0,4)
comb[j,k]   = mixes[j*4+k+8] * scale[2] + base[j*4+k+8]                     # j,k in [0,4)
comb = softmax(comb, -1) + eps
comb = comb / (col_sum + eps)            # one extra col-normalize
repeat sinkhorn_iters-1 times:
    comb = comb / (row_sum + eps)        # row-normalize
    comb = comb / (col_sum + eps)        # col-normalize
```
hc_post(x [b,s,d], residual [b,s,hc,d], post, comb) -> [b,s,hc,d]:
```
y = post[...,None]*x[...,None,:] + sum(comb[...,None]*residual, dim=hc_res)
```
hc_head(x, hc_head_fn, hc_head_scale, hc_head_base) -> [b,s,d] (drafter output-stage; SIGMOID not sinkhorn):
```
flat = x.flatten(hc,d); mixes = linear(flat, hc_head_fn)*rsqrt(...)
pre = sigmoid(mixes*scale + base) + eps
y = sum(pre[...,None]*x, dim=hc)         # mtp.2 only; hc_head_fn [4, hc_dim=16384]
```

## 7. forward_head (mtp.2; model.py:859-873)
```
x = hc_head(h, hc_head_fn, hc_head_scale, hc_head_base)    # [1,5,dim]
logits = head(norm(x))                                     # SHARED target lm_head [vocab,dim] -> [1,5,vocab]
output_ids = [anchor]                                      # block_size+1 = 6
for i in range(block_size=5):
    markov_logits, markov_embed = markov_head(output_ids[i])   # markov_w1 embed [rank=256], markov_w2 -> [vocab]
    logits[i] += markov_logits
    output_ids[i+1] = argmax(logits[i] / temp)             # temp=1.0 (greedy sample)
confidence = confidence_head(x, markov_embeds)             # proj([dim+256]) -> scalar per pos
```
The Markov head is SEQUENTIAL: each drafted token conditions the next via
`markov_w1(token) -> embed -> markov_w2(embed) -> logits_bias`. output_ids is
[anchor, d0, d1, d2, d3, d4]; markov_head reads output_ids[i] to bias logits[i].

## 8. Reuse map: ds4 Metal kernels vs new

| component | ds4 reuse | new |
|---|---|---|
| RMSNorm | ds4 rms_norm kernel (BF16) | — |
| MLA attention (window) | ds4 metal_graph_decode_attention (window path) | DSpark topk pattern + anchor-KV-from-main_x |
| MoE (gate+experts+shared) | ds4 MoE kernels (Q4_K experts) | — |
| hc_pre/hc_post/hc_head | ds4 hc_pre/post kernels (Sinkhorn) | hc_head (drafter output stage) |
| sparse_attn | ds4 sparse attention kernel | — |
| main_proj/main_norm | — | new matmul+rmsnorm (mtp.0 input stage) |
| embed (draft noise block) | ds4 embed kernel | noise-token fill + HC-expand |
| markov head (sequential) | — | new: small rank-256 matmuls + lm_head bias, sequential |
| confidence head | — | new (Phase 6 only; not needed for B2) |

The drafter is structurally a 3-layer mini-target + a small input/output stage.
Most of the compute reuses ds4's validated target kernels; the genuinely-new
surface is main_proj, the noise-block embed, the DSpark topk + anchor-KV, the
hc_head, and the sequential Markov head. The numpy oracle validates all of it.

## 9. Oracle inputs (from Phase-4 capture)
- `main_hidden_pos152.npy` [12288] = concat(mean_hc(layer40/41/42)) — the drafter input.
- anchor `input_ids` = the token ds4 generated at pos 152 (greedy argmax of target).
- weights: all 81 dspark.gguf tensors (F32/BF16/Q8_0/Q4_K), via gguf_loader.py.
- shared embed + lm_head: needed for forward_embed + forward_head. NOT in
  dspark.gguf — must be read from the target GGUF (token_embd.weight [dim,vocab],
  output.weight [dim,vocab]). The oracle reads these via the same loader (F16
  target tensors; ~2 GB for lm_head).
