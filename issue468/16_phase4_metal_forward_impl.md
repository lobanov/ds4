# Phase 4 — Metal drafter forward: implementation spec

Status: allocation wiring DONE (enable_dspark threaded through metal_graph_alloc_raw_cap;
drafter GPU buffers allocated; --dspark loads without crash). The forward FUNCTION
itself is the remaining substantive work — this doc is the precise plan, with the
validated numpy oracle (issue468/dspark_oracle/) as the ready spec to port.

## Validated foundation (why this is an implementation task, not research)

The numpy oracle (issue468/dspark_oracle/forward.py) runs the complete DSpark
forward_spec and produces **57.9% greedy acceptance, 2.79 avg accepted prefix**
(issue468/15_acceptance_first_results.md). The algorithm is proven correct; the
Metal port is a faithful translation to GPU kernel calls reusing ds4's primitives.

## GPU buffers (allocated, in ds4_gpu_graph)
- dspark_main_hidden [3*DS4_N_EMBD]  — concat mean-hidden of target L40/41/42
- dspark_main_x [DS4_N_EMBD]        — projected+normed drafter input
- dspark_draft_hc [5*hc_dim]        — HC-expanded 5-position draft block
- dspark_kv_cache[3] [DS4_N_SWA*DS4_N_HEAD_DIM] — per-layer window KV

## The forward function: metal_graph_dspark_eval_draft

Modeled on metal_graph_eval_mtp_draft_from_hc (ds4.c:20092, the existing MTP-1
drafter forward). Stages (each maps to numpy oracle + ds4 GPU primitives):

### 0. main_hidden capture (in decode loop, at layers 40/41/42)
Replace the disk-dump capture point with a GPU mean+store into dspark_main_hidden.
cur_hc at layer L is [HC, EMBD]; mean over HC -> [EMBD]; concat L40/41/42 -> [3*EMBD].
Needs a small mean-over-hc kernel OR reuse: ds4_gpu has hc reduce ops; check
ds4_gpu_hc_weighted_sum_* (uniform weights = mean). Correctness-first: can
readback+CPU mean+upload initially.

### 1. forward_embed (mtp.0 input stage) — trivial primitives
- main_x = rmsnorm(main_hidden @ main_proj.T, main_norm)  [main_proj Q8_0, matmul_q8_0 + rms_norm_weight]
- draft tokens = [anchor, NOISE×4] (block_size=5)
- x = embed[draft_tokens]  [ds4_gpu_embed_tokens_hc or batch embed]
- HC-expand: repeat to [5, HC, EMBD]  [ds4_gpu_repeat_hc_tensor]

### 2. Three DSparkBlocks — the heavy piece
Each block: hc_pre -> attn -> hc_post -> hc_pre -> ffn -> hc_post.
- hc_pre/post: ds4_gpu_hc_split_sinkhorn + hc_weighted_sum (batched variants used in
  metal_graph_encode_layer_attention_batch). Reuse directly.
- attn (window MLA, compress_ratio==0): the batched window path in
  metal_graph_encode_layer_attention_batch. DSpark-specific changes:
  (a) anchor KV slot populated from main_x (rmsnorm + wkv + rotary), not from draft embed;
  (b) DSpark topk pattern [0..n_real-1] ++ [win..win+4] (get_dspark_topk_idxs).
  The batch attention kernel accepts a topk index tensor; pass the DSpark pattern.
- ffn (MoE): ds4 MoE batch kernels (Q4_K experts, gate, shared). Reuse metal_graph_encode_layer_ffn_batch.

### 3. forward_head (mtp.2 output stage)
- hc_head: drafter output-stage HC reduce (sigmoid, not Sinkhorn). Small; needs a
  new tiny kernel or reuse hc_weighted_sum with sigmoid weights. See numpy hc_head.
- norm: rms_norm_weight
- logits = lm_head(norm(hc_head(x)))  — SHARED target output.weight (reuse output head matmul)
- Sequential Markov head (5 iters): for i in 0..4: bias = markov_w2(markov_w1(out[i])); argmax.
  markov_w1 is an embedding lookup [vocab,256], markov_w2 a matmul [256,vocab]. Rank-256, tiny.
  This is sequential (5 small matmuls) — fast on Metal.

## Reuse map (what's genuinely new vs reused)
| stage | reuse ds4 | new |
|---|---|---|
| main_hidden capture | hc reduce ops | mean-over-hc store into buffer |
| main_proj + main_norm | matmul_q8_0 + rms_norm | — |
| noise-block embed | embed + repeat_hc | NOISE_TOK fill |
| 3× block attn/ffn/hc | batched encode kernels | DSpark topk + anchor-KV-from-main_x |
| hc_head | hc ops | sigmoid-variant reduce |
| norm + lm_head | output head matmul | — |
| Markov head | small matmuls | sequential loop (5 iters) |

The genuinely-new GPU code is small: the mean-over-hc capture, the DSpark topk
pattern construction, anchor-KV threading, hc_head sigmoid, and the Markov loop.
The heavy kernels (MLA, MoE, Sinkhorn HC) are all reused from the batched encode path.

## Validation (phase4-refcheck)
After the Metal forward emits draft tokens for the fixed code prompt + captured
main_hidden, compare against the numpy oracle's tokens (oracle_ref.npz). Token
agreement (greedy argmax) expected; dequant tolerance allowed. The numpy oracle is
the regression target (CUDA model.py.forward_spec blocked — issue468/12).

## CLI hook
A research-only env flag (e.g. DS4_DSPARK_DUMP_DRAFT) to run the drafter forward
after each decode step and dump draft tokens, for the refcheck. No permanent flag.
