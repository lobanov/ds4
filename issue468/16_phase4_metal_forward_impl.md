# Phase 4 — Metal drafter forward: implementation spec

Status: allocation wiring DONE (enable_dspark threaded through metal_graph_alloc_raw_cap;
drafter GPU buffers allocated; --dspark loads without crash). INPUT STAGE DONE +
VALIDATED (main_proj+main_norm match numpy oracle EXACTLY, max err 0.0). The 3
DSparkBlocks need a genuinely-new non-causal sparse-attention kernel (recon below);
the output stage is pending. The forward FUNCTION is the remaining work, with the
validated numpy oracle (issue468/dspark_oracle/) as the spec to port.

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

## Stage 1: forward_embed (mtp.0 input stage) — DONE + VALIDATED
Implemented as `metal_graph_dspark_input_stage` (ds4.c):
- main_x = rmsnorm(main_hidden @ main_proj, main_norm)  [ds4_gpu_matmul_q8_0_tensor
  (main_proj Q8_0, in_dim=3*dim, out_dim=dim) + ds4_gpu_rms_norm_weight_tensor]
- draft block = embed([anchor, NOISE×4]) + HC-expand via metal_graph_upload_prompt_embeddings_hc
  (drafter shares target token_embd) -> batch_cur_hc [block,hc,dim]
Validation (DS4_DSPARK_PROBE_INPUT): main_x matches oracle EXACTLY (max err 0.0,
corr 1.0) on pos152 code-prompt capture. Two prerequisite bugs found+fixed:
  (a) dspark model map not finalized -> added accelerator_cache_model_tensors
      (parallel to MTP); without it GPU matmuls read zeros.
  (b) BF16 norms vs F32-only rmsnorm kernel -> converted main_norm/mtp.2.norm to
      F32 in build_dspark_template.py + updated dspark_weights_validate_layout.
      (markov_w1/w2/confidence_proj still BF16 — verify their kernels at stage 3.)

## Stage 2: Three DSparkBlocks — attention sub-block DONE + VALIDATED
Each block: hc_pre -> attn -> hc_post -> hc_pre -> ffn -> hc_post.
- hc_pre/post/ffn: REUSABLE from metal_graph_encode_layer_batch.
- attn sub-block: IMPLEMENTED as metal_graph_dspark_encode_attention + VALIDATED
  (after_attn_hc corr=0.99997 vs oracle on pos152; F16 hc_fn residual). Reads
  batch_cur_hc, writes batch_after_attn_hc. Two bugs found+fixed:
   (a) hc_attn_fn/hc_ffn_fn must be F16 (ds4_gpu_matmul_f16_tensor reads F16);
       converted in build_dspark_template.py. NaN otherwise.
   (b) DSpark MLA shares rope dims between k and v: the FlashAttention kernel
       rotates k but does NOT inverse-rotate the output, so an INVERSE rope_tail
       (inverse=true) on batch_heads at the query positions is required before
       the output projection (matches model.py apply_rotary(o,inverse=True)).
       Without it corr=0.85; with it corr=0.99997.
  KV layout: anchors [0..n_real-1], this-step anchor [n_real], draft [n_real+1..n_real+5]
  -> contiguous gathered [0..n_real+5] for the noncausal attention.
- ffn sub-block: REUSE metal_graph_encode_layer_ffn_batch (reads batch_after_attn_hc,
  writes batch_next_hc). NOT yet wired into the drafter block loop.
- REMAINING for stage 2: chain 3 blocks (attn sub-block + ffn_batch), swap
  batch_cur_hc <- batch_next_hc between layers. Ring-handling for n_real near
  WIN=128 (currently sequential layout; probe uses small n_real).

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
