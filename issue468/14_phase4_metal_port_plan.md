# Phase 4 — Metal drafter port: de-risking + plan

After standing up the numpy oracle (issue468/dspark_oracle/, the Phase-4
regression target), audited ds4's existing Metal graph to scope the port.

## Key de-risk: the drafter reuses the BATCH encode path, not single-token decode

`metal_graph_encode_token_raw_swa` (single-token decode) is NOT the right reuse
base — DSpark drafts a BLOCK of 5 positions per cycle, not one token. The
drafter forward is structurally a batched encode.

The closest existing analog is `metal_graph_verify_suffix_tops` (ds4.c:21285):
it already drives a batch of N positions through all 43 layers via
`metal_graph_encode_layer_batch` (the batched MLA+MoE+HC path shared with
prefill). That is the exact machinery the drafter needs — batch_size = 5.

So the Metal drafter port is NOT a from-scratch attention rewrite. It reuses:
- `metal_graph_encode_layer_batch` for the 3 drafter blocks (MLA attn + MoE FFN
  + HC mixing, all already batched and validated in prefill/verify).
- `ds4_gpu_attention_output_q8_batch_*` (the batched MLA window path).
- ds4 MoE batch kernels (Q4_K experts), HC pre/post batch kernels.

## Genuinely-new code (the DSpark-specific surface)

1. **Input stage (mtp.0):** main_proj matmul (3*dim -> dim) + main_norm RMSNorm,
   both trivial via existing metal_graph_matmul_plain_tensor / rms_norm kernels.
   Then noise-block embed (token ids = [anchor, NOISE×4]) + HC-expand (broadcast
   to hc_mult copies). NOISE_TOK=128799.
2. **Anchor-KV threading:** the drafter's window slot for the anchor position is
   populated from main_x (target hidden), not from the drafter's own embedding.
   One small kernel or a reuse of the wkv+rmsnorm+rotary path on main_x, written
   to the batch KV cache at the anchor slot. The numpy oracle's
   dspark_attention_prefill is the reference.
3. **DSpark topk pattern:** get_dspark_topk_idxs = [0..min(win,start+1)-1] ++
   [win..win+block-1]. The batch attention kernel already accepts a topk index
   pattern; this is just a different fixed pattern than the target's window+compress.
4. **Output stage (mtp.2):** drafter hc_head (sigmoid reduce, small — reuse the
   hc_split kernel shape but with the sigmoid branch), output norm, then the
   SHARED target lm_head (already bound), then the SEQUENTIAL Markov head
   (markov_w1 embed + markov_w2 head, rank-256, 5 iterations — small matmuls).
   Confidence head is Phase 6, not needed for B2.

## Validation path (phase4-refcheck)

The numpy oracle (issue468/dspark_oracle/forward.py) IS the regression target
(CUDA reference blocked — issue468/12). After the Metal port emits draft tokens
for the same fixed prompt + main_hidden, compare:
  - draft token agreement (greedy argmax) Metal vs oracle — token-level match
    expected (dequant tolerance means argmax, not bit-exact logits).
  - Optionally logit correlation as a drift check.

Oracle runs in ~8.5s, deterministic, producing oracle_ref.npz. The Metal port
reads the same main_hidden capture + dspark.gguf.

## Plan order

1. Add drafter graph state to ds4_gpu_graph (batch buffers for block_size=5;
   reuse prefill_cap which is >>5).
2. Drafter input stage (main_proj + embed + HC-expand).
3. 3× drafter block via metal_graph_encode_layer_batch (parameterized for
   drafter weights + DSpark topk + anchor-KV).
4. Drafter output stage (hc_head + norm + lm_head + Markov).
5. CLI/research flag to run draft-only and dump tokens; compare to oracle.

The port is bounded because the heavy kernels (MLA, MoE, HC) are reused; only
the DSpark-specific I/O stages + the topk pattern + anchor-KV threading are new.
