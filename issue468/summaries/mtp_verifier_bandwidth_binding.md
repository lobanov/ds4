# MTP bulk draft verifier — memory-bandwidth vs compute binding

Date: 2026-07-05. Source-code audit of the shipped `--mtp` speculative path in
the ds4 Metal engine (this worktree). This is an architectural/roofline
assessment grounded in the verifier source; it is not a new measurement. The
empirical confirmation recipe is at the end.

## Question

The Issue-468 GOAL lists as its main risk that "ds4's verifier and
state-management overhead may erase the benefit of longer accepted prefixes."
This note audits the bulk draft verifier (the target side that verifies the MTP
draft suffix) to determine to what extent it is memory-bandwidth-bound versus
compute-bound, and what that implies for the speculative speedup gate.

## What the verifier is

Two code paths verify the MTP draft suffix with the **full target model**; both
are driven by the state machine `ds4_session_eval_speculative_argmax`
(`ds4.c:27167`):

1. **Batch verifier** `metal_graph_verify_suffix_tops` (`ds4.c:21117`) →
   `metal_graph_encode_layer_batch` (`19261`) → `_attention_batch` + `_ffn_batch`.
   Dense matmuls feed **M = n_tokens** (e.g. `ds4_gpu_matmul_f16_tensor(...,
   n_tokens)` at `18870`; Q8 attn/shared projections likewise), so dense weights
   are amortized over the suffix. Routed experts are dispatched as the **union of
   the 6-per-token selections** via `ds4_gpu_router_select_batch_tensor` +
   `metal_graph_stream_readahead_selected_experts_from_gpu` (`19000`) — only
   activated experts are loaded, not all 256. Self-described as the
   "production-shaped verifier attempt."

2. **N=2 exact verifier** `metal_graph_verify_decode2_exact` (`21218`) → calls
   `metal_graph_encode_decode_layer` (`14842`) **twice with M=1** (matmuls end
   `... g->flat_hc, 1`). This path streams the target weights twice with **zero
   amortization**. It is kept because batch kernels can flip greedy tokens; the
   exact path reproduces the decode numerics token-by-token.

The operative M (suffix length) is capped at `mtp_draft_tokens <= 16`
(`ds4.c:25562`) and is typically **2-4**.

## Verdict: memory-bandwidth-bound across the entire operating range

For a matmul of batch M against a quantized weight, arithmetic intensity is
approximately `2*M / bytes_per_elem`. Apple Silicon (M3-Max-class) roofline
balance point is `TFLOPS / bandwidth` ~= `14 TFLOPS / 400 GB/s` ~= **35 FLOP/byte**
(~70 for the FP16 matrix paths):

| weight category | storage | intensity @ M=2 | @ M=4 | @ M=16 |
|---|---|---:|---:|---:|
| Q8_0 dense (attn q_b / output_b, shared expert, **output head**) | ~1.06 B/elem | ~3.8 | ~7.5 | ~30 |
| IQ2_XXS routed experts | ~0.25 B/elem | ~16 | ~32 | ~128 |

Every Q8_0 dense matmul — including the ~545 MB output-head vocab projection
`[DS4_N_VOCAB=129280, DS4_N_EMBD=4096]` — sits **below the crossover for all
M <= 16**, and the IQ2_XXS experts are below it for the realistic M=2-4. The
verifier would need M ~= 18-37 (Q8_0 dense) to cross into compute-bound
territory, which exceeds the MTP draft cap. **It is bandwidth-bound to a
near-total degree at the suffix sizes actually used.**

## Three independent lines of evidence

1. **The decode path (which the verifier reuses) is engineered as
   bandwidth-bound.** It carries streaming infrastructure that only makes sense
   when weight streaming is the bottleneck: `metal_graph_stream_map_token` /
   `_map_decode_static_all` (static weight mapping), `metal_graph_stream_readahead_layer_decode`
   and `_readahead_selected_experts_from_gpu` (prefetch/compute overlap),
   `g->ssd_streaming`, and ROCm `rocm_graph_batch_selected_async_load` (overlap
   selected-expert loads with shared-expert compute). Readahead is not built for
   compute-bound kernels.

2. **Traffic is ~flat in K at small K.** A verify cycle streams once per cycle
   regardless of K: the output-head projection (~545 MB Q8_0), the per-layer
   dense weights (attn `q_b`/`output_b` dominate at ~35 MB each, plus the Q8
   shared expert ~27 MB — order ~100 MB/layer times `DS4_N_LAYER`, which is
   runtime-loaded from the model shape; V4 Flash is full target depth), and the
   union of activated experts (<= 6*K, typically overlapping). Because intensity
   is well below the crossover, K-way amortization barely reduces time:
   verifying 4 tokens costs roughly the same bandwidth as verifying 2, and
   roughly the same as one decode.
   Order-of-magnitude cross-check against the retained plain baseline
   (`summaries/plain_baseline_matrix.md`, ~38 tok/s => ~26 ms/token): dense
   streaming alone (~6 GB / 400 GB/s ~= 16 ms) accounts for most of a decode,
   i.e. decode itself is bandwidth-floor'd and the verifier shares that floor.

3. **The N=2 exact path is strictly worse on bandwidth.** It runs M=1 twice, so
   it pays the full target weight-streaming cost twice per cycle — pure
   bandwidth overhead, chosen only for greedy-exactness. The batch path is the
   bandwidth-optimized one.

## Implications for the speculative speedup

Because verification is bandwidth-bound and approximately flat in K at the
operative suffix sizes, the speculative gain is governed almost entirely by
*accepted tokens per cycle*, not by making verification cheaper:

```
speedup ~= accepted_tokens / (1 + draft_overhead/decode + verify_floor/decode)
```

where `verify_floor ~= one decode` (the bandwidth floor). This confirms the
GOAL's stated risk as **structurally real**: the verifier cannot be made cheap at
small K because it is bandwidth-floor'd; longer draft blocks only help once
`accepted_tokens` grows enough to amortize that floor. The prefix-1 attention
capture (`metal_graph_capture_prefix1_attn_state`, `ds4.c:13166`) is the
codebase's mitigation for partial-accept cost (avoiding a re-verify on a
1-of-2 accept), not for the bandwidth floor itself.

What would move the binding toward compute-bound (and thus reduce verify cost
per token):
- larger effective K (capped at 16 here, and limited by acceptance);
- multi-request batching server-side (amortizes the same weights across
  requests — the GOAL's "plausible server-side gains" lever);
- a verifier-specific resident hot-weight set (the static-decode-map + readahead
  already approximate this for decode).

## Empirical confirmation (no new instrumentation needed)

The code already exposes the relevant timers:

- `DS4_MTP_TIMING=1` (`ds4.c:27230`) prints per-cycle `draft=` vs `verify=` ms.
  A bandwidth-bound verify will show verify ~= a decode's time and approximately
  flat as draft length changes.
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` (`ds4.c:19320`) and
  `DS4_METAL_LAYER_STAGE_PROFILE` give per-stage encode/execute/read splits; on a
  bandwidth-bound path execute dominates and scales with bytes, not FLOPs.
- GPU counters (Instruments / AMD `gpu-counters`) would show ~full memory
bandwidth utilization at low ALU occupancy during verify.

A clean confirmation run: take one prompt from the baseline corpus, run
`./ds4 --metal -m <target> --mtp <mtp.gguf> --temp 0 -n 256` with
`DS4_MTP_TIMING=1`, and compare the reported `verify=` ms against the per-token
decode time and across different `--mtp-draft-tokens` values (1/2/4). If verify
is ~flat in K and ~= decode time, the bandwidth-floor conclusion holds.

## Scope and limits

- Assessed against the Metal backend on Apple Silicon (the baseline machine for
  this dossier). On CUDA/ROCm with HBM (~3 TB/s) the crossover intensity is
  similar (~40-60 FLOP/byte) but absolute verify time is ~5x lower; the binding
  verdict (bandwidth-bound at small K) is unchanged, the floor is just lower.
- Architecture constants are runtime-loaded from the model shape (`g_ds4_shape`,
  `ds4.c:292`); the intensity argument is robust to layer count and only
  weakly sensitive to exact bandwidth.
- This is the verifier only. Draft-side (MTP head) cost is separate and small by
  comparison (3-layer head vs full target); see the draft/verify timing from
  `DS4_MTP_TIMING` to confirm the split empirically.
