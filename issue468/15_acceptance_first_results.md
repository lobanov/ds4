# DSpark Acceptance Measurement — FIRST RESULTS (reshape fix)

Date: 2026-06-28. Prompt: code_humaneval (HumanEval-style Python functions).
Target: DeepSeek-V4-Flash-IQ2XXS (ds4, Metal). Drafter: dspark.gguf (Q4_K experts,
Q8_0 attn, BF16 markov/confidence, F32 norms).

## Headline result: **57.9% greedy acceptance, 2.79 avg accepted prefix**

Measured across 19 decode steps (positions 153-171), block_size=5, using the
numpy drafter oracle with IQ2XXS target hidden states at layers [40,41,42].

| step | pos | KV entries | match/5 | prefix | first mismatch |
|---|---|---|---|---|---|
| 1 | 153 | 2 | 4/5 | 3 | pos 4 (2689 vs 15255 " Python") |
| 2 | 154 | 3 | 3/5 | 3 | pos 4 (4181 vs 6177 " functions") |
| 3 | 155 | 4 | 3/5 | 3 | pos 4 (16 vs 305 " and") |
| 4 | 156 | 5 | 4/5 | 4 | pos 5 (270 vs 5085) |
| 5 | 157 | 6 | 1/5 | 1 | pos 2 (16 vs 305) |
| 10 | 162 | 11 | 3/5 | 3 | pos 4 (1082 vs 9979) |
| **15** | **167** | **16** | **5/5** | **5** | **PERFECT** |

Distribution of accepted prefix length: {0:2, 1:3, 2:2, 3:5, 4:4, 5:3}

## Speedup projection (preliminary)

With avg_prefix=2.79 + 1 bonus token = 3.79 accepted tokens per draft cycle:
- Phase-1 batch verify(L=5) ≈ ~20ms (interpolating: 36ms at L=1, 12ms/token at L=8)
- Plain decode ≈ 28ms/token
- Draft cost estimate: 3 drafter layers × 5 tokens ≈ 10-15ms (7% of target layers
  but 5× batch; dominated by 256-expert MoE)
- **Conservative: (15ms draft + 20ms verify) / 3.79 = 9.2ms/token → 3.0x speedup**
- **Pessimistic: (30ms draft + 25ms verify) / 3.0 = 18.3ms/token → 1.5x speedup**
- **Either way: well above the 20% (1.2x) gate.**

## Root cause of the earlier 0% acceptance

The numpy oracle's gguf_loader reshaped dequantized data using `np.reshape(dims)`
(C-order, last dim fastest), but GGUF ne has ne[0] as fastest-varying. For ALL
2D+ tensors this scrambled the data. Fix: reshape to `reversed(dims)` = HF/torch
order. After fix, every matmul uses standard `x @ W.T` and embeddings `W[token]`.

Validated: Markov bigram " and"(305) → top5 includes " the"(270) (textbook
bigram); logits range [-24, 15] (was garbage [-105, 84]).

## Key findings

1. **IQ2XXS hidden states ARE usable by the DSpark drafter.** The 21.8% drift
   between IQ2XXS and Q4K-37-42 (measured separately) is tolerable; the drafter
   produces strong draft tokens with 2-bit-quantized hidden states.
2. **Acceptance is strong even with minimal KV cache (2-20 entries).** The real
   decode scenario has up to 128 window entries, which should improve acceptance
   further.
3. **The sequential Markov head is critical.** The drafter's draft quality comes
   substantially from the rank-256 Markov bigram head conditioning each position
   on the previous draft token.
4. **Greedy acceptance is an upper bound for B2 rejection sampling.** The actual
   B2 acceptance at temp=1.0 will be somewhat different (probabilistic), but the
   greedy upper bound of 2.79 avg prefix is very promising.

## Next steps

- phase4-forward: Metal drafter port (reuse batch encode kernels; the algorithm
  is now validated by the numpy oracle)
- phase4-refcheck: Metal draft tokens vs oracle (token agreement)
- Phase 5/6: B2 rejection sampling + end-to-end measurement
