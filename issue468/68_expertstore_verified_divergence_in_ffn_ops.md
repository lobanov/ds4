# ExpertStore Dequant Verified — Identical; Divergence in Non-Expert FFN Ops

Date: 2026-07-01. Twenty-eighth productionization handoff note. Follows
issue468/67 (exact-Q4 disproven). Doc-only research record.

## 1. ExpertStore dequant = standard dequant (IDENTICAL)

Compared the oracle's ExpertStore.expert(0) dequant to a direct
_dequant_q4_k call on the same Q4_K bytes for mtp.2 gate_exps expert 0:
- ExpertStore: min=-0.382599 max=0.378168 mean=0.000140
- Direct:      min=-0.382599 max=0.378168 mean=0.000140
- Max difference: 0.000000

IDENTICAL. The dequant algorithm is NOT the issue.

## 2. The full picture (every component verified)

| component | test | verdict |
|---|---|---|
| Q4_K GPU SIMD accumulation | CPU exact-Q4 = GPU (identical probe) | CORRECT |
| Q4_K dequant algorithm | ExpertStore = _dequant_q4_k (diff 0) | CORRECT |
| hc_pre(attn) | identical input (cos 1.0) | CLEAN |
| q_a + q_b | F32-input test (cos 1.0) | CLEAN |
| KV window | per-step (cos 1.0) | CLEAN |
| attention | CPU F32 = GPU F16 (identical) | CLEAN |
| routing | identical expert selection | CORRECT |
| SwiGLU clamp | 10.0 (matches oracle) | CORRECT |
| hc weights F16 | 0% round-trip error | EXACT |

## 3. Where the divergence ACTUALLY is

My DS4_DSPARK_EXACT_Q4 replaces ONLY the routed expert matmul in the FFN.
Everything else (hc_pre(ffn), hc_post(ffn), routing, SwiGLU, shared expert)
still runs on Metal. The oracle diverges from Metal by cos 0.996-0.998 for
layer 2 — but since the routed expert matmul is now proven identical (CPU
exact-Q4 = GPU), the divergence must be in the OTHER FFN operations:

- **hc_pre(ffn) F16 matmul**: the FFN-branch hc_pre uses
  ds4_gpu_matmul_f16_tensor on hc_ffn_fn weights. The Metal kernel truncates
  the F32 mid-block input to F16 before the MMA. If the mid-block state has
  values that F16 can't represent, the matmul diverges from numpy's F32×F16.
  This was NEVER tested on identical input separately.

- **hc_post(ffn) F16 matmul**: same class of F16 truncation issue.

- **Shared expert (Q8_0)**: uses Q8_0 matmul which was clean for q_a/q_b
  (cos 1.0), but might differ for the shared expert's specific weights.

## 4. The critical untested hypothesis: F16 input truncation in hc_pre(ffn)

The hc_pre(attn) was tested clean (cos 1.0) — but the attn-branch input
(batch_cur_hc = the block input) has different value distributions than the
ffn-branch input (batch_after_attn_hc = the mid-block state after attention +
hc_post(attn)). If the mid-block state has values near F16 precision limits,
the F16 truncation in hc_pre(ffn)'s matmul would diverge from numpy's F32
computation, specifically for layer 2's mid-block values.

This IS testable with the same CPU-side approach: implement an exact F32
hc_pre(ffn) and compare to Metal's F16 hc_pre(ffn) on the identical mid-block
input. But this is a MUCH smaller code change than the exact-Q4 MoE.

## 5. The +8.27% headroom: still potentially real

The oracle's MoE diverges from Metal by cos 0.996-0.998. If the divergence is
in hc_pre(ffn)'s F16 truncation (NOT the Q4_K experts), then fixing hc_pre(ffn)
to use F32 input (not F16-truncated) would close the gap — and this IS a
code-only fix (mirror the P1 markov pattern: read the F16 weights, convert to
F32, use F32 matmul). The Q4_K experts are innocent.

## 6. Next step: isolate hc_pre(ffn) F16 truncation

The exact-Q4 diagnostic proved the routed experts are correct. Now isolate
hc_pre(ffn): dump Metal's batch_after_attn_hc (the mid-block input to hc_pre)
and compare hc_pre(ffn) output on identical input between Metal (F16 matmul)
and the oracle (F32 matmul). If they differ for layer 2 → F16 truncation is the
root cause → fix hc_pre(ffn) to use F32 → re-measure.
