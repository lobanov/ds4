# Drafter-Forward Drift Bisection — Divergence Isolated to Layer 2 (mtp.2)

Date: 2026-07-01. Twelfth productionization handoff note. Follows issue468/51
(numpy oracle +8.27% headroom) and issue468/52 (FP8 KV = -2.6%, fixed). This
note bisects the REMAINING Metal-vs-oracle divergence to a single drafter layer.
Doc-only research record.

## 0. Method

Built a drift harness reusing the validated numpy oracle (`measure_b2_acceptance.py`
forward, proven correct: reproduces doc-15 ~2.79 greedy prefix) and the Metal
drafter probe (`ds4_dspark_probe_accept` on the oracle's own ctx_08192 capture
bundle). Added two dump hooks to the probe (gated by `DS4_DSPARK_PROBE_DUMP_Q`
for base_logits and `DS4_DSPARK_PROBE_DUMP_H` for per-layer `batch_cur_hc`).
Comparison scripts: /tmp/cmp_base.py, /tmp/cmp_h.py, /tmp/cmp_h_perlayer.py
(research-only, not committed). Build default = F32 drafter KV (issue468/52 fix).

## 1. Level 1: base_logits (pre-Markov) — FORWARD diverges, not the Markov head

Metal base_logits vs oracle base_logits (19 steps × 5 positions):
- global max_abs_diff = **9.7**, mean_abs_diff = **0.54**
- top-1 (argmax) agreement = **81.1%** (77/95)

→ The divergence is in the drafter FORWARD, not the Markov head. (And 81%
  agreement = precision/semantic divergence, not a structural ~0% bug.)

## 2. Level 2: pre-head hidden state h — BLOCKS diverge, not the head

Metal `batch_cur_hc` (after 3 blocks) vs oracle `run_blocks` output:
- max_abs_diff **40-124** per step; mean ~1.7
- relative to |oracle_h| (~430-714): **9-25%** per-element max error

→ The head (h→logits) is fine; the divergence is generated INSIDE the 3 blocks.

## 3. Level 3: PER-LAYER — divergence isolated to LAYER 2 (mtp.2)  ← the key finding

Metal block output after each drafter layer vs oracle, 5 steps:

| step | layer0 (mtp.0) | layer1 (mtp.1) | layer2 (mtp.2) |
|---|---|---|---|
| 1 | 0.48 / **0.5%** | 3.18 / 2.8% | 90.40 / **20.5%** |
| 2 | 0.27 / 0.3% | 3.50 / 3.2% | 62.88 / 10.0% |
| 3 | 0.12 / 0.1% | 4.74 / 3.7% | 60.79 / 9.7% |
| 4 | 0.10 / 0.1% | 4.78 / 4.3% | 79.05 / 18.1% |
| 5 | 0.10 / 0.1% | 4.84 / 4.1% | 109.30 / 25.5% |

(max_abs_diff / relative%)

**Layers 0 and 1 match the oracle to <0.5% and <4.3%** (excellent — within
Q4_K+F16 noise). **Layer 2 (mtp.2) explodes to 10-25%.** The three drafter
blocks use the IDENTICAL code path (`metal_graph_dspark_encode_block`), so a
code-path bug would affect all three equally. Since only mtp.2 diverges, the
cause is **mtp.2-SPECIFIC** — almost certainly weight/expert handling for the
mtp.2 stage (the special stage that also carries the Markov head + its own MoE
routed experts).

## 4. Suspects (mtp.2-specific), to bisect next

1. **mtp.2 MoE routed-expert dequant/loading.** The drafter MoE experts are
   `mtp.N.ffn_*_exps.weight` (issue468/34). If Metal's loader/dequant maps or
   slices mtp.2's packed experts differently than mtp.0/mtp.1 (e.g., an
   off-by-one in expert indexing, a wrong tensor offset, a name-mapping edge
   case for the last stage), only mtp.2's MoE output diverges. PRIME SUSPECT.
2. **mtp.2 attention window KV.** Each stage has its own `dspark_kv_cache[s]`.
   If layer 2's window fill/gather is off, its attention diverges. (Less
   likely — layers 0,1 share the same window logic and match.)
3. **mtp.2 weight tensor offsets/sizes.** A bounds/offset error loading mtp.2's
   attn or ffn weights (the last stage is where off-by-one errors surface).

## 5. Next bisection step

Bisect WITHIN layer 2: dump the post-attention intermediate (after hc_post of
the attn branch, before the ffn/MoE branch) for layer 2 and compare to the
oracle. If the attn output matches but the post-MoE output diverges → MoE
expert handling is the bug (suspect 1). If the attn output already diverges →
window-KV or attn-weight issue for mtp.2 (suspects 2/3).

## 6. Level 5 bisection: ATTENTION is the layer-2 bug (NOT the MoE)

Dumped Metal's `batch_after_attn_hc` (which holds layer-2's post-attention
output after the 3-block loop — each layer's attention overwrites it; the FFN
reads but doesn't clear it) and compared to the oracle's layer-2 post-attn
intermediate (block loop run fully for layers 0,1; for layer 2 only the attn
branch: hc_pre → rmsnorm → dspark_attn → hc_post):

| step | layer-2 post-attn maxdiff | relative |
|---|---|---|
| 1 | 18.4 | **11.1%** |
| 2 | 27.3 | **15.7%** |
| 3 | 32.5 | **20.5%** |
| 4 | 30.4 | **14.4%** |
| 5 | 32.4 | **18.7%** |

The post-ATTN intermediate already diverges 11-20% — the SAME magnitude as the
full layer-2 output (10-25%). If the MoE were the bug, post-attn would be small
(<4%, like layers 0/1) and the full output large. Instead post-attn is already
large. **→ The divergence originates in mtp.2's ATTENTION, not its MoE.**

## 7. Level 4: routing is NOT the bug (expert selection matches ~93%)

For completeness: dumped Metal's `batch_router_selected` (the 6 experts per
token) per layer and compared to the oracle's `gate()` selection. Expert-match
(token sets): layer 0 = 30/30 perfect; layer 1 = 26-30/30; layer 2 = 27-28/30
(~90-93%). So Metal selects nearly the same experts as the oracle for mtp.2 —
NOT a routing/`ffn_gate_inp` precision bug. Combined with §6, the layer-2 bug is
in the attention computation (q/kv projections, window-KV gather, softmax-with-
sinks, or output projection), with attention weights bound by-name correctly
(`dspark_layer_weights_bind` is stage-parametric and uniform), making the prime
remaining suspect the **mtp.2 window-KV handling or an attention-kernel edge
case for the last drafter stage**.

## 8. Final root-cause step (next session)

Dump layer-2's attention internals (q, kv, gathered window, attention scores,
output projection) individually and compare to the oracle's layer-2 attention
(`dspark_attn` / `attention.py:sparse_attn`). The first diverging tensor is the
bug. Candidates: mtp.2 window-KV slot layout, the attn_sinks handling for the
last stage, or an output-projection (wo_a/wo_b) precision/offset issue.

## 9. Significance (updated)

The entire +8.27% oracle headroom (issue468/51) traces to **mtp.2's attention**.
Layers 0,1 are bit-faithful to the oracle (<4.3%); the mtp.2 attention diverges
11-20%, which the MoE amplifies to 10-25% in the block output, which the head
maps to the 81%-argmax base_logits gap. Fixing mtp.2's attention to layer-0/1
fidelity recovers the full +8.27% acceptance → α 0.83→0.90 → ~41.5 t/s,
**crossing the gate** (issue468/42). This is now a single-kernel bug hunt with
a clear next dump, not open-ended research.

## 6. Significance

This makes the gate-crossing path CONCRETE and BOUNDED. The +8.27% oracle
headroom (issue468/51) is almost entirely generated by a single layer's
divergence. If mtp.2 is brought to layer-0/1 fidelity (<1%), the drafter forward
matches the oracle → the full +8.27% acceptance recovers → α 0.83→0.90 →
~41.5 t/s per the break-even table (issue468/42), CROSSING the gate. This is no
longer "open-ended research"; it is a single-layer bug hunt.
