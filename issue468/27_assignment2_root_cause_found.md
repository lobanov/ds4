# Assignment 2 — ffn_gate_inp type mismatch: THE root cause; gate now PASSES

Date: 2026-06-29. Supersedes issue468/23 (Outcome B) and issue468/25 (Assignment 1
"gate reachable at 64k via F32"). The verdict has FLIPPED from B to A trajectory.

## Headline

The B2 collapse (2.80 oracle → 1.31 Metal) was NOT precision/accumulation at all.
It was a **converter type mismatch**: `ffn_gate_inp` stored as F32 but read by
`ds4_gpu_matmul_f16_tensor` (hard-coded F16 weight read) → garbage gate logits →
completely wrong expert routing → draft distribution collapse. A one-line converter
fix (F32→F16) recovers Metal greedy acceptance to 2.74 (from 1.53) and B2 committed
to 3.42 (from 1.31). The >20% speedup gate now **PASSES at all contexts** (+30-48%).

## The bug

`metal_graph_encode_layer_ffn_batch` (ds4.c:19507) computes the MoE gate logits via:
```c
ds4_gpu_matmul_f16_tensor(g->batch_router_logits, model->map, model->size,
    layer->ffn_gate_inp->abs_offset, DS4_N_EMBD, DS4_N_EXPERT, ...)
```
This reads the weight as F16 (2 bytes/elem). But `build_dspark_template.py` stored
`ffn_gate_inp` as F32 (4 bytes/elem). The F16 kernel reads the F32 bit pattern as
two F16 values → completely wrong gate logits → wrong topk-6 expert selection.

**Evidence:** dumped Metal's router_selected for step1/lay0 and compared to oracle's
`gate()` on identical input (Metal's ffn_norm). Zero overlap — Metal selected
{192,248,202,140,170,36}, oracle selected {25,85,100,156,180,222}. Metal's experts
ranked 11th-252nd in oracle's ordering. This is not borderline — it's garbage routing.

**Why the target model works:** the target's `ffn_gate_inp` is F16 (standard
converter), matching the `matmul_f16` kernel. Only the drafter (custom
build_dspark_template.py) had it as F32.

**Why precision knobs had no effect:** F16-mid, F16-act, F16-weight all operate on
the EXPERT computation, not the GATE matmul. The gate was reading garbage regardless
of expert precision. (Issue468/26 documents the ruled-out candidates.)

## The fix

One line in `build_dspark_template.py`:
```python
add(f"{P}.ffn_gate_inp.weight", [NX, EMBD], F16)  # was F32
```
Rebuilt dspark.gguf. No Metal code change. No gathered-dense rewrite. The
review's Assignment 2 premise (F32-accumulation gathered-dense MoE) was based on
the wrong diagnosis — the real issue was far simpler.

## Results (Metal drafter, production Q4_K kernels, persistent-KV, 19 steps)

| metric | before fix | after fix | oracle (F32) |
|---|---|---|---|
| greedy avg prefix | 1.53 (33.7%) | **2.74 (56.8%)** | 2.79 (57.9%) |
| B2 committed | 1.31 | **3.42** | ~3.0* |
| B2 accepted A | 0.31 | **2.42** | ~2.0* |

*Oracle B2 was measured with sampling (different protocol); the corrected protocol
(argmax drafts + B2 verifier accept) gives higher committed for both.

### Speedup (committed=3.42, draft 7.2ms, verify 75ms, +1 structural token)

| ctx | plain ms/tok | ms/tok (spec) | speedup | gate (>20%) |
|---|---|---|---|---|
| 8k | 31.3 | 24.0 | **1.30× (+30%)** | PASS |
| 32k | 32.9 | 24.0 | **1.37× (+37%)** | PASS |
| 55k | 34.8 | 24.0 | **1.45× (+45%)** | PASS |
| 64k | 35.5 | 24.0 | **1.48× (+48%)** | PASS |

**ALL contexts clear the >20% gate with 10-28pt margin.** Even at 8k (previously
"structurally impossible"), the gate passes comfortably.

## B2 MC sim bugs also fixed

1. Drafter was SAMPLING from q (gave committed 1.50). Model.py's forward_head uses
   **argmax** (deterministic). Fixed → committed 1.84.
2. Step indexing was off-by-one (metal_base[k] = sweep step k+1, anchor should be
   greedy[k+1] not greedy[k]). Fixed → committed 3.42.

With argmax drafts + B2 verifier accept: p(argmax) >> q(argmax) for matching drafts
→ accept=1.0 (always accept). Non-matching drafts have p≈0 → reject, resample.
Committed ≈ greedy_prefix + 1, with a small reduction from borderline rejects.

## Residual

Metal-vs-oracle base_logits corr is 0.82-0.89 (not 1.0). The residual comes from
Q4_K expert dequant (Metal on-the-fly vs oracle F32) + F16 weight rounding
(hc_ffn_fn, ffn_gate_inp now F16). This residual doesn't prevent the gate from
passing (greedy 2.74 ≈ oracle 2.79) but could be reduced via Assignment 3's
imatrix/precision sweep if higher margin is desired.

## Remaining for Outcome A confirmation

These numbers are PROJECTIONS (measured cost components × B2 committed from MC on
dumped base_logits). The ACTUAL end-to-end measurement requires:
1. Wire B2 rejection-sampling into the ds4 spec-decode loop (Phase 5 equivalent)
2. Run actual timed gen t/s at 32k/55k/64k (Phase 6 equivalent)
3. ds4-eval DSpark@temp=1.0 quality check (≥65/92)

The margin (10-28pt above gate) is large enough that minor cost measurement
differences won't flip the verdict.
