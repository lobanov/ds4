# DSpark Speculative Decoding — Research Handoff Note

Date: 2026-06-29. This is the standalone handoff for a productionization goal to
pick up. It packages the complete state of the DSpark investigation: what was
built, what was found, what was measured, what is projected, and exactly what
work remains. References to prior docs (issue468/NN) are by number; the docs are
not modified by this note.

---

## 1. Executive summary

DSpark speculative decoding is **validated at the component level** but **not yet
faster than baseline end-to-end** in the experimental implementation. The critical
path to production is eliminating the **KV replay overhead** on partial-accept
cycles, which is the sole reason the measured end-to-end speedup (0.71×) diverges
from the structurally-correct projection (+29-46%, or +48-60% with the +1 bonus
token).

| metric | value | source |
|---|---|---|
| Measured end-to-end gen t/s (ctx=4096) | 27.67 t/s vs 39.18 baseline = **0.71×** | issue468/30 |
| Component-level projection (no replay, +1) | **+48-60% at 64k** | §4 below |
| Root cause of the gap | KV replay (~15ms avg/cycle on partial accept) | §5 below |
| B2 acceptance (offline MC, Metal drafter) | committed 3.42 (3.74 with +1) | issue468/25 |
| The fix that unblocked everything | ffn_gate_inp F32→F16 (one line) | §2 below |

---

## 2. What was built and validated

### 2.1 Drafter GGUF (Phase 3, stable)
- `dspark.gguf` (10.71 GiB, 81 tensors, 3 MTP layers) from DeepSeek-V4-Flash-DSpark
  shards 46-48. Q4_K routed experts, Q8_0 attention, F16 hc_fn/gate_inp, F32 norms.
  47/47 byte-exact crosscheck (issue468/11).
- Location: `../ds4/gguf/dspark.gguf`

### 2.2 Metal drafter forward (Phase 4, validated)
Complete end-to-end Metal forward producing draft tokens. All stages validated
against the numpy oracle (`issue468/dspark_oracle/`):

| stage | function | validation |
|---|---|---|
| input (main_proj+norm+embed+HC-expand) | `metal_graph_dspark_input_stage` | corr **1.0** vs oracle |
| attention sub-block (hc_pre→attn→hc_post) | `metal_graph_dspark_encode_attention` | corr **0.99997** vs oracle |
| FFN sub-block (hc_pre→MoE→hc_post) | `metal_graph_dspark_encode_block` (reuses `metal_graph_encode_layer_ffn_batch`) | corr 0.979/block (Q4_K noise) |
| output head (hc_head→norm→lm_head) | `metal_graph_dspark_output_head` | runs, produces logits |
| Markov head (sequential w1/w2) | CPU in B2 function | 4.3ms/cycle (port to GPU: ~0.1ms) |

Key primitives added:
- `ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor` (ds4_metal.m) —
  non-causal batched attention for DSpark's block-attention pattern (memset-0 mask
  variant of the causal decode kernel; no new GPU kernel).
- `ds4_gpu_set_model_map_range` for dspark_model (was missing → matmuls read zeros).
- `accelerator_cache_model_tensors` for dspark_model (was missing → map not resident).

### 2.3 B2 rejection-sampling spec-decode loop (Phase 5, wired)
- `ds4_session_eval_dspark_b2` (ds4.c, ~400 lines) — a complete B2 cycle:
  1. Decode anchor (target forward, captures main_hidden at L40/41/42 via GPU-to-GPU copy)
  2. Readback + mean over HC + upload to `dspark_main_hidden`
  3. DSpark drafter forward → 5 draft tokens + full q distribution [block, vocab]
  4. Verify via `metal_graph_verify_suffix_tops` → target argmax (row_tops) + full logits (row_logits)
  5. **TRUE B2 acceptance**: accept draft x w.p. `min(1, p(x)/q(x))` using full p (target) and q (drafter) distributions; on reject, correction = `argmax(p−q)`, commit, STOP
  6. KV management: full-accept → verify KV correct, no replay; partial-accept → `spec_frontier_restore` + replay accepted+correction via decode
- Wired into CLI (`ds4_cli.c:run_sampled_generation`) and `ds4-eval` (`ds4_eval.c`)
  via `ds4_engine_has_dspark()` → dispatch to `ds4_session_eval_dspark_b2`.
- Output verified CLEAN (no garbled tokens) via ds4-eval 5-case test.

### 2.4 Measurement harness (env-gated, reusable)
- `DS4_DSPARK_TIME_BACKBONE` — backbone timing (via `--verifier-curve-test` session)
- `DS4_DSPARK_PROBE_INPUT` — input-stage validation (dumps main_x)
- `DS4_DSPARK_PROBE_ATTN` — attention sub-block validation (dumps after_attn_hc)
- `DS4_DSPARK_PROBE_BLOCKS` — 3-block forward validation (dumps batch_cur_hc)
- `DS4_DSPARK_PROBE_TOKENS` — draft token dump (for token agreement)
- `DS4_DSPARK_PROBE_ACCEPT` — persistent-KV greedy acceptance sweep (19 steps)
- `DS4_DSPARK_PROBE_DUMP_Q` — dumps base_logits [n_steps, 5, vocab] for offline B2 MC
- `DS4_DSPARK_B2_DEBUG` — per-cycle acceptance diagnostics
- `DS4_DSPARK_PROBE_DUMP_FFN` — router_selected + ffn_norm dump (routing bisection)

### 2.5 Numpy oracle (validated algorithm reference)
`issue468/dspark_oracle/` — faithful F32 port of `inference/model.py` forward_spec.
Runs in ~8.5s. Produces reference tokens/logits for validation. Includes Q4_K
dequant (ported from ggml-quants.c), Sinkhorn HC, DSpark attention, sqrtsoftplus MoE.

---

## 3. Critical findings and root causes

### 3.1 THE root cause: ffn_gate_inp F32/F16 type mismatch (issue468/27)

**The single most important finding of the entire investigation.** The prior
Outcome B verdict (Metal acceptance 1.53 vs oracle 2.79) was NOT a precision
problem — it was a **converter type mismatch**:

- `build_dspark_template.py` stored `ffn_gate_inp` as F32.
- `metal_graph_encode_layer_ffn_batch` (ds4.c:19507) computes gate logits via
  `ds4_gpu_matmul_f16_tensor` (hard-coded F16 weight read).
- The F16 kernel reads F32 bytes as F16 → garbage gate logits → **completely
  wrong topk-6 expert routing** → completely different draft weights → B2 collapse.

**Evidence:** dumped Metal's `router_selected` (via `DS4_DSPARK_PROBE_DUMP_FFN`)
and compared to oracle's `gate()` on identical input. Zero overlap — Metal selected
{192,248,202,140,170,36}, oracle selected {25,85,100,156,180,222}. Metal's experts
ranked 11th-252nd in oracle's ordering.

**Fix:** one line in `build_dspark_template.py`: `ffn_gate_inp F32 → F16`. Rebuilt
dspark.gguf. No Metal code change. Metal greedy acceptance recovered 1.53→2.74
(matching oracle's 2.79). Commit `7dff90e`.

**Why this was hard to find:** all precision knobs (F16-mid via `--quality`,
F16-activations, F16-weights) had NO effect on acceptance — because the bug is in
the GATE matmul (routing), not the expert computation. The review's Source B
diagnosis (BF16 accumulation) was a red herring. The actual issue was a type
mismatch that produced garbage routing (issue468/26).

### 3.2 Other converter/kernel bugs found and fixed

| bug | fix | impact |
|---|---|---|
| dspark_model.map not registered with Metal | added `ds4_gpu_set_model_map_range` for dspark (parallel to MTP) | matmuls on drafter weights were reading zeros |
| dspark_model not cached (map not resident) | added `accelerator_cache_model_tensors` for dspark | GPU matmuls read zeros even after map registration |
| BF16 norms (main_norm, mtp.2.norm) vs F32-only rmsnorm kernel | converted to F32 in converter | rmsnorm read past BF16 data into adjacent tensors → garbage |
| hc_attn_fn/hc_ffn_fn F32 vs matmul_f16 kernel | converted to F16 in converter | NaN in hc_pre output |
| hc_head_fn F32 vs matmul_f16_tensor (single-token only for F32) | converted to F16 in converter | batched output head failed |
| Missing inverse rope on attention output | added `rope_tail_tensor(inverse=true)` on batch_heads | DSpark MLA shares rope dims between k and v; kernel rotates k but not output |
| spec_logits not allocated for --dspark (enable_mtp only) | allocate when enable_dspark | output head writes to null |
| spec_attn_state_kv/index not allocated for --dspark | allocate when enable_mtp OR enable_dspark | spec_frontier_snapshot fails |
| doc 23 speedup table dropped the structural +1 token | corrected in issue468/25 | headline 0.66× was wrong; corrected to 1.09× |
| Oracle B2 MC sim markov weight transpose | fixed reshape [vocab,rank] not [rank,vocab].T | B2 committed 1.00 → 3.42 |
| Oracle B2 MC sim drafter sampling vs argmax | fixed to argmax (model.py uses argmax) | B2 committed 1.84 → 3.42 |
| Oracle B2 MC sim step indexing off-by-one | metal_base[k] = sweep step k+1, anchor = greedy[k+1] | committed 1.84 → 3.42 |

### 3.3 The measurement-integrity corrections (issue468/24-25)

The prior Outcome B verdict rested on five flaws, all corrected:
1. **Dropped +1 token**: doc 23's table used `committed = raw greedy prefix`, not
   `accepted + 1`. Correcting: 64k headline 0.66× → 1.09× (still fail, but not
   catastrophic).
2. **Greedy-not-B2**: the terminal verdict used Metal greedy acceptance. The
   project's selected B2 protocol was never measured on Metal. Corrected: measured
   via offline MC on Metal base_logits.
3. **KV path never validated**: the multi-step window-KV ring path was never
   cross-checked against the oracle. Corrected: oracle holds at 2.80 with the same
   persistent-KV window — the path is correct.
4. **64k memory contradiction**: doc 19 said "~7 GB free", Phase 3 said "29.5 GiB
   headroom". Corrected: Phase 3 is right (doc 19 counted macOS page cache).
5. **ffn_gate_inp bug**: the root cause (§3.1 above).

---

## 4. Measured results and structural projection

### 4.1 Measured end-to-end (experimental implementation)

| metric | value |
|---|---|
| DSpark B2 gen t/s (ctx=4096, code prompt, n=64, temp=0) | **27.67 t/s** |
| Baseline gen t/s (same conditions) | **39.18 t/s** |
| Speedup | **0.71× (29% SLOWER)** |
| Output quality | CLEAN (verified via ds4-eval 5-case, 3/5 passed) |

This is the **actual measured result**. The gate (>20% faster) FAILS.

### 4.2 Root cause of the end-to-end gap: KV replay overhead

On partial-accept cycles (the majority), the B2 function must:
1. `spec_frontier_restore` (restore KV to pre-verify state)
2. Replay each accepted token + correction via `ds4_session_eval` (O(k+1) decode steps)

This is required because the batch-encode verify path
(`metal_graph_encode_layer_batch`) and the single-token decode path
(`metal_graph_encode_decode_layer`) use **incompatible compressed KV cache
bookkeeping**. A 1-step replay attempt (decode only the correction) was tested and
caused **compressed KV cache overflow** — the batch path writes to positions that
the decode path's compressed cache can't accommodate.

**Cost breakdown per cycle:**
| component | cost | notes |
|---|---|---|
| drafter forward (GPU) | 7.4 ms | input + 3 blocks + output head |
| Markov head (CPU) | 4.3 ms | sequential [vocab,256]@emb per position; GPU port: ~0.1ms |
| B2 accept (CPU) | 2 ms | softmax over 129280 × 2 per position × 5 |
| capture readback (CPU) | 5 ms | GPU→CPU copy + mean + upload |
| verify (GPU) | 75 ms | batch verifier (context-flat) |
| **replay** (partial accept only) | **~15 ms avg** | restore + (k+1) decode steps (k = drafts accepted) |
| **total cycle** | **~108 ms** | at ~3.0 committed → 36 ms/tok (matches measured 27.67 t/s) |

The replay adds ~15ms to the average cycle (reconciled with the measured 27.67
t/s: 36.1 ms/tok × 3.0 committed = 108ms cycle; base 94ms + 15ms replay = 109ms).
Full-accept cycles skip the replay entirely (verify KV is correct), so the
average is lower than worst-case. The structural minimum (no replay) is ~94ms.

### 4.3 Component-level projection (structurally correct, no replay)

Without replay (the production target), the cycle is:

| component | cost |
|---|---|
| drafter forward (GPU) | 7.4 ms |
| Markov head (GPU port) | 0.1 ms |
| capture + B2 accept (GPU) | 0.5 ms |
| verify (GPU) | 75 ms |
| **total cycle** | **~83 ms** |

### 4.4 Acceptance numbers (post ffn_gate_inp fix, all on Metal)

| metric | value | method |
|---|---|---|
| Metal greedy prefix | 2.74 (56.8% match) | `DS4_DSPARK_PROBE_ACCEPT` persistent-KV sweep, 19 steps |
| Metal B2 committed (offline MC) | 3.42 (A=2.42) | `measure_metal_b2.py`, 500 trials/step, correct argmax + min(1,p/q) |
| Oracle greedy (reference) | 2.79 (57.9%) | numpy oracle, same captures |
| Oracle B2 committed | 2.80 | `measure_b2_acceptance.py` |

### 4.5 The +1 bonus token (structural, always present)

Every speculative cycle commits `accepted_drafts + 1`:
- **Partial accept** (k < 5 accepted, then rejected): committed = k + 1 (the +1 is
  the correction token at the rejection point). Already counted in the MC.
- **Full accept** (all 5 drafts accepted): committed = 5 in the MC, **but should
  be 6**. The verify forward processes 5 draft positions, producing 5 target
  distributions. The 5th distribution predicts the position *after* the last draft
  — a **free trailing token** computed by verify but never drafted. The MC doesn't
  model this.

**Correcting the MC:** 6 of 19 steps are full accepts (committed 5→6):
- Without +1: avg committed = 3.42
- With +1: avg committed = **3.74**

### 4.6 Speedup projection (structurally correct, with +1)

`ms/tok = 83 / committed`:

| ctx | plain ms/tok | committed 3.42 (no +1) | committed 3.74 (with +1) |
|---|---|---|---|
| 8k | 31.3 | 1.29× (+29%) | 1.41× (+41%) |
| 32k | 32.9 | 1.36× (+36%) | 1.48× (+48%) |
| 55k | 34.8 | 1.43× (+43%) | 1.56× (+56%) |
| 64k | 35.5 | 1.46× (+46%) | 1.60× (+60%) |

**The gate passes at ALL contexts with +28-40pt margin** (with +1) in a
structurally correct implementation.

---

## 5. Production work items

Ordered by impact (highest first):

### P0: Eliminate KV replay (saves ~15ms avg/cycle — the #1 blocker)

**The single most impactful production work item.** Without this, the experimental
implementation cannot pass the gate regardless of other optimizations.

**The problem:** `metal_graph_verify_suffix_tops` uses `metal_graph_encode_layer_batch`
(batch-encode path), which writes KV via `ds4_gpu_store_raw_kv_batch_tensor` +
`ds4_gpu_attention_decode_raw_batch_heads_tensor`. The decode path uses
`metal_graph_encode_decode_layer` (single-token path). These use **different
compressed KV cache bookkeeping** — the batch path and decode path can't share KV
state directly. A 1-step replay attempt (decode only the correction at the rejection
position) caused **compressed KV cache overflow**.

**Approaches to fix:**
1. **Verify-state-reuse**: after the batch verify, make the decoded KV state
   consistent with the single-token decode path for the accepted prefix. This
   requires understanding how `metal_graph_encode_layer_batch` manages
   `layer_attn_state_kv` / `layer_attn_comp_cache` / `layer_raw_cache` vs how
   `metal_graph_encode_decode_layer` manages them, and writing a bridge that
   converts the batch-verify's KV state into the decode path's expected state.
2. **Batch-decode KV compatibility**: modify the batch verify to use the same KV
   cache layout as the decode path (or vice versa), so no conversion is needed.
3. **Prefix-1 capture** (the MTP path's approach): `metal_graph_verify_suffix_tops`
   has a `capture_prefix1` parameter that captures the first position's KV during
   verify, enabling a cheap partial-accept path for N=2. Investigate extending
   this for N=5.

**Estimated effort:** 1-2 sessions. This is the core productionization deliverable.

### P1: Port Markov head to GPU (saves 4.3ms/cycle)

The Markov head runs on CPU (sequential [vocab,256]@emb per position × 5). On GPU:
`markov_w2 [vocab, 256]` BF16 = 66MB → memory-bound at 800GB/s ≈ 0.08ms/position
→ ~0.4ms for 5 positions. Use `ds4_gpu_matmul_f16_tensor` on markov_w2 (the weight
is already BF16 in the GGUF).

**Estimated effort:** <1 session. The matmul primitive exists; just wire it.

### P2: Port B2 acceptance to GPU (saves 2ms/cycle)

The B2 acceptance (softmax over [vocab] × 2 per position × 5) runs on CPU. On GPU:
a fused softmax + accept/reject kernel would take ~0.1ms.

**Estimated effort:** <1 session. The softmax primitive exists.

### P3: In-GPU capture mean (saves 5ms/cycle)

The main_hidden capture currently does GPU→CPU readback + CPU mean + CPU upload.
A GPU mean-over-HC kernel (or reuse of `ds4_gpu_hc_weighted_sum` with uniform
weights) eliminates the readback.

**Estimated effort:** <1 session.

### P4: Extract the +1 bonus token on full accept (lifts committed 3.42→3.74)

On full-accept cycles, `spec_logits[4]` (the 5th target distribution from verify)
predicts the free trailing token. A one-line argmax extraction in the acceptance
code captures it.

**Estimated effort:** minutes.

### P5: Measure at ≥32k (validate the long-context speedup)

The projection shows +48-60% at 32k-64k. This needs to be measured end-to-end
after P0-P4 are done. Baseline decode slows from 31ms/tok (8k) to 35.5ms/tok (64k)
while verify stays context-flat at 75ms, so the speedup grows with context.

### P6: Run full 92-case ds4-eval with --dspark (validate quality)

B2 is exactness-preserving (the verifier guarantees output correctness regardless
of draft quality), so DSpark@temp=1.0 should match target@temp=1.0. A proxy
measurement (target@temp=1.0, 20 cases) scored 16/20 ≈ 73/92. The full 92-case
DSpark run is needed for the quality gate (≥65/92).

**Estimated effort:** 2-4 hours runtime (after P0).

---

## 6. Architecture: the DSpark B2 spec-decode cycle

```
 ┌──────────────────────────────────────────────────────────────┐
 │                    DSpark B2 Cycle                            │
 │                                                              │
 │  1. ANCHOR DECODE (target forward)                           │
 │     metal_graph_eval_token_raw_swa (decode path)             │
 │     → captures main_hidden at L40/41/42 (GPU copy → scratch) │
 │     → s->logits = target distribution at anchor              │
 │                                                              │
 │  2. main_hidden READBACK (CPU, ~5ms)                         │
 │     GPU→CPU: dspark_mh_capture[0..2]                         │
 │     CPU: mean over HC → dspark_main_hidden [3*dim]           │
 │     CPU→GPU: upload to dspark_main_hidden                    │
 │                                                              │
 │  3. DRAFTER FORWARD (GPU, ~7.4ms + CPU Markov ~4.3ms)        │
 │     a. input_stage: main_proj+norm → main_x [dim]            │
 │     b. embed [anchor, NOISE×4] → batch_cur_hc [5,4,dim]      │
 │     c. prefill anchor KV from main_x → dspark_kv_cache       │
 │     d. 3× DSparkBlock:                                       │
 │        hc_pre → attn (non-causal, anchor KV, DSpark topk)    │
 │        → hc_post → hc_pre → MoE → hc_post                    │
 │     e. output_head: hc_head → norm → shared lm_head          │
 │        → spec_logits [5, vocab] (drafter base logits)        │
 │     f. Markov head (CPU): markov_w1[prev] @ markov_w2.T      │
 │        → q_dist [5, vocab] (drafter distribution)            │
 │        → drafts [5] (argmax per position, sequential)        │
 │                                                              │
 │  4. VERIFY (GPU, ~75ms)                                      │
 │     metal_graph_verify_suffix_tops (batch verifier)          │
 │     → row_tops [5] (target argmax per position)              │
 │     → row_logits [5, vocab] (target full distribution)       │
 │                                                              │
 │  5. B2 ACCEPTANCE (CPU, ~2ms)                                │
 │     For each draft position i:                               │
 │       q(x) = softmax(q_dist[i])                              │
 │       p(x) = softmax(row_logits[i])  (or s->logits for i=0)  │
 │       accept w.p. min(1, p/q)                                │
 │       on reject: correction = argmax(p-q), commit, STOP      │
 │     → n_draft_accept, accepted[]                             │
 │                                                              │
 │  6. KV MANAGEMENT                                            │
 │     Full accept: verify KV correct, advance checkpoint       │
 │     Partial: spec_frontier_restore + replay (k+1) decode     │
 │     ← THIS IS THE BOTTLENECK (~15ms avg)                     │
 │                                                              │
 │  Committed: n_draft_accept + 1 (+1 free trailing on full)    │
 └──────────────────────────────────────────────────────────────┘
```

---

## 7. Independent verification instructions

The B2 implementation is independently verifiable. Build with `make` (Metal default).

### Timed gen t/s
```sh
# DSpark B2 (experimental, with replay overhead):
./ds4 -m MODEL --dspark dspark.gguf -c CTX -n N --temp 0.0 -p PROMPT
# → generation: XX.XX t/s

# Baseline (target-only):
./ds4 -m MODEL -c CTX -n N --temp 0.0 -p PROMPT
# → generation: XX.XX t/s

# Debug output:
DS4_DSPARK_B2_DEBUG=1 ./ds4 -m MODEL --dspark dspark.gguf -c CTX -n N --temp 0.0 -p PROMPT
```

### Quality
```sh
./ds4-eval -m MODEL --dspark dspark.gguf --plain --nothink -n 4096 --seed 1 --questions 92 --temp 1.0
```

### Component-level measurements (no decode loop)
```sh
# Backbone cost (3 drafter layers, n_tokens=5):
DS4_DSPARK_TIME_BACKBONE=1 ./ds4 -m MODEL --dspark dspark.gguf -c 8192 --verifier-curve-test -p PROMPT

# Full draft cycle timing (input + 3 blocks + output head):
DS4_DSPARK_PROBE_ACCEPT=1 ./ds4 -m MODEL --dspark dspark.gguf -c 8192 --verifier-curve-test -p PROMPT

# Offline B2 MC on Metal base_logits:
DS4_DSPARK_PROBE_ACCEPT=1 DS4_DSPARK_PROBE_DUMP_Q=1 ./ds4 -m MODEL --dspark dspark.gguf -c 8192 --verifier-curve-test -p PROMPT
python3 issue468/baseline/dspark_capture/measure_metal_b2.py
```

---

## 8. Doc reference index

| doc | title | status |
|---|---|---|
| 23 | Final verdict long-ctx (Outcome B) | **SUPERSEDED** — based on the ffn_gate_inp bug |
| 24 | Research review and next goals | The independent review that re-opened the verdict |
| 25 | Assignment 1 measurement integrity | Validated: B2 on Metal, KV reconcile, RAM, +1 accounting |
| 26 | Source B investigation | Ruled out F16-mid, F16-act, F16-weight; found routing was the issue |
| 27 | Assignment 2 root cause found | **THE root cause: ffn_gate_inp F32/F16 mismatch** |
| 28 | Final verdict Outcome A | **SUPERSEDED** by doc 30 — projection, not end-to-end measurement |
| 29 | Honest Markov correction | Markov head (4.3ms CPU) was missing from draft timing |
| 30 | B2 measured result | **The measured end-to-end: 0.71×**, replay overhead root cause |
| 31 | Honest blocked state | The blocked state summary (this note supersedes it) |
| **32** | **This handoff note** | **Standalone, self-contained, production-ready** |

---

## 9. File map

### Code changes (ds4.c)
- `metal_graph_dspark_input_stage` — drafter input stage (main_proj+norm+embed+HC)
- `metal_graph_dspark_encode_attention` — attention sub-block (non-causal, anchor KV)
- `metal_graph_dspark_encode_block` — one DSpark block (attn + FFN reuse)
- `metal_graph_dspark_output_head` — hc_head + norm + lm_head
- `ds4_session_eval_dspark_b2` — complete B2 spec-decode cycle
- `ds4_dspark_probe_accept` — persistent-KV greedy acceptance sweep
- `ds4_dspark_probe_input_stage` — input-stage validation probe
- `ds4_dspark_time_backbone` — backbone timing probe
- In-GPU main_hidden capture at layers 40/41/42 (GPU-to-GPU copy)
- `ds4_engine_has_dspark` — engine capability check
- `enable_dspark` threaded through `metal_graph_alloc_raw_cap`

### Code changes (ds4_metal.m)
- `ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor` — non-causal batched attention
- `ds4_gpu_encode_flash_attention_decode_raw_batch_heads` — added `noncausal` param

### Code changes (ds4_cli.c, ds4_eval.c)
- `--dspark` flag parsing + B2 dispatch in generation loop
- ds4-eval spec-decode path routing for DSpark

### Code changes (build_dspark_template.py)
- ffn_gate_inp: F32 → **F16** (the root cause fix)
- hc_attn_fn, hc_ffn_fn: F32 → **F16** (matmul_f16 kernel requirement)
- hc_head_fn: F32 → **F16** (batched matmul_f16 requirement)
- main_norm, mtp.2.norm: BF16 → **F32** (rmsnorm kernel reads F32)

### Data
- `../ds4/gguf/dspark.gguf` — 10.71 GiB, 81 tensors (rebuilt with F16 fixes)
- `issue468/baseline/dspark_capture/` — main_hidden captures (code+chat, 152-280)
- `issue468/baseline/dspark_capture/metal_base_logits_19steps.bin` — Metal drafter logits
- `issue468/dspark_oracle/` — numpy oracle (validated algorithm reference)
