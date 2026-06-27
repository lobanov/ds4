# Phase 1 — Verifier Cost Curve: Results & Branch Decision

Status: **EXECUTED.** Microbench `ds4_engine_verifier_curve_test` (behind
`--verifier-curve-test`) measured `verify(L)` for L=1..8 on all three verifier
kernels at ctx ∈ {2048, 4096, 8192}, for chat and code prompts. 20 timed
iterations + 3 warmup per cell, median reported. Plus the internal phase
breakdown of the exact verifier at L=2 (Branch A data) via
`DS4_VERIFY_CURVE_BREAKDOWN`.

Machine: Apple M5 Max / 128 GB / Metal, `--power 100`, wall power. Same git
SHA family as Phase 0 (`f54bee1`+ research-code addition). Raw CSV/logs in
`issue468/baseline/verifier_curve_{chat,code}.{csv,log}`.

**Headline: Branch B selected.** Only the batch verifier is sub-linear; every
exact path is linear-or-worse. See §4.

---

## 1. The three kernels (recap)

| kernel | site | exact? | available at |
|---|---|---|---|
| `metal_graph_verify_suffix_tops` (batch) | `ds4.c:21117` | **no** | L=1..16 |
| `metal_graph_verify_decode2_exact` (exact-fused) | `ds4.c:21218` | yes | **L=2 only** |
| `metal_graph_eval_token_raw_swa` ×L (sequential) | `ds4.c:19461` | yes | L=1..N |

## 2. verify(L) curves — median ms (chat ≈ code; chat shown, code within ~1 ms)

| L | batch 2k/4k/8k | seq_exact 2k/4k/8k | exact_fused 2k/4k/8k |
|---|---|---|---|
| 1 |  31 / 36 / 36 |  26 / 26 / 31 | — |
| 2 |  42 / 50 / 50 |  52 / 52 / 62 | **56 / 57 / 66** |
| 3 |  51 / 58 / 57 |  78 / 79 / 93 | — |
| 4 |  70 / 69 / 69 | 105 / 112 / 126 | — |
| 5 |  76 / 75 / 75 | 131 / 143 / 157 | — |
| 6 |  80 / 79 / 79 | 158 / 174 / 189 | — |
| 7 |  88 / 87 / 95 | 184 / 205 / 220 | — |
| 8 |  97 / 95 / 96 | 212 / 238 / 253 | — |

Observations:

- **Chat ≈ code** to within ~1 ms everywhere → verifier cost is a pure
  kernel/system property, prompt-independent. The simulator needs only one
  curve.
- **Batch is context-flat** (batch L=8 ≈ 96 ms at 2k/4k/8k). The SWA window
  makes decode attention cost independent of prefix length for the batched path.
- **Batch per-token amortized** drops 36 ms (L=1) → 12 ms (L=8).
- **Sequential is perfectly linear**, ~26 ms/token at 2k/4k, ~31 ms/token at 8k
  (= the Phase 0 baseline decode step).
- **Batch beats sequential at every L≥2** and at every context; the gap widens
  with L. Batch is *slower* than sequential only at L=1 (batched-path fixed
  overhead) — so the scheduler must never choose L=1 via batch (use plain
  decode). Matches the DSpark design (schedule 0 or ≥2).
- **exact_fused N=2 is the slowest of all three** at L=2: 56–66 ms vs sequential
  52–62 ms vs batch 42–50 ms. `fused/seq = 1.07–1.09` (the "fusion" adds capture
  overhead with **zero** speedup); `batch/fused = 0.74–0.88`.

## 3. Batch sub-linearity (the simulator input) @ ctx=8192

| L | batch (ms) | batch/seq | batch per-tok (ms) | marginal batch (ms) |
|---|---|---|---|---|
| 1 | 36.0 | 1.16 | 36.0 | — |
| 2 | 49.9 | 0.80 | 24.9 | +14 |
| 3 | 57.4 | 0.62 | 19.1 | +7 |
| 4 | 69.0 | 0.55 | 17.2 | +12 |
| 5 | 74.7 | 0.47 | 14.9 | +6 |
| 6 | 79.2 | 0.42 | 13.2 | +4 |
| 7 | 94.9 | 0.43 | 13.6 | +16 |
| 8 | 95.6 | 0.38 | 12.0 | +1 |

(marginal = batch(L) − batch(L−1); noisy at L=6→7→8 but trend is ~4–8 ms/token)

The marginal cost of adding a verified token to the batch (~4–8 ms) is roughly
**a third of a sequential decode step** (~26–31 ms). That is the entire economic
basis for speculative decoding here — but it is realized **only for accepted
tokens**, and only on the non-exact batch kernel.

## 4. Exact-verifier breakdown (Branch A data) — L=2, averaged over 26 samples

| ctx | layers (2× decode-layer + capture) | out0 (head+argmax+2 readbacks) | out1 (head+1 readback) | total |
|---|---|---|---|---|
| 2048 | **53.3 ms (95%)** | 1.31 ms | 1.22 ms | 56.3 ms |
| 4096 | **54.0 ms (95%)** | 1.31 ms | 1.22 ms | 56.9 ms |
| 8192 | **63.5 ms (96%)** | 1.30 ms | 1.21 ms | 66.4 ms |

**Branch A verdict: not viable without new kernel work.**

1. The Phase 0 / Phase 1 hypothesis that headroom lived in the output-head /
   full-vocab readbacks is **disproven**: that overhead is ~2.5 ms (5%).
   Eliminating it entirely saves ~5%.
2. **95% of the cost is the 2× single-token decode-layer dispatches.** The
   current `decode2_exact` does not actually fuse the two positions — it runs the
   single-token decode kernel twice with a prefix-1 capture between them. That is
   why `fused/seq ≈ 1.08`: it is sequential decode plus capture overhead.
3. Making the exact path sub-linear therefore requires a **genuinely bit-exact
   batched layer kernel** (batched attention + MoE reductions that match the
   single-token path bit-for-bit). The existing batch verifier *is* batched but
   is **not** bit-exact — so bit-exact batching is exactly the unsolved hard
   part, not an incremental win.
4. The exact path's per-token cost also **grows with context** (53→64 ms,
   +21%), where the batch kernel is flat — a further mark against any exact-fused
   approach.

Conclusion: Branch A is a research-grade kernel project (bit-stable batched
reductions across the MoE/attn path) with uncertain payoff. **Defer.** Revisit
only if Branch B's quality validation fails *and* a bit-exact batched layer
kernel becomes feasible.

## 5. Stop/Go gate resolution (from updated PLAN.md)

- *Exact `verify(L)` can be made sub-linear → Branch A.* **No** — exact is
  linear-or-worse; the only headroom is a 5% readback trim, and the 95% layer
  cost needs new bit-exact batched kernels.
- *Only batch is sub-linear, exact stays linear/super-linear → Branch B.*
  **This is the outcome.** Batch is smoothly sub-linear and context-flat.
- *Even batch is cliffy at low L → stop local.* **No** — batch is smooth.

**→ PROCEED to Phase 2 on Branch B.** Drop "exact greedy output preservation"
from the primary success gate (it is impossible to satisfy locally without a new
exact-batched kernel). The primary gate becomes: **≥20% greedy-throughput-class
improvement on the batch verifier, with the non-exactness shown quality-neutral
on real-world task benchmarks.**

## 6. What this commits Phase 2 to (Branch B)

1. **Simulator input = the batch `verify(L)` curve** (§3, context-flat, so a
   single curve suffices). Combine with the draft-cost and acceptance data from
   Phase 0 (draft ≈ 2 ms; pos-2 conditional acceptance ≈ 0.41 default / would
   need a better drafter to reach DSpark's ≈ 0.6+).
2. **Benchmark-quality methodology promoted to load-bearing** (was Phase 7): the
   batch verifier's logit drift must be shown quality-neutral on
   HumanEval/MBPP/MT-Bench/Arena-Hard-style evals. Phase 0 already established
   default `--mtp` is *token-different* from target-only, so this is the risk to
   retire, not merely a robustness check.
3. The scheduler rule (empirically grounded): verify at L=0 (plain decode) or
   L∈[2,γ]; **never L=1 via batch** (batch L=1 is slower than sequential).

## 7. Instrumentation delivered (research-only, env/flag-gated)

- `ds4_engine_verifier_curve_test()` in `ds4.c`, behind CLI flag
  `--verifier-curve-test` (mirrors the `ds4_engine_metal_graph_*_test` family).
  Declared in `ds4.h`; help in `ds4_help.c`. Uses `spec_frontier_snapshot`/
  `restore` for per-iteration isolation (identical to production's speculative
  path).
- `DS4_VERIFY_CURVE_BREAKDOWN` env var: per-phase probes inside
  `metal_graph_verify_decode2_exact` (the §4 data). Zero cost in production
  (unset → no `now_sec()` / `fprintf`).
- **Note:** the microbench must be run with `--mtp <mtp.gguf>` — the spec
  snapshot/`spec_logits`/`spec_prefix1_*` buffers are only allocated when
  `enable_mtp` is set. The drafter itself is not exercised.

### Reproduce

```sh
export MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
export MTP="../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
export DS4_VERIFY_CURVE_BREAKDOWN=1   # optional: capture the §4 breakdown to stderr
./ds4 -m "$MODEL" --mtp "$MTP" --verifier-curve-test --power 100 -c 8192 \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  > issue468/baseline/verifier_curve_chat.csv \
  2> issue468/baseline/verifier_curve_chat.log
# repeat with "$(bash issue468/prompts/code_humaneval.sh)" for the code curve
```
