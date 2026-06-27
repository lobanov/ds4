# Phase 1 — Verifier Cost Curve Study: Execution Plan (PAUSED, awaiting go-ahead)

Status: **prepared, not started.** This doc defines the microbench, its
placement, the measurement matrix, and the exact commands. No engine code has
been written and no benchmark has been run. Confirm the placement decision in
§3 before I touch `ds4.c`.

Goal (from updated `PLAN.md`): produce `verify(L)` curves for L=1..5 on **all
three** verifier kernels, plus the internal cost breakdown of the exact verifier
at L=2. Output selects Branch A / Branch B / stop.

## 1. The three kernels to measure (confirmed from code)

| kernel | site | L range | exact? | shape today |
|---|---|---|---|---|
| `metal_graph_verify_suffix_tops` | `ds4.c:21117` | 1..16 | **no** (batch, layer-fused) | sub-linear at L=2 (40 ms) |
| `metal_graph_verify_decode2_exact` | `ds4.c:21218` | **2 only** | yes (decode-kernel ×2) | super-linear at L=2 (58–67 ms) |
| `metal_graph_eval_token_raw_swa` | `ds4.c:19461` | 1..N (called ×L) | yes (normal decode kernel) | linear (~28 ms × L) |

Key constraint driving the design: the exact-fused verifier exists only at N=2.
For L>2, "exact" can only mean **sequential exact** (kernel #3 called L times).
So the exact curve for L>2 is the sequential path; the fused exact is a single
L=2 point. The microbench must make this explicit rather than papering over it.

## 2. What the microbench must isolate

Each measurement times **only the verifier kernel** on a synthetic suffix of L
greedy tokens appended after a prefilled prompt — no drafter, no accept logic,
no replay. Per cell:

1. prefill the prompt to the target context (untimed)
2. take L real greedy tokens from the target (so the suffix is realistic, not
   random — acceptance drift depends on real continuations)
3. for each kernel: snapshot graph frontier, run the kernel on the L-token
   suffix, restore; repeat K iterations; record **median** wall ms
4. for the exact-fused N=2 kernel at L=2 only, also record the **internal
   breakdown**: (a) 2× decode-layer dispatch, (b) per-layer prefix-1 capture,
   (c) 2× output-head, (d) 2× full-vocab readback — by adding temporary
   `now_sec()` checkpoints inside `metal_graph_verify_decode2_exact` behind the
   same env flag (Branch A data)

The snapshot/restore reuses the existing `spec_frontier_snapshot`/`restore`
machinery already in `ds4_session_eval_speculative_argmax`, so each iteration
starts from an identical frontier and the kernel cost is measured cleanly.

## 3. Placement decision (recommendation — needs your sign-off)

**Recommended: a new research-only function `ds4_engine_verifier_curve_test()`,
mirroring the existing `ds4_engine_metal_graph_*_test` family.**

- Declare in `ds4.h` next to line 193:
  `int ds4_engine_verifier_curve_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);`
- Implement in `ds4.c` near the other test functions (after line 25337).
- Wire a `--verifier-curve-test` flag in `ds4_cli.c` (next to
  `--metal-graph-test` at line 1575; branch next to line 879) and a one-line
  help entry in `ds4_help.c` (next to line 259).

Why this over the alternatives:

| option | verdict |
|---|---|
| **new `ds4_engine_*_test` + `--verifier-curve-test` flag** (recommended) | Smallest surface; reuses model load + session; kernels are `static` in `ds4.c` so they're only reachable from inside `ds4.c` anyway; matches the established research-function convention; trivially removable. |
| extend `ds4-bench` with `--spec`/`--verifier-sweep` | Larger change to a tool with existing users; the static kernels aren't visible to `ds4_bench.c` without new exposure. Defer (it's the Phase 0 harness gap, but not Phase 1's job). |
| new standalone `ds4_verifier_bench` binary | New build target + Makefile wiring + duplicated model-load boilerplate; the kernels are still `static` so it doesn't even work without exposing them. Reject. |

The function is **research-only and env/flag-gated**; it does not add a
permanent semantic variant (per `AGENT.md`). It touches only the speculative
verifier path and the CLI/help wiring — not SSD streaming, CUDA, distributed,
or the default Metal decode path.

## 4. Measurement matrix

Fixed by `PLAN.md` Work item 2: L × context × prompt class.

- L ∈ {1, 2, 3, 4, 5}
- context ∈ {2048, 4096, 8192}
- prompt class ∈ {chat (`prompts/chat_general.txt`), code (`prompts/code_humaneval.sh`)}
- kernels ∈ {batch, exact-fused-N2 (L=2 only), sequential-exact}
- iterations per cell: 20 timed after 3 warmup; report **median** (Metal
  variance warrants median, not mean)

Cells: 5 L × 3 ctx × 2 prompt = 30 per kernel; exact-fused only contributes its
L=2 cells (6). Total ≈ 96 timed sweeps. Each sweep is one short prefill + 20
kernel iterations, so the whole matrix is a few minutes of GPU time.

Machine/thermal protocol identical to Phase 0 (`03_baseline_command_set.md` §0):
M5 Max, `--power 100`, single process, wall power, record git SHA.

## 5. Outputs

- `issue468/baseline/verifier_curve.csv` — one row per (kernel, L, ctx, prompt,
  median_ms, p10_ms, p90_ms).
- `issue468/baseline/verifier_curve_exact_l2_breakdown.txt` — the internal
  phase breakdown of `metal_graph_verify_decode2_exact` at L=2 (Branch A data).
- `issue468/06_phase1_results.md` — filled curve tables + the one-paragraph
  verdict (sub-linear / linear / cliffy; where the headroom is) + the Branch
  A/B/stop recommendation.

## 6. Stop/Go (from updated `PLAN.md`)

- Exact `verify(L)` can be made sub-linear → **Branch A**: proceed to Phase 2
  with exact costs.
- Only batch is sub-linear; exact stays linear/super-linear → **Branch B**:
  Phase 2 adds benchmark-quality methodology; drop exact-greedy from primary
  gate.
- Even batch is cliffy at low L → **stop local**; reconsider server-side only.

## 7. Commands to run once approved (sketch — not executed)

```sh
export MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
# After implementing ds4_engine_verifier_curve_test + --verifier-curve-test flag:
./ds4 -m "$MODEL" --verifier-curve-test \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  --csv issue468/baseline/verifier_curve_chat.csv 2> issue468/baseline/verifier_curve_chat.log
./ds4 -m "$MODEL" --verifier-curve-test \
  -p "$(bash issue468/prompts/code_humaneval.sh)" \
  --csv issue468/baseline/verifier_curve_code.csv 2> issue468/baseline/verifier_curve_code.log
```

MTP GGUF is **not** loaded for Phase 1 — the microbench calls the target
verifier kernels directly; the drafter is out of scope.

## 8. Open questions before I start coding

1. **Placement:** approve the `ds4_engine_verifier_curve_test` +
   `--verifier-curve-test` approach (§3), or do you want `ds4-bench` extended
   instead?
2. **L=2 internal breakdown (§2.4):** OK to add temporary `now_sec()`
   checkpoints inside `metal_graph_verify_decode2_exact` behind an env flag?
   This is the only edit that touches a production verifier kernel (read-only
   timing probes), so I want explicit sign-off.
3. **Iterations:** 20 timed + 3 warmup, median — acceptable, or prefer a fixed
   wall-time budget per cell?
4. **L ceiling:** cap at L=5 (matches `PLAN.md`), or also probe L=8 to see the
   knee? The batch kernel supports up to 16.

Awaiting your call before any of this runs.
