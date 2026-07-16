# Lead 08 re-assessment — task 1 (reorient): cost floor + "outperforms batch" bar, re-derived from the M3 landscape

Date: 2026-07-15. Goal: `mrmkwnp6-6n9z9x` (Lead 08 fused-verify-kernel re-assessment).
Purpose: anchor the viability bar to the ACTUAL current batch verifier in the live M3 full stack (not the old Phase-B corpus), re-confirm the Phase-B floor/gate/exactness starting points still hold, and frame what the next task (cheap-probes) must decide.

## 1. Starting-point confirmation — the Phase-B verdict still holds

M3 (levers 1–5) did **not** touch the verify kernel (`metal_graph_verify_suffix_tops` is unchanged). M3 worked the draft side (Metal drafter 45→7.6 ms) + the decode side (anchor-reuse recovered the standalone-anchor decode) + the scheduling (STS verify_n adaptation). So the Phase-B floor/gate/exactness findings are still valid:

- **+20% gate threshold:** `verify_ms(4) ≤ 50.5 ms` (sound, re-confirmed).
- **Byte floor at decode effective bandwidth (~410–450 GB/s, gate-B resolved):** `verify_ms(4) ≈ 39–43 ms` → ~1.34–1.44× (clears +20% *if* a sublinear+exact kernel reaches decode bw).
- **Verify headroom = GPU `layer_execute` bandwidth inefficiency** (NOT host overhead, which is ~3 ms / ~5%). The verify path runs the routed-expert stream at **~190 GB/s** vs decode's ~410–450 GB/s — i.e. the verify is at ~42–46% of decode bandwidth on the expert stream. That gap is the recoverable term.
- **Exactness gap:** the batch path diverges in **down+sum6 + attention (HC/compressor)**; the gate/up IQ2XXS paired reduction is **already shared + bit-identical** (both the decode `id` kernel and the batch `addr` kernel call the same `_impl`: same dequant, MAC order, `simd_sum`+`*0.25`). The expert **weight load is NOT shared across the 2 tokens** in either path → the cost saving is unrealized (the fusion target).
- **Existing primitives (re-confirmed, the gap a fused kernel must close):**
  - `verify_suffix_tops` (batch): **sublinear** (~27.3 ms@K2, ~66 ms@K4), **NOT greedy-exact** (0.64% argmax flip, TV ~0.0104, max_abs up to 4.56).
  - `decode2_exact` (sequential): **greedy-exact** (TV=0), **LINEAR** (~28 ms/token, ~0.85× baseline).
  - **No config swap** yields sublinear+exact (settled by Phase B → do not re-litigate; novel kernels required).

## 2. The current batch verifier, LIVE in the M3 full stack (the bar to beat)

Measured from `issue468/artifacts/dspark_m3_bench/large_corpus/full_stack.jsonl` (n=176, full corpus, bootstrap-CI run that gave +4.9% vs plain):

| metric | value |
|---|---|
| `dspark_verify_ms_mean` | **62.07 ms/cycle** |
| `dspark_verify_n_mean` | **3.779** |
| `dspark_draft_ms_mean` | 6.64 ms |
| `dspark_decode_ms_mean` | 0.671 ms |
| `dspark_total_ms_mean` | 69.38 ms |
| `dspark_verified_mean` | 2.811 tokens/cycle |
| **verify share of cycle** | **89.5%** |
| verify ms / verified-token | 22.08 ms |

Reconciliation: Phase-B fit `verify_ms(K) ≈ 40 + 6.6·K` → at K=3.78 ≈ 64.9 ms ≈ the live 62 ms ✓. **The verify is now 89.5% of the cycle** (even more dominant than the 93-subset's ~80%, because the full corpus's longer prompts raise verify_n). This is the strongest-possible framing for the Lead 08 prize — any verify reduction now translates almost 1:1 to end-to-end speedup (draft+decode are already ~7.3 ms combined).

## 3. The re-derived bars

- **PRIMARY viability bar (user-set: "outperforms the existing batch verifier, any improvement"):**
  a fused kernel's `verify_ms(K)` **strictly < the batch verifier at matched K**. Anchored reference: **~62 ms at verify_n≈3.78** (live full stack), i.e. **~65 ms at K=4** (Phase-B long bench). Any reduction below this counts as "outperforms." This is *easier* than the +20% gate.
- **Floor (theoretical max reduction):** `verify_ms(4) ≈ 39–43 ms` (decode bw). Headroom from the live 62 ms ≈ **~20 ms (~32% reduction potential)**.
- **Stretch (+20% project gate):** `verify_ms(4) ≤ 50.5 ms` → ~12 ms / ~19% reduction from 62 ms.

So the viability question is sharp: **can a fused (shared-load + bit-exact) verify kernel shave any of the ~20 ms bandwidth-gap headroom — i.e. push verify_ms below ~62 ms at matched K — while preserving greedy-exactness?**

## 4. What task 2 (cheap-probes) must decide (the falsification inputs)

Three cheap, no-novelel-kernel measurements feed the gate-A falsification call:

1. **Bandwidth probe:** can a union-expert load-once kernel reach decode bw (~410–450 GB/s) on the expert stream? If the achievable bandwidth caps well below decode bw, the ~20 ms headroom is unreachable → **falsifies the cost viability** (no point building). (The Phase-B expert-stream slope ~190 GB/s is the current cost; decode bw is the target.)
2. **Confirmatory MoE-equality harness:** gate-C's pending "consistent with" → "confirmed" — a literal identical-input kernel-equality check (the gate/up `_impl` shared reduction, run with byte-identical inputs on both paths). If the shared reduction is bit-identical given identical inputs → the "exactness-by-construction" thesis for gate/up holds → the fusion can reuse it. If not → the exactness story is shakier.
3. **Exactness-gap re-confirm:** re-confirm the divergent stages (down/sum6 + attention; gate/up already shared). Confirms the fidelity-gate target (`max_abs==0` vs M=1) is the right exactness bar + that the fusion target is the **down+sum6 + attention**, not gate/up.

## 5. Decision rule (carried into task 3 / gate A)

- **Falsify (NO-GO, skip prototype):** if the bandwidth probe shows the headroom is unreachable, OR the MoE-equality harness refutes the shared-reduction bit-identicality (undermining exactness-by-construction for the easy stage).
- **Escalate (build the M=2 routed-expert prototype):** if the bandwidth probe is plausibly reachable + the MoE-equality confirms the shared reduction — then the single-stage prototype is the cheapest real measurement of both the cost saving + the fidelity gate.

Carried forward to `02_*` probe artifacts under `issue468/artifacts/lead08_reassessment/`.
