# Lead 09 — Union expert de-dup WITHOUT compute-fusion (continuation of Lead 08)

Date: 2026-07-16. Status: **proposed continuation** (not yet probed). Lineage: directly from
Lead 08's NO-GO (`pending/lead_08_fused_verify_kernel.md`). Lead 08 falsified the *mechanism*
(fuse the per-token gate+up compute into one M=2 kernel); it did **not** falsify the *thesis*
(expert de-dup is a real DRAM saving). This lead carries the asset forward on a corrected
mechanism.

## The asset Lead 08 left us (still valid)
- The verify is **89.5 % of the M3 cycle** (62.07 ms of 69.38 ms) → a saving amplifies ~1:1
  (~10 ms = +20 % over plain).
- The probe's pair-vs-unique was DECISIVE: the routed-MoE cost tracks **PHYSICAL pairs**
  (K×6), not unique experts → each token's 6 experts are loaded separately from DRAM
  (cache-miss) → **de-dup is a real DRAM-bandwidth saving** of (K×6 − n_unique) expert loads.
  For K=2 with typical overlap that is ~3 of 12 expert loads (the reliable ~6–8 ms prize).
- `metal_graph_verify_suffix_tops` (the production batch verifier) does NOT de-dup today — it
  loads each token's experts separately even within one layer-major pass.

## The lesson Lead 08 taught (the mistake to not repeat)
Lead 08's M=2 fused the two tokens' gate+up **compute** into one kernel holding **2× the
per-thread state** (`yl_a[32]`, `yl_b[32]`, `sumg_a/u_a`, `sumg_b/u_b` ≈ 68 floats vs M=1's
~34). That blew the GPU register file → occupancy collapse → **4.4× slower** (cold all-layers
sweep: M=2 13.3 vs M=1×2 3.0 ms/layer). Codex gate B: even fixing the M=2's 18-vs-12-stream
bug, the bound was ~2.9× slower. **The compute-fusion is occupancy-bound on this hardware —
wrong mechanism.** The de-dup was real but overwhelmed.

## The corrected thesis
**De-dup the expert LOADS; keep the per-token gate+up COMPUTE in the efficient single-token
form (no 2× per-thread state).** Concretely: load each union expert (or expert *tile*) ONCE,
then run the two tokens' gate+up sequentially, reusing the single-token register set. The
expert dequant/load is shared (the saving); the compute stays single-token-efficient (no
occupancy collapse).

## Two candidate mechanisms
1. **Sequential-tile fused kernel (the M=2 redesigned).** Per ib32 tile: load the expert tile
   ONCE into shared memory (small — one tile, not the 0.3 GiB expert), then compute token A's
   gate+up MAC with the single-token register set (`yl[32]` + `sumg/u`), then token B's
   *reusing the same registers*. Dequant shared; compute single-token; down stays per-token
   (selection-ordered, as in Lead 08). Register footprint ≈ M=1 (occupancy-safe) — this is the
   fix for the exact failure mode codex diagnosed.
2. **Union-aware batch verify (production integration).** Modify `metal_graph_verify_suffix_tops`'s
   routed-MoE stage to compute the K-token expert union, load each unique expert once (ordered
   for cache / into a scratch), and apply it per-token with the existing per-token kernel. No
   new fused-compute kernel; the de-dup lives inside the existing efficient batch structure.
   (This is the suffix_tops integration Lead 08 deferred.)

## Measurement-first probe (DECIDE GO/NO-GO before any kernel build)
The crux is the **cache**: does the GPU hold the union experts across the per-token reads?
- **P1 — union-prefetch test (free, no kernel):** in the Lead 08 harness, dispatch a light
  read of the 9 union experts, then measure M=1×2. If M=1×2 drops from 3.0 ms (cold) toward
  the cached ~0.4 ms, the de-dup is captured by **ordering alone** (a prefetch/reorder win,
  no new kernel — the cheapest possible GO). NB: the probe's existing readahead flag
  (`DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED`) gave −0.04 ms, so a generic
  readahead is already a no-op — P1 must test a *targeted union* prefetch, not the flag.
- **P2 — register/occupancy estimate for the sequential-tile kernel:** from the M=1 kernel's
  register footprint + the shared-mem tile, confirm the redesigned kernel is occupancy-safe
  (registers ≈ M=1, not 2×). This is the difference from Lead 08's M=2.
- **P3 (only if P1 is negative):** a minimal sequential-tile prototype on one layer, measured
  cold (all-layers sweep, as in Lead 08) — does it beat M=1×2?

## Decision rule
- **GO** if P1 shows the de-dup is captured by ordering (cheap), OR P2+P3 show a sequential-tile
  kernel that is occupancy-safe AND beats M=1×2 cold.
- **NO-GO** if the cache does not hold across dispatches (P1 negative) AND a load-once kernel is
  still occupancy-bound or not faster (P3) — i.e. the de-dup cannot be captured without the
  register pressure that killed Lead 08.

## Why this is a fresh lead, not a revival of Lead 08
Lead 08's M=2 is **simultaneous** compute-fusion (2× registers → occupancy collapse). This lead
is **sequential** compute with shared loads (single-token registers → occupancy-safe). Same
thesis (de-dup), different mechanism — and the measurement-first probe (P1's targeted
union-prefetch) is a free check Lead 08 never ran (it jumped straight to the fused kernel).

## Hard-won constraints to carry over
- Measure COLD (all-layers sweep, fresh experts → DRAM); a tight single-layer loop is cached
  and hides the de-dup (Lead 08's early cost number was misleading until the cold sweep).
- The cold baseline is M=1×2 = 3.0 ms/layer (lenient — the real batch verifier amortizes more,
  so the bar to clear is ≤ ~3.0 ms/layer, ideally ≪).
- Env-gate any new path default-off; ≥1 codex checkpoint before a verdict; don't grind past a
  bounded bug-hunt.
