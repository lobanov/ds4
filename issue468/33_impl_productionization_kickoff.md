# DSpark Productionization — Impl Kickoff & Clean Port

Date: 2026-06-30. This is the first productionization handoff note, recorded in
parallel on the research branch (`dspark`) while the production code is being
built in a separate worktree (`/Users/lobanov/Projects/ds4-dspark-impl`, branch
`dspark-impl`). It picks up from the standalone handoff (issue468/32).

Purpose: record the state of the productionization effort as it proceeds, so the
research record stays continuous with the impl work. These notes are documentation
only — no code, no binaries; all production code lives in `dspark-impl`.

---

## 1. Setup and branch topology

| repo / worktree | branch | role |
|---|---|---|
| `/Users/lobanov/Projects/ds4-dspark` | `dspark` (902b6f9) | research record: validated code + issue468/* docs |
| `/Users/lobanov/Projects/ds4-dspark-impl` | `dspark-impl` (cc33048 base) | production target: clean, self-contained port |
| `/Users/lobanov/Projects/ds4` | `local-gen-with-dist-prefill` | shared mainline (model + gguf assets) |

The two DSpark branches diverged from common base `80ebbc3` (397 vs 333 commits
ahead respectively), so this is a real port/integration, not a fast-forward.

---

## 2. The clean port (commit f220ebe)

The validated research code is **purely DSpark work** — 65 commits since the common
base, with no incidental/non-DSpark changes mixed in. The code-only diff against
the base is exactly **2249 insertions / 24 deletions** across 6 files:

| file | delta | content |
|---|---|---|
| `ds4.c` | +2034 | `metal_graph_dspark_*`, `ds4_session_eval_dspark_b2`, probes |
| `ds4.h` | +10 | dspark model/weights structs, `enable_dspark` |
| `ds4_gpu.h` | +16 | noncausal batched attention tensor decl |
| `ds4_metal.m` | +96 | `ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor` |
| `ds4_cli.c` | +37 | `--dspark` flag + B2 dispatch |
| `ds4_eval.c` | +80 | ds4-eval `--dspark` routing |

**Method:** `git diff 80ebbc3 902b6f9 -- <6 files>` → `git apply --3way` onto
`dspark-impl`. Applied **cleanly to all 6 files, no conflicts, no conflict
markers** (3-way merge resolved the few contextual hooks like
`enable_mtp` → `enable_mtp || enable_dspark` automatically).

All validated fixes from issue468/32 §3 preserved verbatim:
- `ffn_gate_inp` F32→F16 (THE root cause, §3.1)
- `hc_attn_fn`/`hc_ffn_fn`/`hc_head_fn` F32→F16
- `main_norm`/`mtp.2.norm` BF16→F32
- inverse rope on attention output
- `spec_logits` / `spec_attn_state_kv` allocated for `enable_dspark`
- `dspark_model.map` registration + `accelerator_cache_model_tensors`

This is a **faithful** port — no algorithm changes, no redesign.

---

## 3. Runtime reproduction (the port is functionally correct)

Build succeeds. The real test of a faithful port is that the validated baseline
behavior reproduces end-to-end. It does:

```
DS4_DSPARK_B2_DEBUG=1 ./ds4 \
  -m .../DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark .../dspark.gguf -c 4096 -n 64 --temp 0.0 --seed 1 \
  -p "Write a Python function that returns the nth Fibonacci number."
```

Observed:
- **Clean, coherent output** — "...We'll implement a function that returns the nth
  Fibonacci number. The Fibonacci sequence is defined as F(0)=F(0)..." (no
  garbling; B2 exactness preserved).
- **B2 cycles execute correctly** — per-cycle `n_accept` (drafts accepted) ranging
  2..6, with ACCEPT/reject decisions and corrections. A full-accept cycle shows
  `n_accept=6 (drafts=5)` — the +1 free trailing token from verify is already
  materializing in the draft path (relevant to P4).
- **`generation: 28.25 t/s`** vs research headline **27.67 t/s** — the ~0.71×
  baseline state reproduces within run-to-run noise. This confirms the port is
  faithful and the **P0 blocker (KV replay overhead) is intact and reproducible**,
  exactly as issue468/32 §4-§5 describe.

The `main_hidden[0..5]` capture readback and the Metal drafter forward both run,
so the GPU-to-GPU capture at L40/41/42 and the drafter input/encode/output stages
all survived the port.

---

## 4. Warning cleanup (commit 13aff30)

The research code built with 10 compiler warnings (all in the DSpark additions,
inherited verbatim by the port). Cleared, behavior-preserving:

- `ds4_dspark_probe_accept` (5 warnings): the `DSPARK_LOAD_MH` macro can `goto done`
  before the `FILE *qdump_fp = NULL;` declaration, leaving the done-label
  `fclose(qdump_fp)` to read it uninitialized. Moved the declaration above the
  first macro call.
- `ds4_session_eval_dspark_b2` (4 warnings): `-Wsign-compare` (`int` vs `uint32_t
  block`) at four loop/comparison sites. `block` is the compile-time const
  `DS4_DSPARK_BLOCK_SIZE` (=5); cast to `int`.

Result: `make ds4 ds4-eval` → **0 warnings, 0 errors**. Re-ran the smoke test
(n=16): output still clean, 6 B2 cycles, ~26 t/s — behavior confirmed unchanged.

---

## 5. Confirmed baseline state and next: P0 (KV replay elimination)

The productionization kickoff is done. The `dspark-impl` branch now contains a
clean, warning-free, self-contained port of the validated DSpark code that
**reproduces the measured 0.71× end-to-end result**.

This establishes the controlled baseline required before optimization. The next
and most impactful item is **P0** (issue468/32 §5): eliminate the KV replay
overhead on partial-accept cycles (~15ms avg/cycle — the sole reason the measured
result diverges from the structurally-correct +48-60% projection). Without P0 the
gate (>1.0× baseline) cannot pass regardless of other work.

P0 options under consideration (from issue468/32 §5):
1. Verify-state reuse — convert the batch-verify KV state into the decode path's
   expected compressed-KV state for the accepted prefix.
2. Batch-decode KV compatibility — unify the KV cache layout between the batch
   verify path and the single-token decode path.
3. Prefix-1 capture extension — generalize `metal_graph_verify_suffix_tops`'s
   `capture_prefix1` (used by the MTP path for N=2) to N=5.

The compressed-KV-cache incompatibility between `metal_graph_encode_layer_batch`
(batch verify) and `metal_graph_encode_decode_layer` (decode) is the core problem
to understand before choosing an approach.

---

## 6. Commit log (dspark-impl)

| commit | summary |
|---|---|
| `cc33048` | base (impl branch tip before DSpark work) |
| `f220ebe` | dspark: port validated research code (faithful, as-validated baseline) |
| `13aff30` | dspark: clear inherited compiler warnings (behavior-preserving) |
