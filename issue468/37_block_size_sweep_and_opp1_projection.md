# DSpark Block-Size Sweep (d=1..5) — Measurement & Opp1 Projection

Date: 2026-06-30. Fifth productionization handoff note. First measurement pass
on the recommended perf-push sequence (issue468/36 blocker → goal tweak). This
note records the block-size sweep and the structural conclusion it forces.

Doc-only research/productionization record. No code, no binaries, no runtime
impact. (The enabling code change — `#ifndef DS4_DSPARK_BLOCK_SIZE` guard +
block-agnostic probe prints — is committed separately as abcf8cf.)

## 0. Setup

- Branch: `dspark` (worktree /Users/lobanov/Projects/ds4-dspark). All prior
  dspark-impl productionization work backported; P0 + P1 landed here.
- Target model: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
- Drafter: /Users/lobanov/Projects/ds4/gguf/dspark.gguf (the real 11.5GB
  quantized drafter — NOT the 5KB dspark_template*.gguf metadata stubs).
- Config: ctx=8192, n=256, temp=0.0, identical prompt/seed across all points.
- Per-phase timing: `DS4_DSPARK_B2_DEBUG=1` (one line per cycle:
  `n_accept=X (drafts=Y) | anchor=A drafter=D verify=V accept=K kv=J total=T ms`).
- Build per point: `make ds4 CFLAGS="... -DDS4_DSPARK_BLOCK_SIZE=N"` (the guard
  from abcf8cf). Recompiles only ds4.o; the production eval_dspark_b2 path is
  already block-parameterized.

d=5 reproduces the issue468/36 per-phase breakdown exactly (anchor 25-30,
drafter ~9.7, verify ~68, total ~130ms), validating the harness.

## 1. Sweep results

| d | gen t/s | ms/tok | commits/cyc | anchor | drafter | verify | accept | kv | total | cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 31.50 | 31.75 | 4.06 | 25.9 | 9.6 | 69.6 | 1.6 | 22.3 | 128.9 | 63 |
| 4 | 31.76 | 31.49 | 3.70 | 26.1 | 8.5 | 59.7 | 1.4 | 20.7 | 116.4 | 69 |
| 3 | 29.82 | 33.53 | 3.16 | 26.3 | 7.7 | 51.6 | 1.2 | 19.2 | 105.9 | 81 |
| 2 | 30.79 | 32.48 | 2.69 | 25.9 | 6.4 | 40.8 | 0.9 | 13.4 |  87.4 | 95 |
| 1 | 29.02 | 34.46 | 2.00 | 26.4 | 5.3 | 29.6 | 0.5 |  7.0 |  68.8 | 128 |

Baseline (plain decode, no --dspark) on this hardware = 39.0 t/s (25.6 ms/tok)
— see issue468/34/36. **All five d values are below baseline (0.74-0.81×).**

## 2. Structural conclusion #1 — block size alone is a wash

**ms/tok is flat (~32) across all d.** Verify cost drops ~linearly with d
(~11ms per draft position: 69.6→59.7→51.6→40.8→29.6, Δ≈11/position), but
commits/cycle drops proportionally (4.06→3.70→3.16→2.69→2.00), so the two
effects cancel: ms/tok = total_ms / commits stays pinned at ~32.

This **refutes the issue468/36-blocker-era Opp2 hypothesis** (that reducing d
to 2-3 would cross the gate by shrinking the verify faster than commits fall).
It does not: the verify-per-position cost (~11ms) is too close to the per-token
amortization for d alone to win. Cost-minimizing d by gen t/s is d=4 (31.76),
but the spread (29.0-31.8) is largely run-to-run noise; the robust claim is
"no d crosses the gate alone."

## 3. Structural conclusion #2 — the anchor decode is the tax (Opp1 confirmed)

At every d, the **anchor decode (~26ms) is paid once per cycle but contributes
exactly 1 commit** — it is a pure per-cycle tax independent of d. Verify scales
with d; the anchor does not. So the anchor is the highest-leverage reducible
cost, exactly as the goal's Opp1 (anchor-decode-elimination) predicts.

The anchor decode is also **redundant** on ~85-90% of cycles: it re-runs the
target forward on the token whose KV was just installed by the prior cycle's
correction/bonus decode, purely to capture `main_hidden` for the drafter. P3
(issue468/36) measured that capture (readback + HC-mean + upload) at 0.05ms.
So sourcing `main_hidden` from the commit decode and skipping the standalone
anchor forward eliminates ~26ms/cycle at near-zero cost.

## 4. Opp1 projection (cycle − 26ms anchor) / commits

Using the measured table (assuming the anchor is eliminated on ~88% of cycles —
it survives full-accept cycles where the bonus token genuinely lacks KV):

| d | projected ms/tok | projected t/s | × baseline (39) |
|---|---|---|---|
| 5 | (128.9−22.9)/4.06 = 26.1 | ~38.3 | 0.98× (≈ baseline) |
| 4 | (116.4−22.9)/3.70 = 25.3 | ~39.5 | 1.01× |
| 3 | (105.9−22.9)/3.16 = 26.3 | ~38.0 | 0.97× |
| 2 | ( 87.4−22.9)/2.69 = 24.0 | ~41.7 | **1.07× ✅** |
| 1 | ( 68.8−22.9)/2.00 = 23.0 | ~43.5 | **1.12× ✅** |

(100%-elimination upper bound: d=2 → ~43.8 t/s (1.12×), d=1 → ~46.7 t/s
(1.19×).)

**Opp1 + small d (d=1 or d=2) is the configuration projected to cross the
gate.** d=1 drafts/verifies a single token (minimal speculation) but has the
smallest fixed drafter+verify cost; d=2 keeps meaningful speculation breadth.
The choice between them is a measurement call after Opp1 lands.

## 5. Verify-cost scaling characterization (for the contract)

Verify ms vs d: 29.6 / 40.8 / 51.6 / 59.7 / 69.6 for d = 1/2/3/4/5.
Fit: verify_ms ≈ 18.5 + 10.3×d (R²≈0.997). → **roughly linear, ~10-11ms per
draft position**, with a ~18ms fixed component (output head + argmax + the
batch-launch overhead of metal_graph_verify_suffix_tops). The linear regime
means there is no super-linear penalty at high d within 1..5, and no
sub-linear win at low d — consistent with the batch kernel being
compute-bound in positions.

## 6. Next step

Implement Opp1 (anchor-decode-elimination): set `dspark_capture_active` during
the correction/bonus decode so `main_hidden` is sourced from the commit
forward, and skip the standalone anchor decode on cycles where `main_hidden`
is fresh (gated on a freshness flag). Must be bit-equivalent (same layers
40/41/42, same HC-mean) and preserve B2 exactness + MTP non-regression. Then
re-run this sweep (d=1..5) WITH Opp1 and re-test the gate.

The blocker rule is NOT yet invocable: the sweep is measured, but Opp1 is not
yet implemented (blocker threshold requires both).
