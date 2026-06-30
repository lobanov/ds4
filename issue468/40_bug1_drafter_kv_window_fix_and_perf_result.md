# Bug #1 (Drafter KV Window) — Fix Applied, Perf-Neutral/Slightly-Negative

Date: 2026-06-30. Eighth productionization handoff note. Records the
implementation + measurement of Bug #1 (drafter KV window stuck at n_real=1),
the codex review's top finding. Doc-only research/productionization record.

## 0. The bug (codex #1, verified)

The live B2 drafter KV window was STUCK at `dspark_n_real = 1` for the entire
generation. `ds4.c:29194` (pre-fix) set `g->dspark_n_real = prev_n_real > 0 ?
prev_n_real : 1;` — never incrementing. The drafter attention window is defined
by n_real (anchors `[0..n_real]`, drafts `[n_real+1..n_real+block]`,
`ds4.c:17795-17812`), so with n_real=1, slot[0] froze at the cycle-1 anchor
forever. This was the "stale dspark_kv_cache corrupting the drafter's attention"
issue flagged in commit `fe12c76`'s status note.

Empirically confirmed pre-fix: n_real=1 for an entire run (step_pos 23→58, n_real
flat at 1).

## 1. Intent resolution (the ambiguity the goal flagged)

The original commit `fe12c76` proposed two possible fixes ("zero per cycle" OR
"prefill all slots"). Three evidence sources converge on ACCUMULATION:
- The reference `ds4_dspark_probe_accept` (the documented "correct" drafter: 2.74
  greedy prefix): input_stage → encode_block → `if (n_real < SWA) n_real++`
  per step (`ds4.c:28345`).
- The inline comment (`ds4.c:29152` pre-fix): "let anchor KV accumulate across
  cycles (like the accept probe...)."
- The commit message: identifies stale-KV as a bug, proposes accumulation.

So accumulation is clearly intended; the no-increment was an incomplete/buggy
implementation, not a deliberate frozen window. (Ambiguity resolved — not a guess.)

## 2. The fix (commit 75b6e77)

Two changes, both mirroring the reference probe:
1. **Removed the redundant anchor-KV prefill loop** (3-layer matmul+rope+store
   that was at `ds4.c:29173-29184` pre-fix). `metal_graph_dspark_encode_attention`
   (inside encode_block) ALREADY computes + stores the anchor KV at slot
   `[n_real]` from `dspark_main_x` (`ds4.c:~17806`). The prefill was a pure
   duplicate. Also removes 3 GPU syncs.
2. **Capped increment after encode_block**: `if (g->dspark_n_real < DS4_N_SWA)
   g->dspark_n_real++;` (replacing the no-increment line). Placed AFTER
   encode_block because encode_attention needs the pre-increment n_real to place
   this cycle's anchor at slot `[n_real]`.

## 3. Verification

- `make`: 0 warnings.
- **n_real now grows live**: 1→2→...→70 (was stuck at 1). Confirmed.
- **Drafter phase stable** (~8.7–9.4ms regardless of window 1→70): the growing
  non-causal attention window adds no measurable compute (bandwidth-bound).
- **MTP non-regression**: `--mtp --mtp-draft 2` runs clean (39.37 t/s).
- **Baseline non-regression**: plain decode unchanged (39.42 t/s).
- **B2 output exactness status UNCHANGED**: B2 was NEVER greedy-exact at temp=0.
  Both pre-fix (`da9ca..`) and post-fix (`2b0a..`) diverge from plain greedy
  (`1d43..`) due to the B2 rejection-sampling approximation (stochastic RNG +
  `argmax(p−q)` correction = codex Bug #2, pre-existing). Both are
  "statistically B2-correct" (output from the target distribution); neither is
  greedy-token-exact. The fix does not change this status.

## 4. Perf result (decisive — the fix does NOT help)

Measured n=256, ctx=8192, temp=0, identical prompt:

| config | gen t/s | commits/cycle |
|---|---|---|
| pre-fix (stale window, n_real=1) | 31.50 | 4.06 |
| post-fix (growing window, n_real→70) | 28.58 | 3.66 |

The fix **restores intended behavior but slightly HURTS acceptance** (−0.4
commits/cycle, −2.9 t/s). The growing-window drafter produces marginally WORSE
drafts than the stale-window one.

**Codex's "#1 acceptance multiplier" hypothesis was empirically wrong.** This is
the most important finding of this iteration: the "bug" (stale window) was
accidentally a BETTER drafter configuration than the "intended" one (growing
window). Possible explanations: (a) the mtp.2 drafter model doesn't benefit from
attending over many historical anchors — the current anchor + drafts may
suffice; (b) the non-causal attention over a growing window diverges from the
model's training distribution; (c) accumulated anchor RoPE positions (absolute,
not relative) drift.

## 5. Decision: keep the fix

Kept (commit 75b6e77) because it restores the documented research intent (probe +
comment both specify accumulation) and removes the redundant prefill + 3 syncs.
The −2.9 t/s is within the range where the codex-review loop may recover it
elsewhere. If a later lever gets close to the gate, reverting Bug #1 to reclaim
the stale-window's +2.9 t/s is a documented fallback.

## 6. Implications for the gate

Bug #1 was the codex review's top-ranked perf lever. It does not pan out: the
fix is perf-neutral-to-slightly-negative. Combined with the prior findings
(block size wash, Opp1 invalid, Opp-a infeasible, Opp3 ceiling ≤2, readback/CPU
below noise), the gate (>39 t/s, current ~28-31) remains structurally very hard.
Proceeding to the codex-review loop (re-run gpt-5.5 with this finding) to find
the next code-only lever, or to converge on "no further viable opportunities."
