# PR #482 (DSpark B2 rejection sampling + adaptive block sizing) — Device Test Report

Date: 2026-06-30. Tenth productionization handoff note. Independent on-device
test of GitHub PR #482 (antirez/ds4). Doc-only research/productionization
record — no code, no binaries, no runtime impact on the `dspark` branch.

## 0. PR identity

- URL: https://github.com/antirez/ds4/pull/482
- Title: "DSpark B2 rejection sampling + adaptive block sizing"
- Source: `machiabeli/ds4`, branch `work-dspark`, HEAD `3efb306674d9748479327fa25d9ed28259fc27ea`
- Authors: machiabeli (co-authored audreyt, lobanov)
- Size: +4453/−216 across 17 files (new `ds4_dspark_runtime.c/.h`, `gguf-tools/deepspec/`, +2477 lines in ds4.c)
- Builds on PR #480 (@audreyt): a from-scratch DSpark loader/Metal drafter — a DIFFERENT lineage from the local `dspark` branch's own B2/P0 implementation.

## 1. PR's claimed results (from the PR body)

> M5 Max 128GB, Q2 base + Q4K drafter
> | Prompt | Baseline | DSpark | Speedup |
> | Repetitive JSON (30x) | 38.72 | 52.81 | +36.4% |
> | Diverse JSON (20 items) | 38.85 | 42.29 | +8.9% |
> | Markdown tables | 42.54 | 45.21 | +6.3% |
> | Thinking: prime checking | 14.71 | 17.61 | +19.7% |
> "All outputs token-identical to non-speculative greedy decode at temp=0.
>  Lossless by construction at any temperature."

Key PR features: (1) B2 rejection sampling `b2_rejection_sample` (170 lines,
log-space stable); (2) off-by-one fix (verify `row[i]` predicts `drafts[i+1]`);
(3) RNG persistence in session struct; (4) adaptive block sizing
(`DS4_DSPARK_ADAPTIVE=1`, block=2 default → escalate after full commit).

## 2. Test environment

- Hardware: Apple M5 Max 128GB (the same device the PR's numbers claim).
- OS: macOS. Backend: Metal (the PR's target).
- Target model: `/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` (the Q2 base).
- Drafter: `/Users/lobanov/Projects/ds4/gguf/dspark.gguf` (the validated 11.5 GB Q4K drafter — same drafter the local `dspark` branch uses).
- Isolation: PR built in a throwaway git worktree at `3efb306`; the local `dspark` branch was NOT modified.

## 3. Build

Commands (in the PR worktree):
```
git remote add machiabeli https://github.com/machiabeli/ds4.git
git fetch machiabeli work-dspark
git worktree add /tmp/pr482-test machiabeli/work-dspark
cd /tmp/pr482-test && make
```
Result: **builds clean, exit 0, 0 warnings / 0 errors**. Binary `ds4` = 1,281,536 bytes. (`ds4-eval`, `ds4-bench`, `ds4-server`, `ds4-agent` all link clean.)

The PR loads the DSpark drafter via **`--mtp DSpark.gguf`** (not a `--dspark` flag); it detects the model kind (`kind=dspark`) and engages the block-spec runtime. Sanity load confirmed: `ds4: draft model loaded: ... (kind=dspark, draft=5, runtime_mtp=yes)`.

## 4. Invocation conventions (env vars)

- `DS4_SPEC_TEMP=<t>` — engage B2 rejection sampling (stochastic; the PR's headline mode). Unset = greedy-match path.
- `DS4_DSPARK_ADAPTIVE=1` — adaptive block sizing (block=2 default, escalate to 5 after a full commit).
- `DS4_MTP_TIMING=1` — per-cycle timing to stderr (`drafted=N committed=M snapshot=.. verify=.. replay=.. total=..`).

## 5. Test matrix + results

All runs: `ctx=8192`, `temp=0.0`, identical model+drafter, n as noted, single run each (run-to-run noise ~±0.5 t/s on this device). `MODEL` and `DSPARK` are the paths in §2.

### 5.1 Code prompt (creative, low acceptance) — n=256

Prompt: `"Write a Python function that reverses a singly linked list, with comments."`

| # | command | gen t/s | vs baseline |
|---|---|---|---|
| A | `./ds4 -m $MODEL -c 8192 -n 256 --temp 0.0 -p "$P"` | **39.19** | 1.00× |
| B | `./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 256 --temp 0.0 -p "$P"` (PR greedy default) | **27.73** | 0.71× |
| C | `DS4_SPEC_TEMP=0.6 ./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 256 --temp 0.0 -p "$P"` (PR B2) | **29.60** | 0.76× |
| D | `DS4_SPEC_TEMP=0.6 DS4_DSPARK_ADAPTIVE=1 ./ds4 ... ` (PR B2 + adaptive) | **26.77** | 0.68× |

### 5.2 Repetitive JSON prompt (the PR's claimed +36% regime) — n=400

Prompt: `'Generate a JSON array of 30 user objects. Each object must have exactly these keys in this order: "id" (incrementing integer starting at 1), "name", "email", "age". Use the same template for every object. Start: [{"id":1,'

| # | command | gen t/s | vs baseline | PR claim |
|---|---|---|---|---|
| A | `./ds4 -m $MODEL -c 8192 -n 400 --temp 0.0 -p "$P"` | **38.72** | 1.00× | — |
| B | `./ds4 -m $MODEL --mtp $DSPARK ... ` (PR greedy default) | **26.71** | 0.69× | — |
| C | `DS4_SPEC_TEMP=0.6 ./ds4 ... ` (PR B2) | **27.53** | 0.71× | — |
| D | `DS4_SPEC_TEMP=0.6 DS4_DSPARK_ADAPTIVE=1 ./ds4 ... ` (PR B2 + adaptive) | **31.07** | **0.80×** | claimed +36% (52.81) |

**The PR's headline +36% on repetitive JSON does NOT reproduce on this device — measured −24% (31.07 vs 38.72).** The PR is slower than baseline on every prompt/configuration tested.

### 5.3 Head-to-head vs the local `dspark` branch (same prompts, n=400)

| prompt | baseline | PR (best mode) | local `dspark` (P0, no replay) |
|---|---|---|---|
| Code (creative) | 38.56 | 27.24 (B2) | 27.07 |
| Repetitive JSON | 38.82 | 29.40 (adaptive) | 25.90 |

The PR and the local branch are in the **same ballpark** (~0.68–0.80× baseline) on this device. Neither crosses the gate. (The local branch's lower JSON number here reflects its Bug #1 "growing window" fix slightly hurting acceptance; the stale-window config was ~5% faster — a documented local fallback.)

## 6. Root-cause: the PR runs WITHOUT P0 (KV replay on every cycle)

Per-cycle timing (`DS4_MTP_TIMING=1`) on the PR's structured run exposed the cause directly. The PR pays a KV **replay** on 100% of cycles — including every full-accept cycle:

Example PR cycles (repetitive JSON, `DS4_SPEC_TEMP=0.6 DS4_DSPARK_ADAPTIVE=1`):
```
drafted=5 committed=3 verify=64.8ms replay=77.9ms  total=144.2ms
drafted=5 committed=5 verify=64.0ms replay=131.8ms total=196.9ms   <- FULL accept, still replays!
drafted=5 committed=5 verify=67.4ms replay=134.6ms total=202.3ms   <- FULL accept, still replays!
drafted=5 committed=1 verify=65.3ms replay=26.8ms  total=92.6ms
```
Aggregate over 55 cycles: **acceptance 63.9%** (genuinely high — the PR *should* win here), but **100% of cycles replayed**, avg replay 74.9ms. **19 of 19 full-accept cycles paid the replay.** A full-accept cycle = verify 64ms + replay 132ms = 196ms / 5 commits = 39ms/tok ≈ baseline. The replay erases the speculation win.

This is **exactly the problem the local branch's P0 already solved** (per-position frontier checkpointing → `kv=0.0` on full-accept cycles; visible in local timing: `n_accept=6 drafts=5 ... kv=0.0 total=106ms`). The PR's own commit history confirms it: commit `e12e664` "[dspark] Revert to correct replay path, document compression schedule" — the PR reverted away from an attempted replay-elimination because of the compressed-KV overflow, the same blocker P0 later solved with per-position captures.

## 7. Correctness claims — verified, do NOT hold on this device

### 7.1 "Token-identical greedy at temp=0" — FAILS

Commands (n=200, repetitive JSON prompt, temp=0.0):
```
./ds4 -m $MODEL -c 8192 -n 200 --temp 0.0 -p "$P" | md5                           # plain greedy
./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 200 --temp 0.0 -p "$P" | md5             # PR default
DS4_SPEC_TEMP=0.6 ./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 200 --temp 0.0 -p "$P" | md5   # PR + B2
```
Hashes:
- plain greedy:        `803c62bf431e9256dc1c37dfc5b86c14`
- PR default (greedy): `73a23a43c438f58ecd46797dc76973b3`  (≠ greedy)
- PR + B2 (temp=0):    `62024951de377b73b26fe1f789137e36`  (≠ greedy)

**Neither PR mode reproduces plain greedy at temp=0.** Root cause: `metal_graph_verify_suffix_tops` has a documented batch-vs-decode argmax divergence (batch reductions can flip greedy tokens vs sequential decode). This is the same reason the local branch's temp-aware B2 experiment failed its go/no-go (issue468/41). The claim is not achievable on the fast batch verifier used by both the PR and the local branch.

### 7.2 "Lossless B2 at any temperature" — NOT lossless as implemented

The PR's `b2_rejection_sample` computes `p/q` against the drafter softmax, but the draft tokens are chosen by `sample_argmax` (greedy), not sampled from `q`. Rejection sampling is only distribution-correct if proposals are actually drawn from the proposal distribution. (This was flagged independently by codex review of the PR.) So the PR's stochastic B2 is **not actually lossless**.

## 8. Conclusions

1. **The PR does not reproduce its claimed speedups on this M5 Max.** Every tested configuration (greedy default, B2 stochastic, B2+adaptive) is slower than plain-decode baseline on both creative and structured prompts (0.68–0.80×). The headline "+36% on repetitive JSON" measured −24%.
2. **Root cause is the absence of P0.** The PR pays a KV replay on 100% of cycles (including full-accept), averaging ~75ms/cycle. The local branch's P0 eliminated this (per-position frontier checkpointing → `kv=0.0` on full-accept). The PR's commit log confirms it reverted away from replay-elimination.
3. **The PR's correctness claims do not hold on this device.** "Token-identical greedy at temp=0" fails (batch-vs-decode argmax divergence); "lossless B2 at any temp" is not lossless as implemented (greedy proposals, not sampled-from-q).
4. **No new lever for the local branch.** The PR's useful ideas (off-by-one fix, RNG persistence, noncausal kernel) are already present in the local branch; its B2 residual sampler is a quality/correctness idea (not perf); its adaptive block sizing hurts the local creative prompt and is mixed on structured. Testing the PR did not surface any path to the perf gate that the local codex-review loop (issue468/41, CONVERGED) had missed.
5. **The PR and the local branch are performance-comparable on this device** (both ~0.7–0.8× baseline), but for different reasons: the PR lacks P0 (replay-bound), the local branch has P0 but is acceptance-bound on creative prompts with a fast-flat baseline.

## 9. Reproducibility

Exact commands (from §5, with `$MODEL`/`$DSPARK` = the paths in §2, `$P` = the prompt):
```
# Build the PR
git remote add machiabeli https://github.com/machiabeli/ds4.git
git fetch machiabeli work-dspark
git worktree add /tmp/pr482-test machiabeli/work-dspark
cd /tmp/pr482-test && make   # 0 warnings, exit 0

# Baseline
./ds4 -m $MODEL -c 8192 -n 256 --temp 0.0 -p "$P"

# PR default (greedy-match)
./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 256 --temp 0.0 -p "$P"

# PR + B2 rejection sampling (headline mode)
DS4_SPEC_TEMP=0.6 ./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 256 --temp 0.0 -p "$P"

# PR + B2 + adaptive block sizing
DS4_SPEC_TEMP=0.6 DS4_DSPARK_ADAPTIVE=1 ./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 256 --temp 0.0 -p "$P"

# Per-cycle timing (diagnose replay)
DS4_MTP_TIMING=1 ./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 256 --temp 0.0 -p "$P" 2>timing.log

# Greedy-exactness check (hashes must match for the claim)
./ds4 -m $MODEL -c 8192 -n 200 --temp 0.0 -p "$P" 2>/dev/null | md5
./ds4 -m $MODEL --mtp $DSPARK -c 8192 -n 200 --temp 0.0 -p "$P" 2>/dev/null | md5
```
All raw outputs captured during testing: `/tmp/pr_struct_timing.txt` (PR per-cycle, structured prompt), `/tmp/mine_struct.txt` (local branch per-cycle, structured prompt).
