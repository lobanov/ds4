# Verifier optimization attempt 1 — runtime verify-length simplification

Date: 2026-07-02.

## Lead

Test the lowest-risk verifier optimization already present in code: runtime verify-length truncation via `DS4_DSPARK_VERIFY_N`, keeping the same batch verifier (`metal_graph_verify_suffix_tops`) but verifying fewer draft positions.

This is a verifier-path simplification lead, not a new kernel.

## Independent verification before measurement

Re-read the live verifier path:

- `ds4_session_eval_dspark_b2(...)` already computes `eff_block` from `DS4_DSPARK_VERIFY_N`
- verification still goes through `metal_graph_verify_suffix_tops(...)`
- the drafter block size remains 5; only the verified suffix length changes
- this preserves the existing B2 acceptance machinery and does not touch baseline or MTP paths unless `--dspark` is active

## Measurement setup

Common settings:

- model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- drafter: `dspark.gguf`
- prompt: `issue468/prompts/chat_general.txt`
- `ctx=8192`
- `n=256`
- `temp=1.0`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- `DS4_DSPARK_B2_DEBUG=1`

Baseline plain decode on the same prompt/setup:

- plain generation: **35.84 t/s**

Oracle references used for gap reporting:

- oracle accepted drafts/cycle: **4.10**
- oracle committed tokens/cycle: **5.10**

## Results

| verify_n | gen t/s | accepted drafts/cycle | delta vs oracle accepted | committed tokens/cycle | delta vs oracle committed | verify ms | ms/committed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 30.32 | 2.312 | -1.788 | 4.234 | -0.866 | 70.181 | 25.867 |
| 4 | **32.09** | **2.400** | **-1.700** | **4.188** | **-0.912** | 60.584 | **23.807** |
| 3 | 27.09 | 1.738 | -2.362 | 3.476 | -1.624 | 53.188 | 26.386 |
| 2 | 24.88 | 1.406 | -2.694 | 2.992 | -2.108 | **42.303** | 26.859 |

## Interpretation

### What improved

- verify time falls monotonically as expected:
  - `70.2ms -> 60.6ms -> 53.2ms -> 42.3ms`
- `verify_n=2` is the only configuration that clearly beats the task's raw verify-time target (`<51ms`)

### What did not improve enough

- reducing verify depth also reduces accepted drafts/cycle and committed tokens/cycle sharply
- the saved verify time does **not** amortize the lost accepted work
- best end-to-end setting is `verify_n=4`, but it still only reaches **32.09 t/s**, below:
  - the local plain baseline **35.84 t/s** on this benchmark prompt
  - the stronger Opp-a-only sampled prompt result (**36.71 t/s**)

## Main conclusion

Runtime verify-length simplification is **not** the gate-crossing verifier optimization.

- It can reduce verifier cost substantially.
- But the acceptance loss dominates.
- The best measured simplified setting (`verify_n=4`) remains below baseline.
- The only setting that meets the raw verifier-ms goal (`verify_n=2`) is much worse end-to-end.

So this lead is a measured negative result for the current goal:

- **verify path simplification alone does not close the live-vs-oracle gap**
- a successful verifier optimization likely needs a **better per-position cost curve** (custom microbatch verifier / kernel work), not just fewer verified positions

## Non-regression spot checks

- build: `make ds4 ds4-eval` warning-clean
- plain decode unchanged on the benchmark prompt: **35.84 t/s**
- MTP spot check still runs: **37.58 t/s**

## Outcome of this iteration

This iteration attempted the simplest verifier optimization already available in the code and measured it honestly. It reduced verify ms, but did not improve end-to-end DSpark enough to beat baseline. The next verifier task, if continued, should be a real verifier-kernel optimization rather than more suffix-length truncation.
