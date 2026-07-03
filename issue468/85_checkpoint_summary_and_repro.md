# DSpark checkpoint — current state, changes made, and exact repro steps

Date: 2026-07-03.
Branch: `dspark`.

This note is a self-contained checkpoint of the latest DSpark work in this branch.
It summarizes:

- what code and docs changed
- what was measured
- what knobs are currently considered the best-performing validated settings
- how to reproduce the recent runs exactly
- what conclusions are safe vs unsafe to draw from the data

---

## 1. Current status in one paragraph

DSpark remains functionally correct and opt-in via `--dspark`, with B2 exactness preserved. The most important shipped runtime improvement in the current branch is the **merged correction-anchor** path, enabled by `DS4_DSPARK_MERGE_CORRECTION=1`, together with `DS4_DSPARK_TARGET_POS0=1`. That path improved measured DSpark throughput on one temp=1 prompt and is the current best validated runtime configuration. A later verifier simplification sweep (`DS4_DSPARK_VERIFY_N`) reduced verifier milliseconds but hurt end-to-end throughput overall. A subsequent live verifier microprofile showed that verifier time is dominated by the layer batch compute itself, not by top-k/readback/output-head overhead. The remaining path to a meaningful additional speedup appears to require deep verifier-kernel work rather than a bounded Tier 1–2 optimization.

---

## 2. Relevant commits

Recent commits on this branch:

- `b92a0d6` — `dspark: merge correction decode into next anchor`
- `0d188a5` — `issue468: record verifier and review iterations`
- `07ac1fb` — `dspark: profile live verifier cost`
- `b69c858` — `issue468: record tier1 tier2 review round 2`

At the time of writing, `HEAD` is:

- `b69c858` — `issue468: record tier1 tier2 review round 2`

---

## 3. Files changed in this checkpoint window

### Production code

- `ds4.c`
- `ds4.h`
- `ds4_cli.c`
- `ds4_eval.c`

### Research / checkpoint notes

- `issue468/80_oppa_merge_correction_anchor_investigation.md`
- `issue468/81_verifier_n_simplification_measurement.md`
- `issue468/82_tier1_tier2_review_round1.md`
- `issue468/83_verifier_profile_round2.md`
- `issue468/84_tier1_tier2_review_round2.md`

---

## 4. What changed in code

### 4.1 Merged correction-anchor path

Main idea:

On partial accept, instead of doing a standalone correction decode immediately, the code can carry the correction token forward as the next cycle's already-emitted anchor.

This required:

- adding session state for a pending anchor
- teaching the DSpark B2 path about an anchor that has already been emitted
- updating CLI/eval callers to consume the pending anchor before calling normal sampling

Implemented pieces:

- `ds4_session_eval_dspark_b2(...)` now takes `bool first_token_already_emitted`
- added `ds4_session_take_dspark_pending_anchor(ds4_session *s, int *token)`
- added `dspark_pending_anchor_valid` / `dspark_pending_anchor` to `struct ds4_session`
- updated `ds4_cli.c` and `ds4_eval.c` to consume pending anchors before `ds4_session_sample(...)`
- env gate: `DS4_DSPARK_MERGE_CORRECTION`
- debug log prints `logical_commits` and `emitted_anchor`

Why it matters:

- it removes the separate correction decode from merged partial-accept cycles
- it reduced the measured `kv` tail cost substantially on the sampled prompt
- it is currently the main proven end-to-end DSpark runtime win in this branch

### 4.2 Verifier runtime truncation knob

A runtime override was exposed for the number of verified positions:

- `DS4_DSPARK_VERIFY_N`

This does **not** change drafter block size; it only changes how many positions the current verifier path actually verifies.

Measured result:

- lower verify depth reduces raw verifier milliseconds
- but reduces acceptance enough that the end-to-end result is worse than the full setting on the benchmark prompt

Current guidance:

- leave `DS4_DSPARK_VERIFY_N` unset for best validated end-to-end behavior

### 4.3 Verifier microprofile instrumentation

Added profiling instrumentation inside:

- `metal_graph_verify_suffix_tops(...)`

Gate:

- `DS4_DSPARK_VERIFY_PROFILE=1`

What it reports per call:

- upload time
- layer batch pass time
- output head time
- top-1 selection time
- top-index readback time
- logits readback time
- total verifier wall time

Purpose:

- determine whether the remaining verifier cost is in readback/top-k/plumbing or in the layer compute itself

Measured answer:

- overwhelmingly in the layer batch compute

Important note:

- this instrumentation is for diagnosis, not for production speed
- keep it **off** for normal runs

---

## 5. Best current runtime knobs

For the current branch, the main validated performance knobs are:

### Enable

- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`

### Leave unset

- `DS4_DSPARK_VERIFY_N`
- `DS4_DSPARK_B2_DEBUG`
- `DS4_DSPARK_VERIFY_PROFILE`
- `DS4_DSPARK_DISABLE`
- `DS4_DSPARK_CPU_ATTN`
- `DS4_DSPARK_EXACT_Q4`
- all `DS4_DSPARK_PROBE_*`
- all `DS4_DSPARK_TIME_BACKBONE*`

### Optional / experimental only

- `DS4_DSPARK_NREAL_CAP`
  - not established as a single best universal production setting in the latest measured state
- `DS4_DSPARK_FP8`
  - not a recommended default for the current validated path

---

## 6. Repro prerequisites

### Repo / branch

```sh
cd /Users/lobanov/Projects/ds4-dspark
git checkout dspark
git pull --ff-only
```

### Build

```sh
make ds4 ds4-eval
```

### Model paths used in recent runs

Target model:

```sh
../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

DSpark drafter:

```sh
../ds4/gguf/dspark.gguf
```

### Prompt paths used in the recent notes

Benchmark prompt:

```sh
issue468/prompts/chat_general.txt
```

Short sampled prompt used in the first Opp-a A/B note:

- the note describes it as a short Python fibonacci prompt
- use the exact same prompt text if you want an exact rerun of that A/B result
- do **not** compare that result directly against the benchmark-chat baseline unless you are using the same prompt

---

## 7. Shared-machine lock rule for repro

In this environment, another `ds4` / `ds4-eval` / `ds4-bench` / `ds4-server` process may hold the lock.

Replication rule used in this work:

- do **not** stop the run plan
- wait and poll every 2 minutes until the other instance exits

Convenience polling loop:

```sh
while pgrep -f '(^|/)(ds4|ds4-eval|ds4-bench|ds4-server)( |$)' >/dev/null; do
  echo "$(date '+%H:%M:%S') waiting for ds4 lock"
  pgrep -fl '(^|/)(ds4|ds4-eval|ds4-bench|ds4-server)( |$)' || true
  sleep 120
done
echo "$(date '+%H:%M:%S') ds4 lock clear"
```

---

## 8. Exact repro commands

## 8.1 Best current DSpark runtime configuration

This is the current recommended DSpark configuration for measured performance runs in this branch:

```sh
export DS4_DSPARK_TARGET_POS0=1
export DS4_DSPARK_MERGE_CORRECTION=1

./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 \
  -n 256 \
  --temp 1.0 \
  --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)"
```

After the run:

```sh
unset DS4_DSPARK_TARGET_POS0
unset DS4_DSPARK_MERGE_CORRECTION
```

---

## 8.2 Plain baseline on the benchmark prompt

Used for comparison against the DSpark benchmark prompt run:

```sh
./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 8192 \
  -n 256 \
  --temp 1.0 \
  --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)"
```

Expected note reference:

- plain generation on this benchmark prompt/setup: `35.84 t/s`

---

## 8.3 Merged correction-anchor A/B repro

### Merge OFF

```sh
export DS4_DSPARK_TARGET_POS0=1
unset DS4_DSPARK_MERGE_CORRECTION
export DS4_DSPARK_B2_DEBUG=1

./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 \
  -n 64 \
  --temp 1.0 \
  --seed 1 \
  -p "<same short Python fibonacci prompt used in issue468/80>"
```

### Merge ON

```sh
export DS4_DSPARK_TARGET_POS0=1
export DS4_DSPARK_MERGE_CORRECTION=1
export DS4_DSPARK_B2_DEBUG=1

./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 \
  -n 64 \
  --temp 1.0 \
  --seed 1 \
  -p "<same short Python fibonacci prompt used in issue468/80>"
```

### What to collect

From stderr / debug logs, collect:

- `generation: ... t/s`
- accepted drafts/cycle
- logical committed tokens/cycle
- `ms/logical-commit`

Reference results from `issue468/80`:

| mode | gen t/s | accepted drafts/cycle | logical committed/cycle | ms/logical-commit |
|---|---:|---:|---:|---:|
| merge OFF | 34.69 | 2.786 | 4.571 | 28.652 |
| merge ON  | 36.71 | 3.000 | 4.812 | 22.609 |

Cleanup:

```sh
unset DS4_DSPARK_TARGET_POS0
unset DS4_DSPARK_MERGE_CORRECTION
unset DS4_DSPARK_B2_DEBUG
```

---

## 8.4 Verifier-length sweep repro

This matches the `issue468/81` benchmark-chat setup.

Common env:

```sh
export DS4_DSPARK_TARGET_POS0=1
export DS4_DSPARK_MERGE_CORRECTION=1
export DS4_DSPARK_B2_DEBUG=1
```

Loop:

```sh
for N in 5 4 3 2; do
  export DS4_DSPARK_VERIFY_N="$N"
  echo "=== verify_n=$N ==="
  ./ds4 \
    --backend metal \
    -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --dspark ../ds4/gguf/dspark.gguf \
    -c 8192 \
    -n 256 \
    --temp 1.0 \
    --seed 1 \
    -p "$(cat issue468/prompts/chat_general.txt)"
done
```

Expected reference table from `issue468/81`:

| verify_n | gen t/s | accepted drafts/cycle | committed tokens/cycle | verify ms | ms/committed |
|---|---:|---:|---:|---:|---:|
| 5 | 30.32 | 2.312 | 4.234 | 70.181 | 25.867 |
| 4 | 32.09 | 2.400 | 4.188 | 60.584 | 23.807 |
| 3 | 27.09 | 1.738 | 3.476 | 53.188 | 26.386 |
| 2 | 24.88 | 1.406 | 2.992 | 42.303 | 26.859 |

Cleanup:

```sh
unset DS4_DSPARK_TARGET_POS0
unset DS4_DSPARK_MERGE_CORRECTION
unset DS4_DSPARK_B2_DEBUG
unset DS4_DSPARK_VERIFY_N
```

---

## 8.5 Live verifier microprofile repro

This matches `issue468/83`.

```sh
export DS4_DSPARK_TARGET_POS0=1
export DS4_DSPARK_MERGE_CORRECTION=1
export DS4_DSPARK_VERIFY_N=5
export DS4_DSPARK_B2_DEBUG=1
export DS4_DSPARK_VERIFY_PROFILE=1

./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 \
  -n 128 \
  --temp 1.0 \
  --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)"
```

Expected high-level output from `issue468/83`:

- generation: `24.46 t/s`
- accepted drafts/cycle: `2.657`
- logical committed/cycle: `4.571`
- ms/token: `32.691`

Expected average profile attribution:

| component | ms |
|---|---:|
| upload | 0.027 |
| layers | 96.218 |
| output head | 0.056 |
| top-1 kernel | 2.077 |
| read tops | 0.000 |
| read logits | 0.037 |
| verify total | 98.415 |

Cleanup:

```sh
unset DS4_DSPARK_TARGET_POS0
unset DS4_DSPARK_MERGE_CORRECTION
unset DS4_DSPARK_VERIFY_N
unset DS4_DSPARK_B2_DEBUG
unset DS4_DSPARK_VERIFY_PROFILE
```

---

## 8.6 MTP spot check repro

This is only a non-regression spot check; it is not part of the DSpark speed path.

```sh
./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp \
  -c 8192 \
  -n 256 \
  --temp 0.0 \
  --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)"
```

Reference note value:

- MTP spot check: `37.58 t/s`

---

## 9. How to parse the debug output

When `DS4_DSPARK_B2_DEBUG=1` is enabled, useful fields include:

- `n_accept`
- `logical_commits`
- `emitted_anchor`
- per-cycle timing fields including verifier timing

For reporting, use:

- **accepted drafts/cycle**
- **committed tokens/cycle** or **logical committed/cycle** as appropriate
- **delta vs oracle accepted** using oracle accepted `4.10`
- **delta vs oracle committed** using oracle committed `5.10`
- **ms/token** or **ms/logical-commit**
- **end-to-end generation t/s**

---

## 10. Safe conclusions from the current data

### Safe

- `DS4_DSPARK_MERGE_CORRECTION=1` is a real and useful optimization in the current code
- `DS4_DSPARK_TARGET_POS0=1` is part of the current best validated runtime setup
- lowering `DS4_DSPARK_VERIFY_N` is not a winning production setting on the measured benchmark prompt
- the live verifier profile shows the dominant cost is the verifier layer batch compute
- small readback/top-k/output-head cleanup is not enough to close the remaining gap

### Not safe

- do **not** compare the `36.71 t/s` sampled Python-prompt result directly to the `35.84 t/s` benchmark-chat plain baseline
- do **not** treat the profiled verifier milliseconds as directly interchangeable with the earlier unprofiled timing numbers
- do **not** claim the blocker rule is already satisfied by two consecutive no-viable-lead reviews; `issue468/84` explicitly corrected that framing

---

## 11. Current honest project state

Implemented and measured:

- merged correction-anchor path ✅
- verifier-length simplification sweep ✅
- verifier microprofile ✅
- independent Tier 1–2 review round 1 ✅
- independent Tier 1–2 review round 2 ✅

Current technical conclusion:

- no bounded Tier 1–2 code-only lead remains under the present scope
- any further meaningful verifier gain likely requires deep Metal kernel work or a scope change

Current process conclusion:

- the branch has a clean checkpoint for future continuation
- repro commands above are sufficient to rerun the latest documented measurements

---

## 12. Minimal “fastest validated current DSpark” command

If you only want the shortest reliable rerun of the best current validated DSpark settings:

```sh
export DS4_DSPARK_TARGET_POS0=1
export DS4_DSPARK_MERGE_CORRECTION=1

./ds4 \
  --backend metal \
  -m ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark ../ds4/gguf/dspark.gguf \
  -c 8192 \
  -n 256 \
  --temp 1.0 \
  --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)"
```

Then clear env:

```sh
unset DS4_DSPARK_TARGET_POS0
unset DS4_DSPARK_MERGE_CORRECTION
```
