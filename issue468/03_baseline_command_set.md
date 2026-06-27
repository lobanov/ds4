# Phase 0 — Repeatable Baseline Command Set (local Metal)

Every command assumes:

- Repo root is the current directory.
- `make` has been run and `./ds4`, `./ds4-bench` exist.
- **Explicit model paths** (set these once per shell; do not rely on the
  `ds4flash.gguf` symlink so runs are unambiguous):
  ```sh
  MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
  MTP="../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
  ```
  The main GGUF is the same file `ds4flash.gguf` symlinks to; `MTP` is fetched
  via `./download_model.sh mtp` if missing.
- **Only one `ds4`/`ds4-bench` process at a time** (instance lock; per
  `AGENT.md`). Do not run server and bench concurrently.

## 0. Machine / thermal protocol (required for comparable numbers)

Per `CONTRIBUTING.md` speed-regression rules, fix the following *before* the
first run and keep them identical across target-only vs `--mtp`:

- Power: `--power 100` (or note the fixed percentage used).
- Plug in to wall power; let the machine reach a steady temperature.
- Quit other heavy GPU/CPU processes.
- Use the same prompt file, context, `--gen-tokens`, and `--quality` setting for
  a given comparison pair.
- Record: machine name, chip, macOS version, model file + quant, ambient state,
  `ds4 --version` / git SHA. Put this in the header of `04_run_matrix.md`.

Capture the machine fingerprint once:

```sh
{
  echo "# machine fingerprint"
  sysctl -n machdep.cpu.brand_string
  sw_vers | tr '\n' ' '; echo
  system_profiler SPHardwareDataType | grep -E 'Chip|Memory|Model Name'
  echo "git: $(git -C . rev-parse --short HEAD 2>/dev/null || echo unknown)"
} | tee issue468/baseline/machine.txt
```

## 1. Build

```sh
make clean
make          # default Metal build; produces ./ds4 ./ds4-bench ./ds4-server ./ds4_test
```

Optional sanity (only if a model + Metal are available, takes a few minutes):
```sh
make test     # or narrower: ./ds4_test --logprob-vectors
```

## 2. Target-only baseline — structured CSV via `ds4-bench`

This is the primary target-only throughput baseline. It is the cleanest because
`ds4-bench` already emits the standard CSV used for speed regressions.

```sh
mkdir -p issue468/baseline
./ds4-bench \
  -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --step-incr 2048 \
  --gen-tokens 256 --power 100 \
  --csv issue468/baseline/bench_target_only.csv \
  2> issue468/baseline/bench_target_only.stderr.log
```

Notes:
- `promessi_sposi.txt` is long-form prose (≈"chat/general" prompt class).
- For a **code** prompt class at a fixed context, use the CLI command in §4
  instead (the bench needs a very long file to walk context frontiers).
- `gen_tps` in the CSV is the target-only greedy decode throughput at each
  frontier. This is the **denominator** every speculative number must beat.

## 3. Target-only baseline — CLI, same code path as the `--mtp` run

Run the CLI *with* `--mtp` loaded but spec disabled, so the harness cost is
identical to the `--mtp` run in §4. Use `DS4_MTP_SPEC_DISABLE` and the same
prompt.

```sh
# General/chat prompt:
DS4_MTP_SPEC_DISABLE=1 ./ds4 -m "$MODEL" \
  --mtp "$MTP" --mtp-draft 2 \
  -c 8192 --temp 0 -n 256 --power 100 --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  2> issue468/baseline/cli_target_only_chat.log

# Code prompt:
DS4_MTP_SPEC_DISABLE=1 ./ds4 -m "$MODEL" \
  --mtp "$MTP" --mtp-draft 2 \
  -c 8192 --temp 0 -n 256 --power 100 --seed 1 \
  -p "$(bash issue468/prompts/code_humaneval.sh)" \
  2> issue468/baseline/cli_target_only_code.log
```

The CLI prints `ds4: prefill: … t/s, generation: … t/s` at the end
(`ds4_cli.c:532`) — record `generation` t/s.

## 4. `--mtp` speculative baseline — CLI with full timing

Same prompts, same seed, spec **enabled** (no `DS4_MTP_SPEC_DISABLE`). Turn on
the full diagnostic surface from `01_existing_instrumentation.md`:

```sh
export DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1 DS4_MTP_CONF_LOG=1 DS4_MTP_PROBE=1
./ds4 -m "$MODEL" \
  --mtp "$MTP" --mtp-draft 2 \
  -c 8192 --temp 0 -n 256 --power 100 --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  > issue468/baseline/cli_mtp_chat.out \
  2> issue468/baseline/cli_mtp_chat.log

./ds4 -m "$MODEL" \
  --mtp "$MTP" --mtp-draft 2 \
  -c 8192 --temp 0 -n 256 --power 100 --seed 1 \
  -p "$(bash issue468/prompts/code_humaneval.sh)" \
  > issue468/baseline/cli_mtp_code.out \
  2> issue468/baseline/cli_mtp_code.log
unset DS4_MTP_TIMING DS4_MTP_SPEC_LOG DS4_MTP_CONF_LOG DS4_MTP_PROBE
```

What each output file contains:
- `*.out` — the generated tokens (the exact greedy stream; must match the
  target-only `.log` token stream for correctness — see §6).
- `*.log` — per-cycle `mtp timing …` lines (draft/snapshot/verify/prefix/replay/
  total ms + drafted/committed), `mtp conf …` margin lines, `mtp probe …`
  position-1 hit-rate lines, and the final
  `ds4: prefill: … t/s, generation: … t/s` summary.

### Optional: isolate each verifier path at depth 2

To measure the cost of each verifier strategy separately (feeds Phase 1), rerun
the §4 chat command once per knob, keeping everything else identical:

```sh
# Exact N=2 verifier (quality path)
DS4_MTP_STRICT=1 …  (same ./ds4 command) 2> issue468/baseline/cli_mtp_chat_strict.log
# Batch verifier (no decode2)
DS4_MTP_BATCH_VERIFY=1 … 2> issue468/baseline/cli_mtp_chat_batch.log
# Prefix-1 capture partial path
DS4_MTP_CAPTURE_PREFIX1=1 … 2> issue468/baseline/cli_mtp_chat_prefix1.log
# Snapshot+restore every cycle
DS4_MTP_FORCE_SNAPSHOT=1 … 2> issue468/baseline/cli_mtp_chat_snapshot.log
```

These are diagnostic isolations; the "real" `--mtp` number is the unmodified §4
run.

## 5. Per-layer target decode cost (denominator sanity)

To see what a single target decode step costs layer-by-layer (independent of
spec), run a short target-only CLI decode with layer profiling:

```sh
DS4_DECODE_PROFILE_DETAIL=1 ./ds4 -m "$MODEL" \
  -c 4096 --temp 0 -n 32 --power 100 --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  2> issue468/baseline/target_decode_profile.log
```

This gives the layer-level ms breakdown that the
`draft+verify+replay / accepted` speculative math must beat.

## 6. Correctness check (exact greedy preservation)

The generated token stream from §3 (target-only) and §4 (`--mtp`) must be
**byte-identical** for the same prompt/seed/ctx. Quick check:

```sh
# Strip CLI progress noise, compare only the emitted tokens:
./ds4 -m "$MODEL" -c 8192 --temp 0 -n 256 --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  > /tmp/tgt.out 2>/dev/null
./ds4 -m "$MODEL" \
  --mtp "$MTP" --mtp-draft 2 \
  -c 8192 --temp 0 -n 256 --seed 1 \
  -p "$(cat issue468/prompts/chat_general.txt)" \
  > /tmp/mtp.out 2>/dev/null
diff -q /tmp/tgt.out /tmp/mtp.out && echo "OK: greedy stream identical"
```

A mismatch here is a P0 correctness regression and blocks Phase 0 sign-off.

## 7. What to record

For every run, drop the raw files into `issue468/baseline/` (names as above) and
fill the corresponding row in `04_run_matrix.md`. The numbers that matter most
for the Phase 0 decision:

- target-only `gen_tps` (from §2 CSV or §3 summary line)
- `--mtp` `generation … t/s` (§4 summary line)
- per-cycle mean `us_total` and its split (`us_draft` / `us_verify`)
- `mean_committed` (accepted tokens per cycle, incl. first target token)
- position-1 acceptance from `mtp probe hit=H/T` (aggregate over the run)
- the accept/partial/miss counts derived from `DS4_MTP_SPEC_LOG`
