# DSpark Batch-Cap Pre-Benchmark Review V2

**Verdict:** benchmark-ready for the planned short cap sweep. I found no
high-priority blocker in the reviewed scope that should stop the throughput
pass, assuming the batched arms set both `DS4_DSPARK_SCHEDULE_BATCHED=1` and
`DS4_DSPARK_SCHEDULE_BATCH_N={5,4,3}`.

Reviewed scope: `ds4.c`, `ds4.h`, `ds4_spec_bench.c`, `tests/ds4_test.c`. I
did not edit files.

## Remaining Risks

- `DS4_DSPARK_SCHEDULE_BATCH_N` is a cap only inside the batched scheduled
  path. If `DS4_DSPARK_SCHEDULE_BATCHED` is unset, or if `DS4_DSPARK_VERIFY_K`
  disables scheduled verify, the cap is effectively inactive. Require JSON
  confirmation: `scheduled_verify=true`, `schedule_batched=true`, and
  `schedule_batch_limit` matching the arm.
- With `DS4_DSPARK_TIMING` unset, top-level `decode_ms` and
  `tokens_per_second` are valid, but DSpark per-cycle `*_ms` fields are
  intentionally zero. Do not use per-cycle timing means in the throughput pass.
- The fresh-session harness path is clean for this benchmark, but restored
  payload/snapshot DSpark semantics are still not proven by this sweep.
  `ds4_session_load_payload()` resets DSpark counters, but payloads do not
  appear to reconstruct DSpark capture tensors; avoid claims about
  disk-KV/session-restore DSpark correctness without a targeted restore parity
  smoke.
- The parity test covers uncapped batched scheduled behavior against serial
  scheduling. Cap values `3` and `4` are smoke-covered but not as deeply
  regression-tested as the default/cap-5 path.
- Short six-prompt results should be treated as exploratory. Run order, thermal
  state, power setting, and cycle-count differences can dominate small
  throughput deltas.

## Safe Claim Boundaries

Safe to claim after the sweep, if JSON gating passes:

- The current `ds4-spec-bench` path reuses one loaded engine while creating a
  fresh session per run.
- DSpark timing is not forced by `ds4-spec-bench`; throughput pass results are
  from explicit non-diagnostic timing-off runs.
- For the retained short sample, batched scheduled DSpark with caps `5/4/3`
  produced exact speculative argmax output under target verification.
- Throughput comparisons are valid for the current harness, model, backend,
  prompts, env, and cap settings.

Unsafe to claim from this sweep alone:

- General production speedup.
- Restore/disk-cache/session-snapshot DSpark correctness.
- Quality or sampling distribution effects.
- Per-component DSpark timing attribution unless a separate
  `DS4_DSPARK_TIMING=1` diagnostic pass is run.

## Blocker

No high-priority blocker found for the planned short cap sweep. Before running,
explicitly unset `DS4_DSPARK_TIMING` and `DS4_DSPARK_VERIFY_K`, keep
`DS4_DSPARK_CONF_THRESHOLD` fixed, and reject any batched-arm JSONL row where
`schedule_batched` is false or `schedule_batch_limit` is not the intended cap.
