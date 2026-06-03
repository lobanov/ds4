# Issue 304 Decision Log

## 2026-06-03: Correct the earlier Phase 0 save-path conclusion

Decision:

- Do not treat the earlier `distributed KV shards use different layouts` result as evidence that mixed Metal/CUDA immediate merged-`DSV4` staging is broken.
- Treat that earlier result as an invalid baseline caused by mismatched worker/coordinator session context.

Evidence:

- The earlier failing run used:
  - coordinator `ctx=16384`
  - worker default `ctx=32768`
  - and now reports the exact mismatch:
    - `distributed KV shards use different layouts: field=ctx local=16384 remote=32768 remote_layers=22:42`
- The route code accepts this for inference because worker admission only requires `worker_ctx >= coordinator_ctx`.
- Re-running the same Metal/CUDA topology with worker `--ctx 16384` produced:
  - successful route formation
  - successful prefill on `README.md`
  - successful immediate `ds4_session_stage_payload()`
  - payload bytes: `221,006,660`
  - parsed merged `DSV4` header with `ctx_size=16384`

Why this changes the notes:

- The original decision overgeneralized from an invalid reproduction.
- The precise constraint is narrower:
  - distributed generation tolerates worker `ctx >= coordinator ctx`
  - merged shard save requires exact layout equality, so `ctx` must match across participants
- The Phase 2 `DSV4`-first path remains viable if the runbook pins matching `--ctx` on all distributed participants.
