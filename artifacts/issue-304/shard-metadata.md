# Issue 304 Shard Metadata

## Phase 0 planned distributed topology

### Historical local smaller reproducible topology

Planned local topology for `tests/issue304_phase0_local`:

| Role | Layers | Host |
| --- | --- | --- |
| coordinator | `0:21` | `127.0.0.1` |
| worker | `22:output` | `127.0.0.1` |

Current local result:

- route metadata not collected
- blocker: second `ds4` process on same host is refused by the current single-instance lock before route formation

### DGX issue topology

| Role | Layers | Host |
| --- | --- | --- |
| coordinator | `0:21` | `10.77.0.1` |
| worker | `22:output` | `10.77.0.2` |

Live route metadata from the authoritative Phase 0 run:

| Field | Value |
| --- | --- |
| route hops | 1 |
| route summary | `local 0:21 -> 10.77.0.2:34949 Q2 22:output` |
| output owner | worker |
| prompt tokens | 14,318 |
| immediate staged payload bytes | 221,006,660 |
| immediate stage result | success |
| token count at staged checkpoint | 14,318 |
| token hash at staged checkpoint | `0xa34307b6aaa8f956` |
| note | This row captures the merged-save checkpoint immediately after prefill, before any meaningful decode-window measurement rerun |

Reproduced failure mode from the earlier invalid run:

| Field | Value |
| --- | --- |
| failing route summary | `local 0:21 -> 10.77.0.2:42211 Q2 22:output` |
| coordinator `ctx` | 16,384 |
| worker `ctx` | 32,768 |
| stage result | fail |
| stage error | `distributed KV shards use different layouts: field=ctx local=16384 remote=32768 remote_layers=22:42` |

## Full `DSV4` payload probe

Observed from the new local Phase 1 probe (`ctx=16384`, saved token count `1`):

| Field | Value |
| --- | ---: |
| magic | `DSV4` |
| version | `2` |
| `ctx_size` | 16384 |
| `prefill_cap` | 4096 |
| `raw_cap` | 4352 |
| `raw_window` | 128 |
| `comp_cap` | 4098 |
| `saved_tokens` | 1 |
| `n_layer` | 43 |
| `head_dim` | 512 |
| `indexer_head_dim` | 128 |
| `vocab` | 129280 |
| `raw_live` | 1 |

## Resume-case `DSV4` metadata

The actual Phase 1 resume checks run with `ctx=192`. For frontiers `127`, `128`, and `129`, the saved local payloads reported:

| Frontier | Payload bytes | `raw_window` | `raw_cap` | `prefill_cap` |
| --- | ---: | ---: | ---: | ---: |
| 127 | 25,574,792 | 128 | 192 | 192 |
| 128 | 25,757,580 | 128 | 192 | 192 |
| 129 | 25,757,584 | 128 | 192 | 192 |

## Representative `DSVL` shard metadata

Single-layer shard round trips captured from `./ds4_test --local-payload-resume`:

| Layer | Payload bytes | `raw_window` | `comp_cap` | `n_comp` | `n_index_comp` | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 262,208 | 128 | 50 | 0 | 0 | first layer |
| 21 | 788,544 | 128 | 50 | 1 | 0 | middle compressed layer |
| 2 | 426,048 | 128 | 50 | 32 | 32 | representative ratio-4/indexer layer |
| 42 | 426,048 | 128 | 50 | 32 | 32 | final/output-adjacent layer |

## Notes

- These shard values come from the stable same-backend Metal run only.
- The single-layer smoke validates `save -> load -> save` byte identity for the sampled layers.
- The DGX/Mac mixed Metal/CUDA Phase 0 run did produce a merged `DSV4` header once the worker `ctx` matched the coordinator `ctx`.

## Phase 2 harness metadata contract

The new `tests/issue304_phase2_handoff` helper is intended to emit the following route and handoff metadata once the DGX/Mac run is executed:

| Field | Intended source |
| --- | --- |
| route hops | `ds4_session_distributed_route_summary()` |
| route summary | `ds4_session_distributed_route_summary()` |
| output owner | `ds4_session_distributed_route_summary()` |
| prompt tokens | encoded chat prompt length on the coordinator |
| payload bytes | merged staged `DSV4` file size |
| token count at handoff checkpoint | `ds4_session_tokens()` on the distributed session |
| token hash at handoff checkpoint | helper-side token hash over the distributed timeline |
| `ctx_size`, `prefill_cap`, `raw_cap`, `raw_window`, `comp_cap`, `saved_tokens`, `n_layer`, `head_dim`, `indexer_head_dim`, `vocab`, `raw_live` | parsed staged `DSV4` header |

Current status:

- helper implementation exists locally
- helper build passed locally and on the DGX host after syncing the required core files

### Phase 2 authoritative DGX/Mac row

| Field | Value |
| --- | --- |
| route hops | 1 |
| route summary | `local 0:21 -> 10.77.0.2:43045 Q2 22:output` |
| output owner | worker |
| prompt tokens | 14,318 |
| payload bytes | 221,006,660 |
| token count at handoff checkpoint | 14,334 |
| token hash at handoff checkpoint | `0xe9f870851b231ebb` |
| `ctx_size` | 16,384 |
| `prefill_cap` | 4,096 |
| `raw_cap` | 4,352 |
| `raw_window` | 128 |
| `comp_cap` | 4,098 |
| `saved_tokens` | 14,318 |
| `n_layer` | 43 |
| `head_dim` | 512 |
| `indexer_head_dim` | 128 |
| `vocab` | 129,280 |
| `raw_live` | 128 |
| note | `token_count` and `token_hash` were captured after the 16-token distributed reference continuation; the serialized payload itself still reflects the immediate post-prefill checkpoint with `saved_tokens=14,318`. |

## Phase 3 authoritative route/layout rows

Phase 3 added one more hard layout constraint beyond matching `ctx`: merged `DSV4` staging also requires worker/coordinator `prefill_cap` equality.

### Official vectors at `ctx=4096`

| Field | Value |
| --- | --- |
| route summary | `local 0:21 -> 10.77.0.2:38705 Q2 22:output` |
| route hops | 1 |
| output owner | worker |
| worker `ctx` | 4,096 |
| coordinator `ctx` | 4,096 |
| worker `prefill_cap` | 2,048 |
| coordinator `prefill_cap` | 2,048 |
| payload bytes, `short_code_completion` | 15,423,992 |
| payload bytes, `short_reasoning_plain` | 14,523,860 |
| note | The first attempted Phase 3 official run failed until worker `DS4_METAL_PREFILL_CHUNK` was aligned to `2048` with the strict local vector environment. |

### Official vectors at `ctx=16384`

| Field | Value |
| --- | --- |
| route summary | `local 0:21 -> 10.77.0.2:32907 Q2 22:output` |
| route hops | 1 |
| output owner | worker |
| worker `ctx` | 16,384 |
| coordinator `ctx` | 16,384 |
| worker `prefill_cap` | 2,048 |
| coordinator `prefill_cap` | 2,048 |
| payload bytes, `short_italian_fact` | 14,841,824 |
| payload bytes, `long_code_audit` | 76,903,324 |
| note | The official-vector environment still pinned `DS4_METAL_PREFILL_CHUNK=2048` even at `ctx=16384`, so the worker had to match that lower `prefill_cap` for merged staging to succeed. |

### Local-golden frontier at `ctx=5000`

| Field | Value |
| --- | --- |
| route summary | `local 0:21 -> 10.77.0.2:36505 Q2 22:output` |
| route hops | 1 |
| output owner | worker |
| worker `ctx` | 5,000 |
| coordinator `ctx` | 5,000 |
| worker `prefill_cap` | 4,096 |
| coordinator `prefill_cap` | 4,096 |
| payload bytes, `long_story_4096` | 80,373,132 |
| note | The local-golden environment reused `DS4_METAL_PREFILL_CHUNK=4096`, so this group needed a different worker launch from the official-vector groups. |

## Phase 3.5 route matrix metadata

Phase 3.5 reused the same model and layer split, but broadened the execution topology to six routes and ran the DGX worker and DGX-side helper at `--power 50`.

### Route mapping

| Route | Meaning | Payload bytes |
| --- | --- | ---: |
| `1` | pure local CUDA prefill+generation | `0` |
| `2` | pure local Metal prefill+generation | `0` |
| `3` | distributed generation `CUDA -> Metal` | `0` |
| `4` | distributed generation `Metal -> CUDA` | `0` |
| `5` | distributed prefill `CUDA -> Metal`, then resumed pure Metal generation | case-dependent merged `DSV4` |
| `6` | distributed prefill `Metal -> CUDA`, then resumed pure CUDA generation | case-dependent merged `DSV4` |

### Phase 3.5 payload rows

| Case | Payload bytes |
| --- | ---: |
| `short_code_completion` | 15,423,992 |
| `short_reasoning_plain` | 14,523,860 |
| `short_italian_fact` | 14,841,824 |
| `long_code_audit` | 76,903,324 |
| `long_story_4096` | 80,373,132 |

Notes:

- Routes `5` and `6` share payload byte counts per case because each pair resumes the same staged `DSV4` checkpoint on opposite hosts.
- Phase 3.5 did not expose new shard-layout incompatibilities beyond the already-known `ctx` and `prefill_cap` alignment constraints.
