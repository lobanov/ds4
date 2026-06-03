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
