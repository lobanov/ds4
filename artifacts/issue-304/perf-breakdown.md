# Issue 304 Performance Breakdown

## Phase 0 baseline

### Historical local loopback topology

Local loopback baseline tooling is implemented, but the actual run is currently blocked on the same host.

Attempted command:

```sh
./tests/issue304_phase0_local
```

Observed result:

| Item | Result |
| --- | --- |
| Tool build | Pass |
| Worker launch | Fail on same-host second-process startup |
| Route formation | Not reached |
| Distributed prefill timing | Not collected |
| Distributed payload stage timing | Not collected |
| Distributed decode timing | Not collected |
| Blocker | single-instance lock in `ds4_engine_open()` refuses a second `ds4` process on the same host |

Planned fields once a second host is used:

- prompt file
- prompt token count
- distributed prefill chunk count
- distributed prefill seconds / t/s
- distributed payload stage seconds / bytes
- distributed decode seconds / t/s
- activation transport bits
- coordinator layer range
- worker layer range

### Issue topology on DGX + Mac

Authoritative command pair:

```sh
ssh dgx-direct 'cd ~/ds4 && ./ds4 -m /home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --coordinator 10.77.0.1 1234'
./tests/issue304_phase0_dgx --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --listen-host 10.77.0.1 --listen-port 1234 --prompt-file README.md --ctx 16384 --gen-tokens 1 --prefill-chunk 256 --activation-bits 32 --coordinator-layers 0:21 --worker-layers 22:output
```

Observed result:

| Item | Result |
| --- | --- |
| Link / addresses | Mac coordinator `10.77.0.1:1234`, DGX worker `10.77.0.2:34949` |
| Prompt file | `README.md` |
| Prompt tokens | 14,318 |
| Context | 16,384 |
| Activation transport bits | 32 |
| Distributed prefill | 38.061 s, 376.19 tok/s |
| Distributed payload stage | 0.569 s |
| Staged payload bytes | 221,006,660 |
| Payload stage status | Success |
| Distributed decode | Not re-measured authoritatively after the `ctx` fix; only a 1-token probe was used to keep the save-path validation focused |
| Route stability | Route formed cleanly and merged payload staging completed |

Notes:

- `ctx=8192` was insufficient for the README prompt in chat format; `ctx=16384` was required.
- The authoritative conclusion from this pass is about merged save viability and payload metadata, not decode throughput.
- The earlier stage failure was not a generic Metal/CUDA incompatibility. It was a context-layout mismatch:
  - coordinator helper used `ctx=16384`
  - worker was left at CLI default `ctx=32768`
  - route formation still succeeded because the worker only needs `ctx >= coordinator ctx`
  - merged `DSV4` stage failed because shard headers require exact layout equality
- Reproduced earlier failure signature:
  - `distributed KV shards use different layouts: field=ctx local=16384 remote=32768 remote_layers=22:42`

## Phase 2 handoff harness

### Authoritative DGX + Mac result

Phase 2 timing collection now has a dedicated tool:

```sh
make tests/issue304_phase2_handoff
```

Authoritative command pair:

```sh
ssh dgx-direct 'cd ~/ds4 && ./ds4 -m /home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --coordinator 10.77.0.1 1234'
./tests/issue304_phase2_handoff --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --listen-host 10.77.0.1 --listen-port 1234 --prompt-file README.md --ctx 16384 --gen-tokens 16 --forced-steps 8 --prefill-chunk 256 --activation-bits 32 --coordinator-layers 0:21 --worker-layers 22:output --payload-out /private/tmp/issue304-phase2-readme.dsv4
```

Observed result:

| Item | Result |
| --- | --- |
| Tool build | Pass |
| Prompt file | `README.md` |
| Prompt tokens | 14,318 |
| Route summary | `local 0:21 -> 10.77.0.2:43045 Q2 22:output` |
| Distributed prefill | `38.183 s`, `374.99 tok/s` |
| Merged payload stage | `0.380 s` |
| Merged payload bytes | `221,006,660` |
| Local payload load | `0.028 s` |
| Distributed decode | `0.983 s` for 16 tokens, `16.28 tok/s` |
| Local decode after handoff | `0.528 s` for 16 tokens, `30.32 tok/s` |
| Handoff logits | exact match |
| Greedy continuation | exact match for 16 tokens |
| Forced-trace result | first bad step `1`, `rms=0.0802887231`, `max_abs=0.507860184` |

Planned fields from `tests/issue304_phase2_handoff`:

- prompt file
- prompt token count
- route summary and output owner
- distributed prefill seconds / t/s
- merged payload stage seconds / bytes
- local payload load seconds
- distributed greedy decode seconds / t/s
- local greedy decode seconds / t/s
- handoff logit comparison metrics
- forced-trace first bad step and drift metrics
