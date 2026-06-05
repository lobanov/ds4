# Issue 304 Performance Breakdown

## Phase 4 final-worker handoff timings

Tool:

```sh
make tests/issue304_phase4_handoff
```

Authoritative 2026-06-05 passing runs:

| Route | Prefill handoff sec | Prefill ref sec | Shard load sec | Local decode sec | Local tok/s | Distributed decode sec | Distributed tok/s | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Metal -> CUDA` | 37.998446 | 38.547497 | 0.431831 | 1.226027 | 13.05 | 0.977782 | 16.36 | Pass |
| `CUDA -> Metal` | 36.136962 | 35.978407 | 0.263383 | 0.522352 | 30.63 | 0.950070 | 16.84 | Pass |

Additional 2026-06-05 instability notes:

| Route | Result | Notes |
| --- | --- | --- |
| `CUDA -> Metal` reused Metal worker, run A | Fail | First mismatch at generated step `11`. |
| `CUDA -> Metal` reused Metal worker, run B | Fail | Diverged from the first generated token; `handoff_first_token=5`, reference `2337`. |
| `Metal -> CUDA` reused CUDA worker, repeat set of 3 | Pass | All three runs matched with identical generated tokens. |

Interpretation:

- The in-process final-worker handoff is viable in both backend directions.
- The local worker decode can be materially faster than continued distributed decode on the same prompt frontier.
- The remaining performance caveat is operational: strict `CUDA -> Metal` parity checks should restart the Metal worker first.

`CUDA -> Metal` divergence characterization from `tests/issue304_phase4_diagnose`:

| Reused-worker run | First mismatch | Sequence character | Top-5 overlap | Top-10 overlap | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| A | `0` | Coherent alternate continuation | `5/5` | `10/10` | Handoff chose `#`; reference chose `This`. |
| B | `11` | Coherent punctuation / formatting variant | `5/5` | `10/10` | Handoff chose ` (\``; reference chose ` (`. |

Diagnostic interpretation:

- The observed reused-Metal-worker divergence is not gibberish.
- The first divergent token remains inside the opposite side's top-5/top-10 candidate set.
- This matches the practical shape already seen in Phase 3.5: near-top1 ranking flips on a backend-sensitive frontier.

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

## Phase 3 vector parity timings

Phase 3 used `tests/issue304_phase3_vectors` with five one-shot coordinator runs per group. The one-shot structure avoided coordinator port reuse races on `10.77.0.1:1234` and gives the relevant worst@5 timing envelope directly.

### Official vectors at `ctx=4096`

Worker environment:

- `--ctx 4096`
- `DS4_METAL_PREFILL_CHUNK=2048`

Worst@5 observed timing by case:

| Case | Payload bytes | Prefill sec | Stage sec | Load sec |
| --- | ---: | ---: | ---: | ---: |
| `short_code_completion` | 15,423,992 | 0.450 | 0.090 | 0.0020 |
| `short_reasoning_plain` | 14,523,860 | 0.394 | 0.076 | 0.0021 |

Notes:

- The `ctx=4096` official cases are cheap enough that phase timing was effectively stable across all five runs.
- These timings are not the blocker; the blocker is same-backend resumed-decode parity.

### Official vectors at `ctx=16384`

Worker environment:

- `--ctx 16384`
- `DS4_METAL_PREFILL_CHUNK=2048`

Worst@5 observed timing by case:

| Case | Payload bytes | Prefill sec | Stage sec | Load sec |
| --- | ---: | ---: | ---: | ---: |
| `short_italian_fact` | 14,841,824 | 0.656 | 0.077 | 0.0022 |
| `long_code_audit` | 76,903,324 | 9.977 | 0.368 | 0.0189 |

Notes:

- `long_code_audit` makes the intended cost split clear:
  - distributed prefill still dominates at about 10 s
  - merged payload stage stayed below 0.4 s
  - local payload load stayed below 0.02 s
- The throughput argument for handoff remains intact; the unresolved problem is decode-state equivalence after load.

### Local-golden frontier at `ctx=5000`

Worker environment:

- `--ctx 5000`
- `DS4_METAL_PREFILL_CHUNK=4096`

Worst@5 observed timing:

| Case | Payload bytes | Prefill sec | Stage sec | Load sec |
| --- | ---: | ---: | ---: | ---: |
| `long_story_4096` | 80,373,132 | 12.262 | 0.367 | 0.0102 |

Notes:

- The 4096-token frontier remains practical from a staging perspective:
  - prefill about 12.1 s
  - merged stage at most about 0.37 s
  - local load about 0.01 s
- That makes the same-backend parity failure more important: there is no evidence here that handoff cost, rather than resumed state evolution, is the main issue.

## Phase 3.5 six-route timings

Phase 3.5 used `tests/issue304_phase35_matrix.py` to run five one-shot repeats per route/case. The DGX worker and DGX-side helper were run with `--power 50`.

### Official vectors

| Case | Route `1` prefill sec | Route `2` prefill sec | Route `3` prefill sec | Route `4` prefill sec | Route `5` load sec | Route `6` load sec | Payload bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_code_completion` | 0.753 | 0.294 | 1.043 | 0.874 | 0.0018 | 0.0219 | 15,423,992 |
| `short_reasoning_plain` | 0.621 | 0.257 | 0.982 | 0.823 | 0.0018 | 0.0680 | 14,523,860 |
| `short_italian_fact` | 0.634 | 0.263 | 1.065 | 0.832 | 0.0024 | 0.0198 | 14,841,824 |
| `long_code_audit` | 19.446 | 10.416 | 12.196 | 9.271 | 0.0111 | 0.0932 | 76,903,324 |

Notes:

- The route `1` vs route `2` gap is the expected CUDA-vs-Metal local prefill cost split.
- Direct distributed generation remained close to the slower of the two local-prefill paths.
- Payload resume cost stayed small relative to prefill cost, even on the longer frontier:
  - route `5` loads stayed around `0.0018-0.0111 s`
  - route `6` loads stayed around `0.0198-0.0932 s`

### Local-golden frontier

| Case | Route `1` prefill sec | Route `2` prefill sec | Route `3` prefill sec | Route `4` prefill sec | Route `5` load sec | Route `6` load sec | Payload bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_story_4096` | 19.855 | 8.933 | 15.193 | 11.276 | 0.0112 | 0.0883 | 80,373,132 |

Notes:

- The same pattern holds on the 4096-token frontier: payload load remains cheap, and the dominant runtime is still prefill.
- That keeps the Phase 3.5 focus on correctness classification rather than route cost.
