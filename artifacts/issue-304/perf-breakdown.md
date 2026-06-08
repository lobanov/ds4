# Issue 304 Performance Breakdown

## Phase 6 profiling snapshot

These measurements focus on the worker-owned local-decode workflow after the
Phase 6 re-scope. The authoritative profiling pass was run on June 8, 2026
over the direct-link route:

- local Mac worker or coordinator: `10.77.0.1`
- DGX coordinator or worker: `10.77.0.2`
- helper: `tests/issue304_phase5_multiturn --allow-turn2-mismatch`
- model context: `ctx=16384`
- generation window: `8` tokens
- distributed parameters: `--prefill-chunk 256 --activation-bits 32`

This uses the same direct-link topology family as the earlier Phase 5 CLI
timings, so the handoff and prefill numbers are directly comparable.

### 2026-06-08 direct-link profiling matrix

Prompt buckets:

- short: helper default one-sentence prompt
- medium: `README.md` first 4 KiB slice
- long: `tests/test-vectors/prompts/long_code_audit.txt`
- very long: full `README.md`

One-shot turn-one timing, `CUDA -> Metal`:

| Prompt bucket | Prompt tokens | Sync sec | Sync tok/s | KV bytes | KV handoff sec | KV MiB/s | Decode sec | Decode tok/s | End-to-end sec | Dominant bottleneck | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| short | 18 | 1.189 | 15.13 | 6,975,720 | 0.022 | 305.94 | 0.209 | 38.23 | 1.420 | sync / prefill | handoff was `1.5%` of total |
| medium | 959 | 3.513 | 273.00 | 18,091,240 | 0.097 | 178.25 | 0.211 | 37.98 | 3.820 | sync / prefill | handoff was `2.5%` of total |
| long | 3,850 | 9.469 | 406.62 | 37,071,080 | 0.118 | 299.03 | 0.217 | 36.91 | 9.804 | sync / prefill | handoff was `1.2%` of total |
| very long | 14,435 | 36.979 | 390.35 | 106,488,040 | 0.239 | 425.47 | 0.261 | 30.64 | 37.479 | sync / prefill | handoff was `0.6%` of total |

One-shot turn-one timing, `Metal -> CUDA`:

| Prompt bucket | Prompt tokens | Sync sec | Sync tok/s | KV bytes | KV handoff sec | KV MiB/s | Decode sec | Decode tok/s | End-to-end sec | Dominant bottleneck | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| short | 18 | 0.851 | 21.16 | 6,975,720 | 0.022 | 302.09 | 0.526 | 15.21 | 1.399 | sync / prefill | handoff was `1.6%` of total; allowed turn-two variance still appears on this tiny prompt |
| medium | 959 | 2.955 | 324.53 | 18,091,240 | 0.049 | 351.74 | 0.561 | 14.25 | 3.566 | sync / prefill | handoff was `1.4%` of total |
| long | 3,850 | 9.099 | 423.14 | 37,071,080 | 0.230 | 153.72 | 0.638 | 12.55 | 9.966 | sync / prefill | handoff was `2.3%` of total |
| very long | 14,435 | 38.579 | 374.16 | 106,488,040 | 0.338 | 300.77 | 0.601 | 13.32 | 39.517 | sync / prefill | handoff was `0.9%` of total |

Reused-session follow-up timing on the same runs:

| Route | Prompt bucket | Follow-up frontier tokens | Follow-up sync sec | KV handoff sec | Decode sec | End-to-end sec | Dominant bottleneck | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `CUDA -> Metal` | short | 42 | 0.244 | 0.028 | 0.209 | 0.480 | sync / prefill | fresh-vs-reused turn-two tokens diverged, but this is allowed Phase 5.5 variance |
| `CUDA -> Metal` | medium | 983 | 0.249 | 0.064 | 0.210 | 0.524 | sync / prefill | reused and fresh turn-two tokens matched |
| `CUDA -> Metal` | long | 3,874 | 0.255 | 0.120 | 0.216 | 0.591 | sync / prefill | reused and fresh turn-two tokens matched |
| `CUDA -> Metal` | very long | 14,459 | 0.262 | 0.228 | 0.261 | 0.751 | sync / prefill | reused and fresh turn-two tokens matched |
| `Metal -> CUDA` | short | 42 | 0.227 | 0.028 | 0.527 | 0.782 | decode | fresh-vs-reused turn-two tokens diverged, but this is allowed Phase 5.5 variance |
| `Metal -> CUDA` | medium | 983 | 0.238 | 0.049 | 0.560 | 0.847 | decode | reused and fresh turn-two tokens matched |
| `Metal -> CUDA` | long | 3,874 | 0.236 | 0.237 | 0.637 | 1.110 | decode | reused and fresh turn-two tokens matched |
| `Metal -> CUDA` | very long | 14,459 | 0.240 | 0.366 | 0.606 | 1.212 | decode | reused and fresh turn-two tokens matched |

Fresh-session follow-up timing from the same helper:

| Route | Prompt bucket | Fresh turn-one sync sec | Fresh follow-up sync sec | Fresh turn-two handoff sec | Fresh turn-two decode sec | Fresh turn-two total sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CUDA -> Metal` | short | 0.976 | 0.260 | 0.032 | 0.207 | 0.499 |
| `CUDA -> Metal` | medium | 3.414 | 0.252 | 0.079 | 0.212 | 0.543 |
| `CUDA -> Metal` | long | 9.365 | 0.263 | 0.112 | 0.217 | 0.592 |
| `CUDA -> Metal` | very long | 36.660 | 0.254 | 0.234 | 0.262 | 0.750 |
| `Metal -> CUDA` | short | 0.587 | 0.229 | 0.035 | 0.536 | 0.799 |
| `Metal -> CUDA` | medium | 2.984 | 0.231 | 0.057 | 0.558 | 0.846 |
| `Metal -> CUDA` | long | 9.131 | 0.235 | 0.092 | 0.633 | 0.959 |
| `Metal -> CUDA` | very long | 38.281 | 0.243 | 0.365 | 0.600 | 1.208 |

Interpretation:

- On the direct-link deployment target, one-shot latency is decisively
  prefill-bound in both route directions.
- KV handoff stays in the "small tail" category for one-shot work:
  - `CUDA -> Metal`: about `0.6%` to `2.5%` of end-to-end time
  - `Metal -> CUDA`: about `0.9%` to `2.3%` of end-to-end time
- Reused-session follow-up is still not transport-dominated enough to justify
  pipelining:
  - `CUDA -> Metal` remains mostly sync-bound
  - `Metal -> CUDA` is more decode-heavy because the CUDA worker generates
    the 8-token window at about `12.6` to `15.2 tok/s`, versus about `30.6`
    to `38.2 tok/s` on the Metal worker
  - even where follow-up handoff reaches about `20%` to `30%` of total on the
    longest frontiers, it is still not the dominant stage
- The practical Phase 6 conclusion on the direct-link deployment target is
  therefore unchanged from the original intuition:
  - distributed prefill dominates one-shot runtime,
  - route-direction differences are more about decode backend cost than KV
    return cost,
  - and KV pipelining is not justified by the current direct-link numbers.

## Phase 5 CLI worker-owned local-decode timings

These measurements use the plain `ds4` CLI and the in-process final-worker
handoff path with stream-based KV transfer. The coordinator was run with
`--debug` so the new KV handoff timing line was emitted directly from the
user-visible path.

### 2026-06-05 CLI measurements

| Route | Prompt | Prompt tokens | KV bytes | KV handoff sec | KV MiB/s | Prefill tok/s | Worker/local generation tok/s | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Metal -> CUDA` | `ping` | 10 | 6,564,072 | 0.025 | 248.27 | 14.27 | 14.88 | tiny prompt sanity check |
| `Metal -> CUDA` | sampled short prompt | 17 | 6,930,664 | 0.028 | 238.89 | 32.76 | 12.06 | one-shot sampled coordinator `--temp 0.7 --top-p 0.95 --min-p 0.05`; output: `A distributed system is a collection of independent` |
| `Metal -> CUDA` | README 4 KiB slice | 958 | 18,091,240 | 0.055 | 313.91 | 314.80 | 14.13 | mid-sized prompt |
| `Metal -> CUDA` | full `README.md` | 14,318 | 105,725,160 | 0.345 | 292.47 | 603.15 | 13.08 | reused CUDA worker |
| `CUDA -> Metal` | `ping` | 10 | 6,564,072 | 0.019 | 335.60 | 10.87 | 38.69 | fresh Metal worker |
| `CUDA -> Metal` | full `README.md` | 14,524 | 107,097,320 | 0.350 | 291.42 | 589.93 | 30.33 | fresh Metal worker |
| local Mac baseline | full `README.md` | 14,318 | n/a | n/a | n/a | 413.44 | 34.78 | no distributed route |

Interpretation:

- For long prompts, distributed prefill still dominates runtime. The new KV
  handoff stayed around `0.35 s` for roughly `100 MiB` payloads in both
  directions.
- That keeps KV handoff in the "small tail" category relative to
  `23-25 s` distributed prefill on the README prompt.
- On the Mac decode side, worker-owned local generation reached
  `30.33 tok/s` against a pure local Mac baseline of `34.78 tok/s`, which
  is close enough that the main performance question is no longer decode
  throughput.
- Sampled one-shot handoff uses the same KV transfer and worker-owned
  decode path as greedy one-shot runs; no separate protocol was needed to
  get sampling through the Phase 5 CLI surface.
- Normal token-by-token eval now uses the same worker-owned local-decode
  transport after a one-time activation push. CLI `--dump-logprobs` and
  `ds4_server` both exercised that surface successfully in both route
  directions on 2026-06-05.
- For very short generations the one-time handoff can still matter to
  first-token latency, but the current data does not justify KV pipelining
  for throughput-oriented Phase 5 benchmarking.

### 2026-06-06 `ds4-eval` local-decode handoff smoke

After the evaluator was patched to use distributed handoff for eligible
`--nothink` local-decode coordinator cases, a fresh `Q1`
`CUDA -> Metal --local-decode` smoke produced:

| Surface | Route | Prompt | Prompt tokens | Generated tokens | Prefill sec | Decode sec | Decode tok/s | Result | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ds4-eval` | `CUDA -> Metal`, worker `--local-decode` | `GPQA Diamond/recNu3MXkvWUzHZr9` | 201 | 539 | 3.023 | 54.426 | 9.903 | Pass | evaluator now uses distributed handoff for eligible `--nothink` local-decode |

Interpretation:

- This is no longer the old per-token evaluator loop mistake. The evaluator
  route now activates the worker-owned local-decode handoff path for the
  tested `--nothink` case.
- Even after that fix, evaluator throughput is still well below the plain
  CLI `CUDA -> Metal` local-decode benchmark (`30.10` to `30.33 tok/s`).
- Code inspection explains the remaining gap: the current worker
  `LOCAL_GENERATE` RPC allocates, fills, and returns a full logits trace for
  every generated token, and the reported `decode_usec` includes that extra
  work.
- So the current evaluator-vs-CLI gap should be treated as protocol
  overhead in the local-generate response path, not as proof that Metal
  local decode itself only sustains `~10 tok/s`.

### 2026-06-06 explicit `ds4-eval` timeout diagnosis

Explicit `CUDA -> Metal` evaluator repro:

| Surface | Route | Questions | Timeout mode | Result | Key timing / error | Notes |
| --- | --- | ---: | --- | --- | --- | --- |
| `ds4-eval` | `CUDA -> Metal`, worker `--local-decode` | 6 | old default `60s` | Fail | case 1 failed after about `60s` with `failed to read frame header: Resource temporarily unavailable` | one-shot local decode had not returned its first response frame before the socket timeout |
| `ds4-eval` | `CUDA -> Metal`, worker `--local-decode` | 6 | `DS4_DIST_SOCKET_TIMEOUT_SEC=600` | Pass | case runtimes `111.9s` to `113.8s`, full run `6/6 passed` in `00h:11m` | proves the blocker was socket policy, not evaluator control flow |
| `ds4-eval` | `CUDA -> Metal`, worker `--local-decode` | 1 | new default `600s`, no env override | Pass | case 1 `113.8s`, `1/1 passed` | verifies the code-side timeout change removes the need for a manual env override |

### 2026-06-06 authoritative `ds4-eval` full runs

The authoritative Phase 5 evaluator comparison is now the full `92`-question
set, not the earlier `4`-question smoke.

| Surface | Route | Score | Runtime | Notes |
| --- | --- | ---: | --- | --- |
| `ds4-eval` | local Metal | `67/92` | `00h:47m` | Mac baseline |
| `ds4-eval` | local CUDA | `69/92` | `01h:39m` | DGX/local CUDA baseline |
| `ds4-eval` | `CUDA -> Metal`, worker `--local-decode` | `65/92` | `03h:01m` | worker-owned local decode stayed active and completed the full suite |

Interpretation:

- The Phase 5 local-decode route is now stable enough to finish the full
  evaluator suite instead of failing on timeout or disconnect.
- Score variance against the fully local baselines is small enough that Phase 5
  closeout should treat it as evaluator variance, not as a routing regression.
- The remaining performance gap versus the plain CLI benchmark is still a
  protocol-overhead question in the current `LOCAL_GENERATE` response path.

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
