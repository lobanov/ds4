# Issue 304 Runbook

## Phase 5 worker-owned local-decode workflow

Phase 5 now has a CLI-visible path. Use these rules for authoritative
validation:

- use the plain `ds4` CLI, not the older issue harnesses, when validating
  the user-visible workflow,
- `--local-decode` belongs on the worker that owns `N:output`,
- `--local-decode` implies full worker residency,
- for strict `CUDA -> Metal` checks, start a fresh Metal worker process,
- for `Metal -> CUDA`, a reused CUDA worker remains acceptable coverage,
- pass `--debug` on the coordinator to record the KV handoff timing line.

The coordinator now logs:

```text
ds4: distributed coordinator: local-decode KV handoff tokens=... layers=... bytes=... total=... MiB/s worker=...
```

This is the timing surface to use when deciding whether KV pipelining is
worth implementing.

### CLI commands

`Metal -> CUDA` worker startup on DGX:

```sh
ssh dgx-direct 'pkill -9 ds4 >/dev/null 2>&1 || true; sh -c "cd ~/ds4; nohup ./ds4 -m ~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --local-decode --coordinator 10.77.0.1 1234 >/tmp/ds4-phase5-worker.log 2>&1 < /dev/null &"'
```

`Metal -> CUDA` one-shot coordinator run on the Mac:

```sh
./ds4 -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --temp 0 --nothink --role coordinator --layers 0:21 --listen 10.77.0.1 1234 --prompt-file README.md -n 8 --debug
```

`Metal -> CUDA` sampled one-shot coordinator smoke on the Mac:

```sh
./ds4 -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --temp 0.7 --top-p 0.95 --min-p 0.05 --nothink --role coordinator --layers 0:21 --listen 10.77.0.1 1234 -p 'Describe a distributed system in one sentence.' -n 8 --debug
```

`CUDA -> Metal` fresh worker startup on the Mac:

```sh
./ds4 -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --local-decode --coordinator 10.77.0.2 1234
```

`CUDA -> Metal` one-shot coordinator run on DGX:

```sh
ssh dgx-direct 'cd ~/ds4 && ./ds4 -m ~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --temp 0 --nothink --role coordinator --layers 0:21 --listen 10.77.0.2 1234 --prompt-file README.md -n 8 --debug'
```

### Authoritative 2026-06-05 CLI timings

| Route | Prompt | Prompt tokens | KV handoff bytes | KV handoff sec | Prefill tok/s | Generation tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Metal -> CUDA` | `ping` | 10 | 6,564,072 | 0.025 | 14.27 | 14.88 |
| `Metal -> CUDA` | README 4 KiB slice | 958 | 18,091,240 | 0.055 | 314.80 | 14.13 |
| `Metal -> CUDA` | full `README.md` | 14,318 | 105,725,160 | 0.345 | 603.15 | 13.08 |
| `CUDA -> Metal` | `ping` | 10 | 6,564,072 | 0.019 | 10.87 | 38.69 |
| `CUDA -> Metal` | full `README.md` | 14,524 | 107,097,320 | 0.350 | 589.93 | 30.33 |

Reference local Mac baseline:

```sh
./ds4 -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --temp 0 --nothink --prompt-file README.md -n 8
```

- full `README.md`: prefill `413.44 tok/s`, generation `34.78 tok/s`

Current limitation:

- sampled one-shot CLI coordinator runs now work through the worker-owned
  handoff path as well as greedy ones.
- normal token-by-token distributed eval now also delegates through
  worker-owned local decode after the first post-sync activation on both
  CLI and `ds4_server`.
- reusable-session follow-up after catch-up is still under investigation:
  the immediate post-handoff frontier is now close to fresh-session parity,
  but a follow-up distributed sync on top of that reused state can still
  produce a near-top1 logit flip and divergent turn-two tokens.
- the remaining open issue is reusable-session variance, not lack of
  normal eval/server delegation.

## Phase 4 closeout rules

Phase 4 is complete. Use these rules for ongoing validation and follow-on work:

- run only one DGX `ds4` worker at a time,
- clear stale DGX workers before starting the opposite route direction,
- treat a fresh Metal worker process as the authoritative `CUDA -> Metal` validation path,
- do not rely on repeated `CUDA -> Metal` reruns against the same long-lived Metal worker process,
- do not treat one successful fresh-worker `CUDA -> Metal` pass as proof that immediate reruns will also match,
- and do treat repeated `Metal -> CUDA` runs on a reused CUDA worker as acceptable coverage, since three back-to-back sessions matched.

Current startup guard behavior:

- CUDA startup now rejects a large accelerator residency request before model mapping if free memory is too low.
- Override only for explicit debugging with:

```sh
DS4_DISABLE_STARTUP_MEMORY_GUARD=1
```

Observed DGX rejection example:

```text
ds4: cuda startup memory guard rejected model residency request
free 18.94 GiB, request 42.03 GiB, reserve 10.14 GiB, total 121.69 GiB
```

Safe Phase 4 route order:

1. start exactly one worker for the target route
2. run `tests/issue304_phase4_handoff` for pass/fail validation or `tests/issue304_phase4_diagnose` for mismatch characterization
3. stop that worker
4. verify no stale DGX `ds4` remains before starting the next route

Recommended cleanup checks:

```sh
ssh dgx-direct 'ps -ef | grep "[d]s4" || true'
ssh dgx-direct 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits'
```

## Phase 0 baseline

### Historical localhost preparation

Build the local loopback baseline tool:

```sh
make tests/issue304_phase0_local
```

Run it with the default smaller reproducible topology:

```sh
./tests/issue304_phase0_local
```

Default local topology baked into the tool:

- coordinator host: `127.0.0.1`
- coordinator layers: `0:21`
- worker layers: `22:output`
- prompt file: `README.md`
- context: `8192`
- distributed prefill chunk: `256`
- distributed activation bits: `32`
- generation sample window for baseline decode timing: `16` greedy tokens

What this tool attempts to verify:

- a distributed coordinator can form a route,
- a multi-chunk distributed prefill can complete,
- `ds4_session_stage_payload()` can run immediately after prefill,
- a staged merged `DSV4` payload exposes normal header metadata,
- current distributed decode can be timed on the same session.

Current local result:

- build: pass
- run: blocked locally
- blocker: the worker is a second `ds4` process on the same host, and `ds4_engine_open()` currently refuses a second process with the single-instance lock before route formation begins
- observed error: `ds4: another ds4 process is already running ... refusing to start`

Implication:

- same-host loopback is not currently a viable Phase 0 baseline path without changing instance-lock policy
- local Phase 0 work can still prepare the tool, artifact schema, and exact command set
- authoritative distributed measurements still need a second host, which in practice means moving on to the DGX/Mac topology

### DGX + Mac authoritative baseline

Build the coordinator helper locally:

```sh
make tests/issue304_phase0_dgx
```

Start the DGX worker over SSH:

```sh
ssh dgx-direct 'cd ~/ds4 && ./ds4 -m /home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --coordinator 10.77.0.1 1234'
```

Run the local coordinator helper on the Mac:

```sh
./tests/issue304_phase0_dgx --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --listen-host 10.77.0.1 --listen-port 1234 --prompt-file README.md --ctx 16384 --gen-tokens 100 --prefill-chunk 256 --activation-bits 32 --coordinator-layers 0:21 --worker-layers 22:output
```

Observed requirements and results:

- `ctx=8192` was too small for the `README.md` issue prompt after chat formatting; the authoritative run used `ctx=16384`.
- The worker `--ctx` must match the coordinator session `--ctx` for merged `DSV4` staging to succeed. Route formation alone only requires worker `ctx >= coordinator ctx`.
- Route formed successfully with worker `10.77.0.2:34949`.
- Distributed prefill completed on the README prompt.
- Immediate `ds4_session_stage_payload()` succeeded when both sides used `ctx=16384`.
- Earlier failure with `distributed KV shards use different layouts` was reproduced and narrowed to `field=ctx local=16384 remote=32768`, caused by leaving the worker at its CLI default `--ctx 32768`.

## Environment

- Date: 2026-06-03 15:54:57Z
- Local repo commit: `116e35881679c99cbe33454f95d2b4c96448761b`
- DGX worker repo commit: `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe` plus synced local changes to `ds4.c`, `ds4.h`, `ds4_distributed.c`, `ds4_distributed.h`, `Makefile`, and `tests/issue304_phase2_handoff.c`
- Coordinator host: Apple M5 Max / Metal
- Worker host: NVIDIA GB10 / CUDA
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Local model path: `/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- DGX model path: `/home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model hash:
  - local: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
  - DGX: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`

## Commands

### Phase 1 build

```sh
make ds4_test
```

### Phase 2 build

```sh
make tests/issue304_phase2_handoff
```

### Phase 1 focused path

```sh
./ds4_test --local-payload-resume
```

### Phase 1 cross-backend matrix helper

Local Metal save/reference:

```sh
./tests/issue304_phase1_matrix --mode save --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --payload /private/tmp/issue304-phase1/metal-frontier129.dsv4 --logits /private/tmp/issue304-phase1/metal-frontier129.logits.f32 --tokens /private/tmp/issue304-phase1/metal-frontier129.tokens.txt
```

Local Metal load-check from CUDA reference:

```sh
./tests/issue304_phase1_matrix --mode load-check --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --payload /private/tmp/issue304-phase1/cuda-frontier129.dsv4 --logits /private/tmp/issue304-phase1/cuda-frontier129.logits.f32 --tokens /private/tmp/issue304-phase1/cuda-frontier129.tokens.txt
```

DGX CUDA save/reference:

```sh
ssh dgx-direct 'cd ~/ds4 && ./tests/issue304_phase1_matrix --mode save --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --payload /tmp/issue304-phase1/cuda-frontier129.dsv4 --logits /tmp/issue304-phase1/cuda-frontier129.logits.f32 --tokens /tmp/issue304-phase1/cuda-frontier129.tokens.txt'
```

DGX CUDA load-check from Metal reference:

```sh
ssh dgx-direct 'cd ~/ds4 && ./tests/issue304_phase1_matrix --mode load-check --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --payload ./tests/metal-frontier129.dsv4 --logits ./tests/metal-frontier129.logits.f32 --tokens ./tests/metal-frontier129.tokens.txt'
```

### Phase 1 surrounding regression checks

```sh
./ds4_test --metal-short-prefill --metal-tensor-equivalence --local-golden-vectors
make test
```

### Official/local vector checks

Strict local official top-logprob check:

```sh
./ds4_test --logprob-vectors
```

Local golden-vector drift check:

```sh
./ds4_test --local-golden-vectors
```

Phase 3 should use these as the reference envelope for distributed-prefill-to-local decode. The hosted official API exposes top-logprobs, not full logits, so official-vector comparison is about selected-token and top-logprob agreement rather than bit-exact logit equality.

### Phase 2 DGX/Mac handoff validation

Start the DGX worker with the same `ctx` as the coordinator:

```sh
ssh dgx-direct 'cd ~/ds4 && ./ds4 -m /home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --coordinator 10.77.0.1 1234'
```

Run the local coordinator handoff helper on the Mac:

```sh
./tests/issue304_phase2_handoff --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --listen-host 10.77.0.1 --listen-port 1234 --prompt-file README.md --ctx 16384 --gen-tokens 16 --forced-steps 8 --prefill-chunk 256 --activation-bits 32 --coordinator-layers 0:21 --worker-layers 22:output
```

Optional: retain the merged payload for post-run inspection:

```sh
./tests/issue304_phase2_handoff --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --listen-host 10.77.0.1 --listen-port 1234 --prompt-file README.md --ctx 16384 --gen-tokens 16 --forced-steps 8 --prefill-chunk 256 --activation-bits 32 --coordinator-layers 0:21 --worker-layers 22:output --payload-out /private/tmp/issue304-phase2-readme.dsv4
```

What the helper records:

- route summary and output owner
- prompt token count and token hash
- merged payload bytes and parsed `DSV4` header
- distributed prefill timing
- merged payload stage timing
- local payload load timing
- distributed vs local greedy decode timing
- handoff-point logit comparison
- greedy continuation mismatch step, if any
- forced-token trace first bad step, if any

Observed authoritative Phase 2 result:

- Route summary: `local 0:21 -> 10.77.0.2:43045 Q2 22:output`
- Prompt tokens: `14,318`
- Prefill: `38.182716 s`, `374.99 tok/s`
- Stage: `0.380082 s`
- Payload bytes: `221,006,660`
- Local load: `0.028093 s`
- Distributed decode: `16.28 tok/s`
- Local decode: `30.32 tok/s`
- Handoff logits: exact match
- Greedy continuation: exact match for `16` tokens
- Forced trace:
  - first bad step: `1`
  - forced token before bad step: `5`
  - top1 remained `420`
  - `rms=0.0802887231`
  - `max_abs=0.507860184`

### Phase 3 planned validation

Phase 3 should run the official-vector prompt set through both:

- fully local inference on the decode backend
- distributed prefill -> merged `DSV4` -> local decode on the same backend

Acceptance:

- distributed-handoff output matches fully local output on the same decode backend within the same top-token/top-logprob behavior used by `--logprob-vectors`
- distributed-handoff output remains inside the local-golden drift envelope where full local logits are available
- any remaining `CUDA -> Metal` forced-logit drift is classified separately as backend variance unless it causes same-backend handoff parity or official/local-vector checks to fail

### Phase 3 actual build and sync

Build the new helper locally:

```sh
make tests/issue304_phase3_vectors
```

Ship the local source tree, excluding the GGUF and local build products:

```sh
rsync -az --delete \
  --exclude='.git/' \
  --exclude='gguf/' \
  --exclude='ds4flash.gguf' \
  --exclude='*.o' \
  --exclude='ds4' \
  --exclude='ds4-server' \
  --exclude='ds4-bench' \
  --exclude='ds4-eval' \
  --exclude='ds4-agent' \
  --exclude='ds4_test' \
  --exclude='tests/cuda_long_context_smoke' \
  --exclude='tests/test_q4k_dot' \
  --exclude='tests/issue304_phase0_local' \
  --exclude='tests/issue304_phase0_dgx' \
  --exclude='tests/issue304_phase1_matrix' \
  --exclude='tests/issue304_phase2_handoff' \
  --exclude='tests/issue304_phase3_vectors' \
  ./ dgx-direct:~/ds4/
```

Build the helper remotely before starting the worker:

```sh
ssh dgx-direct 'cd ~/ds4 && make tests/issue304_phase3_vectors'
```

### Phase 3 actual worker launches

Official vectors at `ctx=4096` and `ctx=16384` require worker `prefill_cap=2048` to match the strict local vector environment:

```sh
ssh dgx-direct 'pkill -9 ds4 >/dev/null 2>&1 || true; sh -c "cd ~/ds4; DS4_METAL_PREFILL_CHUNK=2048 nohup ./ds4 -m ~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 4096 --role worker --layers 22:output --coordinator 10.77.0.1 1234 >/tmp/ds4-worker-4096.log 2>&1 < /dev/null &"'
ssh dgx-direct 'pkill -9 ds4 >/dev/null 2>&1 || true; sh -c "cd ~/ds4; DS4_METAL_PREFILL_CHUNK=2048 nohup ./ds4 -m ~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 16384 --role worker --layers 22:output --coordinator 10.77.0.1 1234 >/tmp/ds4-worker-16384.log 2>&1 < /dev/null &"'
```

The local-golden frontier requires worker `prefill_cap=4096`:

```sh
ssh dgx-direct 'pkill -9 ds4 >/dev/null 2>&1 || true; sh -c "cd ~/ds4; DS4_METAL_PREFILL_CHUNK=4096 nohup ./ds4 -m ~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 5000 --role worker --layers 22:output --coordinator 10.77.0.1 1234 >/tmp/ds4-worker-5000.log 2>&1 < /dev/null &"'
```

### Phase 3 actual one-shot loops

The authoritative worst@5 data used five one-shot coordinator runs with short pauses between runs. This avoided coordinator port reuse races on `10.77.0.1:1234`.

Official `ctx=4096`:

```sh
for i in 1 2 3 4 5; do
  sleep 2
  DS4_METAL_DISABLE_METAL4=1 DS4_METAL_PREFILL_CHUNK=2048 \
    ./tests/issue304_phase3_vectors \
      --suite official \
      --repeat 1 \
      --ctx-filter 4096 \
      --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      --listen-host 10.77.0.1 \
      --listen-port 1234 \
      --vector-file tests/test-vectors/official.vec \
      --prefill-chunk 256 \
      --activation-bits 32 \
      --coordinator-layers 0:21 \
      --worker-layers 22:output \
      2>&1 | tee /tmp/issue304-phase3-official-4096-run${i}.log
done
```

Official `ctx=16384`:

```sh
for i in 1 2 3 4 5; do
  sleep 2
  DS4_METAL_DISABLE_METAL4=1 DS4_METAL_PREFILL_CHUNK=2048 \
    ./tests/issue304_phase3_vectors \
      --suite official \
      --repeat 1 \
      --ctx-filter 16384 \
      --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      --listen-host 10.77.0.1 \
      --listen-port 1234 \
      --vector-file tests/test-vectors/official.vec \
      --prefill-chunk 256 \
      --activation-bits 32 \
      --coordinator-layers 0:21 \
      --worker-layers 22:output \
      2>&1 | tee /tmp/issue304-phase3-official-16384-run${i}.log
done
```

Local golden `ctx=5000`:

```sh
for i in 1 2 3 4 5; do
  sleep 2
  DS4_METAL_DISABLE_METAL4=1 DS4_METAL_PREFILL_CHUNK=4096 \
    ./tests/issue304_phase3_vectors \
      --suite local-golden \
      --repeat 1 \
      --ctx-filter 5000 \
      --greedy-steps 8 \
      --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      --listen-host 10.77.0.1 \
      --listen-port 1234 \
      --local-golden-file tests/test-vectors/local-golden.vec \
      --prefill-chunk 256 \
      --activation-bits 32 \
      --coordinator-layers 0:21 \
      --worker-layers 22:output \
      2>&1 | tee /tmp/issue304-phase3-local-golden-5000-run${i}.log
done
```

## Runtime knobs used by `--local-payload-resume`

- `DS4_METAL_PREFILL_CHUNK=4096`
- `DS4_METAL_DISABLE_METAL4=1`
- `DS4_METAL_MOE_TILE_MAX` unset

## Observed payload probe

The Phase 1 probe creates a session with `ctx=16384` and saves a 1-token local checkpoint to inspect the durable `DSV4` layout.

## Phase 3.5 six-route matrix

Build the Phase 3.5 tools locally:

```sh
make tests/issue304_phase35_vectors
```

Sync the tree to the DGX host:

```sh
rsync -az --delete \
  --exclude='.git/' \
  --exclude='gguf/' \
  --exclude='*.o' \
  --exclude='ds4' \
  --exclude='ds4_test' \
  ./ dgx-direct:~/ds4/
```

Build remotely:

```sh
ssh dgx-direct 'cd ~/ds4 && make tests/issue304_phase35_vectors ds4'
```

Run the full matrix with the stable remote host form and DGX power limit:

```sh
python3 tests/issue304_phase35_matrix.py \
  --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --remote-host ilo037@10.77.0.2 \
  --remote-root ~/ds4 \
  --repeat 5 \
  --power 50
```

Targeted reruns are case-filterable:

```sh
python3 tests/issue304_phase35_matrix.py \
  --model ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --remote-host ilo037@10.77.0.2 \
  --remote-root ~/ds4 \
  --suite official \
  --case short_code_completion \
  --repeat 5 \
  --power 50
```

Observed operating notes:

- `ssh dgx-direct` aliasing was less reliable through the driver subprocess path than the explicit `ilo037@10.77.0.2` form.
- Route `6` needs worker shutdown before opening the remote CUDA payload-resume engine.
- Raw outputs were written under `artifacts/issue-304/phase35/raw/`.

- `ctx_size=16384`
- `prefill_cap=4096`
- `raw_cap=4352`
- `raw_window=128`
- `comp_cap=4098`
- `saved_tokens=1`
- `n_layer=43`
- `head_dim=512`
- `indexer_head_dim=128`
- `vocab=129280`
- `raw_live=1`

## Notes

- The actual resume cases run with `ctx=192` (`raw_window + 64`) after the probe discovers `raw_window=128`.
- The original focused Phase 1 harness is same-backend only: Metal save -> Metal load.
- The cross-backend matrix helper used the same `ctx=192`, `frontier=129` boundary case and `tests/long_context_story_prompt.txt` prompt on both hosts.
- `CUDA -> Metal` is currently the only failing Phase 1 whole-payload matrix cell.
