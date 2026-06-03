# Issue 304 Runbook

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
- Local repo commit: `9952733bbc75eedfa1308cfed71b8e2694db978b`
- DGX worker repo commit: `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe`
- Coordinator host: Apple M5 Max / Metal
- Worker host: NVIDIA GB10 / CUDA
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Local model path: `/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- DGX model path: `/home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model hash: not recorded in this pass; `shasum -a 256` was skipped because the file is too large for a short verification loop

## Commands

### Phase 1 build

```sh
make ds4_test
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

## Runtime knobs used by `--local-payload-resume`

- `DS4_METAL_PREFILL_CHUNK=4096`
- `DS4_METAL_DISABLE_METAL4=1`
- `DS4_METAL_MOE_TILE_MAX` unset

## Observed payload probe

The Phase 1 probe creates a session with `ctx=16384` and saves a 1-token local checkpoint to inspect the durable `DSV4` layout.

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
