# Issue 304 Compatibility Matrix

## Phase 0: Distributed merged payload stage

| Date | Commit | Model | Route | Worker ctx | Coordinator ctx | Backend pair | Result | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` local / `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe` DGX | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | `0:21 -> 22:output` | 32768 | 16384 | Metal -> CUDA distributed route, merged `DSV4` save on Metal | Fail | Inference route was healthy, but merged save rejected mismatched shard layout: `field=ctx local=16384 remote=32768 remote_layers=22:42`. |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` local / `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe` DGX | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | `0:21 -> 22:output` | 16384 | 16384 | Metal -> CUDA distributed route, merged `DSV4` save on Metal | Pass | Immediate `ds4_session_stage_payload()` succeeded after distributed prefill; staged payload bytes `221,006,660`. |

## Phase 1: Whole-payload local resume

| Date | Commit | Model | Save backend | Load backend | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | Metal | Metal | Pass | New `./ds4_test --local-payload-resume` path passed at frontiers `1, 3, 4, 5, 127, 128, 129`; logits and greedy continuation matched exactly. |
| 2026-06-03 | save on `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe` DGX, load on local `9952733bbc75eedfa1308cfed71b8e2694db978b` | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | CUDA | CUDA | Pass | `tests/issue304_phase1_matrix` at `ctx=192`, `frontier=129`, prompt `tests/long_context_story_prompt.txt`: top1 matched, top5 overlap `5/5`, top20 overlap `20/20`, `rms=0`, greedy continuation matched for 8 tokens. |
| 2026-06-03 | save on local `9952733bbc75eedfa1308cfed71b8e2694db978b`, load on `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe` DGX | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | Metal | CUDA | Pass | `tests/issue304_phase1_matrix` at `ctx=192`, `frontier=129`: top1 matched, top5 overlap `5/5`, top20 overlap `20/20`, `rms=0`, greedy continuation matched for 8 tokens. |
| 2026-06-03 | save on `477c0e82e2699b35a65fd0a1ed6fe66b41087dfe` DGX, load on local `9952733bbc75eedfa1308cfed71b8e2694db978b` | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | CUDA | Metal | Fail | `tests/issue304_phase1_matrix` at `ctx=192`, `frontier=129`: restore-point logits matched exactly (`top1`, top5/top20, `rms=0`, `max_abs=0`), but 8-token greedy continuation diverged at step `2` with saved token `5655` vs restored token `305`. |

## Representative `DSVL` shard smoke

| Date | Commit | Backend | Layer | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` | Metal | `0` | Pass | Single-layer `DSVL` save/load/save round trip was byte-identical. |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` | Metal | `21` | Pass | Single-layer `DSVL` save/load/save round trip was byte-identical. |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` | Metal | `2` | Pass | Single-layer `DSVL` save/load/save round trip was byte-identical; representative ratio-4/indexer layer. |
| 2026-06-03 | `9952733bbc75eedfa1308cfed71b8e2694db978b` | Metal | `42` | Pass | Single-layer `DSVL` save/load/save round trip was byte-identical; final/output-adjacent layer. |

## Matrix status

- Every planned Phase 1 backend pair has now been exercised at least once.
- The only failing cell is `CUDA -> Metal`, and the current failure is narrower than a restore-point logit mismatch: the loaded Metal session starts from identical logits but diverges during subsequent greedy decode.
- Forced-token follow-up on the same failing cell shows that logits already diverge after the first identical post-load eval token, so the failure is not explained by token-selection branching alone.
