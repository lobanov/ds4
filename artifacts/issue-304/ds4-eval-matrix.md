# Phase 5 `ds4-eval` Matrix

- Date: `2026-06-06`
- Questions: `4`
- Local model: `./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Remote model: `/home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Remote host: `dgx-direct`

| Cell | Score | Runtime (s) | Prompt toks | Gen toks | Prefill (s) | Decode (s) | Backend | Route | Hops | Output | Local decode | Trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| `local-metal-nothink` | 4/4 | 28.492 | 744 | 901 | 4.872 | 23.620 | `metal` | `local` | 0 | `local` | `expected=no, active=no` | [trace](artifacts/issue-304/ds4-eval/raw/2026-06-06-local-metal-nothink.trace) |
| `local-cuda-nothink` | 4/4 | 62.275 | 744 | 848 | 3.664 | 58.612 | `cuda` | `local` | 0 | `local` | `expected=no, active=no` | [trace](artifacts/issue-304/ds4-eval/raw/2026-06-06-local-cuda-nothink.trace) |
| `metal-to-cuda-localdecode-nothink` | 4/4 | 86.553 | 744 | 859 | 5.439 | 81.113 | `metal` | `local 0:21 -> 10.77.0.2:41039 Q2 22:output` | 1 | `worker` | `expected=yes, active=yes` | [trace](artifacts/issue-304/ds4-eval/raw/2026-06-06-metal-to-cuda-localdecode-nothink.trace) |
| `cuda-to-metal-localdecode-nothink` | 3/4 fail=1 | 162.930 | 744 | 828 | 9.680 | 153.250 | `cuda` | `local 0:21 -> 192.168.1.218:49540 Q2 22:output` | 1 | `worker` | `expected=yes, active=yes` | [trace](artifacts/issue-304/ds4-eval/raw/2026-06-06-cuda-to-metal-localdecode-nothink.trace) |
| `metal-to-cuda-full-nothink` | 4/4 | 49.667 | 744 | 812 | 5.434 | 44.235 | `metal` | `local 0:21 -> 10.77.0.2:34989 Q2 22:output` | 1 | `worker` | `expected=no, active=no` | [trace](artifacts/issue-304/ds4-eval/raw/2026-06-06-metal-to-cuda-full-nothink.trace) |
| `cuda-to-metal-full-nothink` | 4/4 | 85.975 | 744 | 817 | 8.690 | 77.287 | `cuda` | `local 0:21 -> 192.168.1.218:49923 Q2 22:output` | 1 | `worker` | `expected=no, active=no` | [trace](artifacts/issue-304/ds4-eval/raw/2026-06-06-cuda-to-metal-full-nothink.trace) |

## Status

- Latest completed `6 x 4` `--nothink` smoke: `5/6` cells passed.
- Passing cells: local Metal, local CUDA, `Metal -> CUDA` local-decode, `Metal -> CUDA` full-distributed, `CUDA -> Metal` full-distributed.
- Remaining failing cell: `CUDA -> Metal` local-decode, failing `SuperGPQA/001b51d76b4d422988f2c11f104a2c6c` with picked `G` vs expected `C`.

## Notes

- `local_decode_expected: yes` with header `local_decode_active: no` is not treated as a routing failure here. The stronger summary field is `local_decode_active_any_case`, and for both `*-localdecode-*` cells it is `yes`, which shows local decode did engage during generation.
- The runner now includes the extra Phase 5 fully distributed cells:
  - `metal-to-cuda-full-nothink`
  - `cuda-to-metal-full-nothink`
- The runner was hardened during troubleshooting to:
  - use stable SSH options for `dgx-direct` (`IPQoS=none`, no control master/path)
  - restart workers per case in fresh-per-case mode
  - wait for worker readiness before starting each case
  - bypass the DGX startup memory guard for remote worker restarts
  - clear stale per-case trace files before merge
  - wait for remote worker shutdown before restart

## Interpretation

- Full-distributed mode is currently healthy in both directions on this `4`-question `--nothink` smoke.
- `Metal -> CUDA` local-decode also passes after the runner cleanup.
- The remaining issue is isolated to `CUDA -> Metal` local-decode under the smoke harness, not to full-distributed routing in general.
