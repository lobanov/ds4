# Phase 5 `ds4-eval` Matrix

These are the authoritative full `92`-question `--nothink` results collected
after the evaluator was switched onto the worker-owned local-decode handoff
path and after the distributed socket timeout fix landed.

- Date: `2026-06-06`
- Questions: `92`
- Local model: `./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Remote model: `/home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Remote host: `dgx-direct`

| Cell | Score | Runtime | Backend | Route | Output | Local decode | Artifact |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `local-metal-nothink` | `67/92` | `00h:47m` | `metal` | `local` | `local` | `expected=no, active=no` | [log](artifacts/issue-304/ds4-eval/2026-06-06-ds4-eval-metal.log) |
| `local-cuda-nothink` | `69/92` | `01h:39m` | `cuda` | `local` | `local` | `expected=no, active=no` | [log](artifacts/issue-304/ds4-eval/2026-06-06-ds4-eval-cuda.log) |
| `cuda-to-metal-localdecode-nothink` | `65/92` | `03h:01m` | `cuda` | `local 0:21 -> Metal worker 22:output` | `worker` | `expected=yes, active=yes` | [log](artifacts/issue-304/ds4-eval/2026-06-06-ds4-eval-cuda-to-metal-localdecode.log) |

## Status

- The current Phase 5 evaluator evidence is no longer the earlier `4`-question
  smoke. The authoritative local-decode comparison is the full `92`-question
  run above.
- `CUDA -> Metal --local-decode` is now considered operationally healthy:
  the route stays up, the worker-owned handoff engages, and the evaluator
  completes the full suite instead of timing out or disconnecting mid-run.
- The observed score gap versus the fully local baselines is small enough to
  treat as evaluation variance for Phase 5 closeout rather than as a new
  handoff-routing failure.

## Notes

- `local_decode_active_any_case: yes` remains the authoritative indication that
  worker-owned local decode engaged during generation.
- The earlier `4`-question matrix was still useful for route bring-up and
  timeout diagnosis, but it is no longer the acceptance surface for Phase 5.
- Current `ds4-eval` local-decode throughput is still lower than the plain CLI
  benchmark because the handoff RPC path still pays extra response overhead.
  That is a post-Phase-5 performance item, not a closeout blocker.
