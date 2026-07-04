# Plain baseline matrix — prompt corpus × 3 temperatures × 3 contexts

Date: 2026-07-04.

Purpose: repeat the **plain local baseline** measurement using the later prompt corpus and record the results in the compact research dossier.

## Matrix

Prompt corpus:
- `issue468/prompts/baseline_corpus/code.txt`
- `issue468/prompts/baseline_corpus/creative.txt`
- `issue468/prompts/baseline_corpus/repetitive.txt`
- `issue468/prompts/baseline_corpus/structured.txt`

Temperatures:
- `0.0`
- `0.5`
- `1.0`

Contexts:
- `4096`
- `8192`
- `16384`

Common settings:
- backend: `metal`
- `n=128`
- `seed=1`
- model: `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

Total runs:
- `4 prompts × 3 temps × 3 ctx = 36 runs`

## Raw retained artifacts

Stored under:
- `issue468/artifacts/plain_baseline_matrix/logs/`
- `issue468/artifacts/plain_baseline_matrix/summary.csv`
- `issue468/artifacts/plain_baseline_matrix/summary.json`

## High-level result

The plain baseline remains very stable across this matrix.

- overall average generation throughput: **37.967 t/s**
- best cell: **39.29 t/s** (`creative`, `temp=0.0`, `ctx=8192`)
- worst cell: **36.37 t/s** (`repetitive`, `temp=1.0`, `ctx=8192`)

## Average generation t/s by context

| ctx | avg generation t/s |
|---|---:|
| 4096 | `37.959` |
| 8192 | `37.919` |
| 16384 | `38.024` |

Interpretation:
- on this short-prompt corpus, plain local decode is effectively **context-flat** from 4k to 16k.

## Average generation t/s by temperature

| temp | avg generation t/s |
|---|---:|
| 0.0 | `38.77` |
| 0.5 | `37.457` |
| 1.0 | `37.674` |

Interpretation:
- `temp=0` is the fastest slice in this matrix.
- `temp=0.5` and `temp=1.0` are slightly slower but still tightly clustered.

## Average generation t/s by prompt family

| prompt | avg generation t/s |
|---|---:|
| code | `38.083` |
| creative | `38.01` |
| repetitive | `37.589` |
| structured | `38.186` |

Interpretation:
- structured, code, and creative are all near 38 t/s.
- repetitive is the weakest family, but only modestly lower.

## Representative rows

| prompt | temp | ctx | prefill t/s | generation t/s |
|---|---:|---:|---:|---:|
| code | 0.0 | 8192 | `120.52` | `39.08` |
| creative | 0.0 | 8192 | `89.16` | `39.29` |
| repetitive | 1.0 | 8192 | `92.33` | `36.37` |
| structured | 0.5 | 16384 | `155.68` | `37.22` |

## Conclusion

This matrix is the updated retained **plain baseline denominator** for the later prompt corpus on the compact dossier branch.

Main takeaway:
- plain local inference is stable at about **38 t/s** across these prompts and across `4k → 16k` context settings, with only modest temperature sensitivity.
