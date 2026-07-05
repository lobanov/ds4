# DSpark drafter GGUF converter (research-scoped copy)

Research-only copy of the DSpark drafter GGUF converter tooling, retained here so
the issue-468 dossier is self-contained and reproducible without reaching into
the main `gguf-tools/` build (which is kept untouched).

## Provenance

Copied verbatim from the sibling `ds4-dspark` worktree on 2026-07-05:

- `deepseek4-quantize.c`  — newer than this repo's `gguf-tools/deepseek4-quantize.c`
- `quants.c`              — byte-identical to this repo's `gguf-tools/quants.c`
- `quants.h`              — byte-identical to this repo's `gguf-tools/quants.h`
- `build_dspark_template.py` — byte-identical to this repo's `gguf-tools/build_dspark_template.py`

The only meaningful difference vs this repo's `gguf-tools/deepseek4-quantize.c` is
drafter (`mtp.`) support: `parse_expert_tensor` recognizes `mtp.N.ffn_<part>_exps`,
the expert HF-name builder emits `mtp.%d.ffn.experts.%d.%s`, and `mtp_unique_map`
maps the drafter input/output-stage tensors (`main_proj`, `main_norm`, `norm`,
`hc_head_*`, `markov_head`, `confidence_head`). Without these the converter cannot
build a drafter GGUF.

## Build

```sh
make
```

produces `./deepseek4-quantize`. Built and used only inside this repo; the binary
is gitignored.

## Why a separate copy

Dossier hygiene rules require research-only instrumentation to be marked clearly
and kept separate from production code. Mirrors the `issue468/dspark_oracle/`
pattern.

## Intended use (issue 468)

Generate an **unquantized** drafter GGUF (F32 ceiling) from the vendored
HuggingFace safetensors, to measure the theoretical draft-quality ceiling of the
DSpark drafter independent of Q4_K routed-expert quantization. Type overrides are
applied via policy flags (`--experts`, `--attention`, `--shared`, `--dense`, ...);
the template only supplies tensor names/shapes/order.
