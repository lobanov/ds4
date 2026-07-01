# DSpark Speculative Decoding

DSpark is a speculative-decoding drafter for DeepSeek-V4-Flash. It runs a small
multi-layer drafter forward to propose a block of draft tokens, then verifies
them with the target model via B2 rejection sampling (exactness-preserving: the
output is guaranteed identical to target-only decoding regardless of draft
quality). See `@issue468/32_research_handoff_note.md` (research branch) for the
full design, measurements, and architecture.

This document covers only **building the drafter GGUF** and **running it**.

## Build the drafter GGUF

The drafter GGUF (`dspark.gguf`, ~10.7 GiB, 81 tensors) is **not** shipped and is
**not** available as a prebuilt file in the GGUF download repo. It must be built
from the Hugging Face safetensors of `deepseek-ai/DeepSeek-V4-Flash-DSpark`
(shards 46-48, ~10.9 GB). The build is two stages:

1. `gguf-tools/build_dspark_template.py` writes a metadata-only template GGUF
   (tensor names, shapes, target quant types, KV metadata — zero weight data).
2. `gguf-tools/deepseek4-quantize` reads the template + the HF safetensors and
   regenerates the quantized bytes (Q4_K routed experts, Q8_0 attention, F16
   gate/hc matrices, F32 norms, BF16 markov/confidence heads).

### One command

```sh
# 1. Fetch the drafter source shards (shards 46-48 + index, ~10.9 GB):
./download_model.sh dspark

# 2. Build the drafter GGUF (writes gguf/dspark.gguf):
make dspark-gguf
```

`./download_model.sh dspark` requires the Hugging Face CLI
(`python3 -m pip install -U huggingface_hub hf_xet`). It downloads into
`gguf/dspark-hf/`. `make dspark-gguf` then builds `deepseek4-quantize` (if needed)
and runs the conversion.

### Overrides

The Makefile targets accept overrides:

```sh
make dspark-gguf DSPARK_HF_DIR=/path/to/hf-dspark DSPARK_GGUF=/custom/out.gguf
```

| variable | default | meaning |
|---|---|---|
| `DSPARK_HF_DIR` | `$(CURDIR)/gguf/dspark-hf` | dir with shards 46-48 + `model.safetensors.index.json` |
| `DSPARK_GGUF` | `$(CURDIR)/gguf/dspark.gguf` | output drafter GGUF path |
| `DSPARK_TEMPLATE` | `$(CURDIR)/gguf/dspark_template.gguf` | intermediate template path |

Individual stages: `make dspark-template` (template only), `make dspark-quantizer`
(build the quantize tool).

### Reproducibility

This toolchain reproduces the validated `dspark.gguf` **byte-for-byte** from the
HF shards (verified: `cmp` of a fresh rebuild against the validated artifact is
identical). The quant recipe embeds all validated fixes from the research:

- `ffn_gate_inp` **F16** (the root-cause fix — F32 caused garbage expert routing)
- `hc_attn_fn` / `hc_ffn_fn` / `hc_head_fn` **F16** (matmul_f16 kernel requirement)
- `main_norm` / output `norm` **F32** (rmsnorm Metal kernel is F32-only)
- markov_head / confidence_head **BF16** (accuracy-critical)

## Run inference

DSpark is enabled with `--dspark <drafter.gguf>`. The target model is passed with
`-m` as usual.

```sh
./ds4 -m <target.gguf> --dspark gguf/dspark.gguf -c 8192 -n 256 --temp 0.0 -p "PROMPT"
```

Diagnostics: `DS4_DSPARK_B2_DEBUG=1` prints per-cycle acceptance (draft tokens
accepted/rejected, correction tokens). `DS4_DSPARK_DISABLE=1` disables DSpark at
runtime even when `--dspark` is passed (useful for A/B comparison against the
target-only baseline).

Quality: `./ds4-eval -m <target.gguf> --dspark gguf/dspark.gguf --plain --nothink -n 4096 --seed 1 --questions 92 --temp 1.0`.

## Status

The drafter forward and B2 verification loop are validated and produce clean,
correct output. The end-to-end speedup is gated on eliminating the KV-replay
overhead on partial-accept cycles (productionization in progress).
