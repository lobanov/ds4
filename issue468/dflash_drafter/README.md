---
library_name: speculators
tags:
- speculative-decoding
- dflash
- deepseek-v4-flash
- draft-model
base_model: deepseek-ai/DeepSeek-V4-Flash
---

# DFlash Speculator for DeepSeek-V4-Flash (all-SWA, Muon, 50k)

A [DFlash](https://github.com/neuralmagic/speculators) draft (speculator) model trained
for speculative decoding with **DeepSeek-V4-Flash** as the verifier.

This is the **best checkpoint** selected by validation loss from a 50k-sample training run.

## Key characteristics

- **Algorithm:** DFlash
- **Verifier / target:** `deepseek-ai/DeepSeek-V4-Flash`
- **Attention:** Sliding-window attention (SWA) on **all** draft layers (`sliding_window=2048`)
- **Optimizer:** Muon
- **Training data:** 50k samples
- **Draft layers:** 5 (`hidden_size=4096`, `head_dim=256`, `hc_mult=4`)
- **Speculative tokens:** 7 (block_size 8)
- **Aux hidden-state layers:** `[3, 13, 23, 32, 42]`
- **dtype:** bfloat16

## Validation metrics (best checkpoint)

| Metric | Value |
|---|---|
| Loss | 1.2559 |
| Full-sequence acceptance | 0.3452 |
| Position 1 acc | 0.7402 |
| Position 2 acc | 0.5120 |
| Position 3 acc | 0.3689 |
| Position 4 acc | 0.2746 |
| Position 5 acc | 0.2121 |
| Position 6 acc | 0.1691 |
| Position 7 acc | 0.1373 |

## Files

- `config.json` / `config.py` — speculator config and model definition
- `model.safetensors` — draft model weights
- `val_metrics.json` — validation metrics for this checkpoint
