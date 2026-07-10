# Lead 04 — FP ceiling capture (Modal + vLLM)

## Scripts (after cleanup — see .pi/skills/vllm-on-modal/SKILL.md for methodology)

| File | Purpose | Status |
|---|---|---|
| `capture_hc_modal.py` | Modal TP=2 app: load native DeepSeek-V4-Flash, post-load apply_model hooks, ds4-format prompt, fetch HC residual via apply_model | **VALIDATED** (v7: p1=0.74 GREEN, non-repeating greedy, replicated ranks) |
| `dspark_hc_patch.py` | The hook module: register_dspark_hooks (mhc_post on clones) + get_dspark_hc_buffer | **VALIDATED** |
| `convert_vllm_to_oracle.py` | Converter: capture [n,4,4096] per layer → mean(hc) → concat → main_hidden [n,12288] | **VALIDATED** (synthetic-tested) |
| `validate_fidelity.py` | Post-capture fidelity gate: shapes, D1, drafter-sanity p=1, GREEN/RED vs 0.8125 | **BUILT** (pending real-model run) |
| `test_capture_logic.py` | Mock unit test of the capture logic (layers 40/41/42 → slots, mhc_post+flatten) | **PASS** |

## Removed (obsolete)
- `capture.py` — extract_hidden_states approach (dead: EAGLE3 missing on DeepseekV4)
- `capture_hc_residual.py` — DGX-Spark local vLLM (blocked: DGX triton JIT broken)
- `smoke_test.py` — initial API derisk (superseded)
- `sitecustomize_dspark.py` — EngineCore import (abandoned: too-early import)
- `pilot.py` — original pilot (bugs, superseded by capture_hc_modal.py)

## Key findings (see SKILL.md for details)
- vLLM V1: model in EngineCore subprocess → use post-load `apply_model` hooks
- `extract_hidden_states` dead (EAGLE3 missing); `mhc_post` on clones works
- Chat template required (raw text → degenerate repeating greedy)
- `enforce_eager=True` required (cudagraphs break forward hooks)
- TP=2 minimum (native 129GB OOMs TP=1); HC residual replicated (no gather)
- ~2 toks/s at eager (acceptable for the 5-prompt pilot)
