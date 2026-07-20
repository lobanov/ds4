## Verdict

- **C1 faithful: questionable.** Directionally matches ds4 F32-internal `hc_head`, but not line-by-line: torch scales the dot after matmul, downcasts `hc_head` output to F16, and LoRA effective weights can be formed in F16.
- **C2 F16==F32: likely-wrong as stated.** The trained artifact’s `|diff|=0` is aggregate p1 equality, not forced equality. Untrained lead10 CPU check: F32 `0.850084`, F16 `0.849523`, top-1 equal `1777/1781`.
- **C3 complete: questionable.** No current-cache overflow in `k_scores`/`forward`, but those norms are still F16 and unlike ds4’s F32 `rms_norm_weight`.

## Unfixed `k_scores` / `forward`

They are in the full drafter path: torch lines [96](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_head.py:96) and [131](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_head.py:131); ds4 does F32 `rms_norm_weight` before logits at [28852](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28852).

Probe results:
- `eval_kloss`: `34485` draft rows, `h_absmax=38.94`, F16 norm `inf_rows=0`.
- `train_kloss`: `104115` draft rows, `h_absmax=42.81`, F16 norm `inf_rows=0`.
- So frozen cached features do not overflow there, but the implementation is still not faithful/robust if LoRA changes HC weights enough to make `|h| > 256`.

## Body F16

Torch F16 body is not faithful to ds4. `drafter_body.py` uses dtype-local norms/MoE (`x*x` in dtype) at [33](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:33), [61](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:61), [93](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:93). ds4 body scratch is `float *` and calls F32 kernels, e.g. [28402](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28402).

So F16 torch-body overflow is an oracle artifact. It does not prove ds4 deployment failure. For deployment, test against live ds4/Metal or CPU drafter, not torch F16 body.

## F16==F32 Re-Derivation

MPS was unavailable in this tool process; I ran CPU checks.

- Old simulated F16 `hc_head`: lead10 `flat_ss inf` on `5301/5451` rows; later norm zeroed `1692` rows.
- Fixed `hc_head`: lead10 `h_absmax=40.28`, no final p1 norm overflow.
- Exactness p1: F16/F32 top-1 equal `80/80`, but logits not equal (`max_abs_diff=0.04797`).
- Lead10 untrained p1: F32 `0.850084`, F16 `0.849523`, top-1 equal `1777/1781`.

## Other Missed F16 Spots

`lm_head` + Markov bias looked safe on exactness: base absmax `37.38`, bias absmax `87.31`, logits absmax `89.88`, all finite. Confidence logits also finite, absmax `8.96`.

Main missed spot is not observed overflow today; it is fidelity: `forward`/`k_scores` final RMSNorm should be F32 like ds4.

## Leading Hypothesis + Decisive Test

Hypothesis: the +2.4 pp likely survives for p1, but the current proof is too narrow and not a full F16 deployment proof.

Decisive test: bake/export the trained `hc_head_fn` LoRA into the dspark GGUF, run live ds4 CPU/Metal drafter on the same held-out prompts, and compare accepted p1/full-block draft top-1 against the torch F32 oracle row-by-row.