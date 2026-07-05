# Issue 468 Status

## State

Framework established for a compact DSpark research dossier.

This branch starts from `main` and is intended to host a curated, low-noise version of the DSpark research record under `issue468/`.

## Current objective

Create a trustworthy active dossier that makes it easy to:

- understand the actual DSpark research goal,
- identify the current conclusions,
- retain useful tools and instrumentation,
- and prevent noisy experimental artifacts from dominating the branch.

## What exists now

- `issue468/README.md` — overview and directory contract
- `issue468/GOAL.md` — distilled research objective from `PLAN.md`
- this `issue468/STATUS.md` — canonical current-state page
- `issue468/AGENTS.md` — artifact hygiene rules
- dossier scaffold for summaries, inventories, artifacts, archive, and references
- retained DSpark numpy oracle with local smoke-checked venv setup:
  - code: `issue468/dspark_oracle/`
  - inventory: `issue468/inventories/numpy_oracle.md`
  - summary: `issue468/summaries/numpy_oracle_retention.md`
- retained exactness/debug bundles for the small exactness corpus across `temp=0/0.5/1.0`:
  - inventory: `issue468/inventories/exactness_small_bundles.md`
  - summary: `issue468/summaries/exactness_small_bundles_and_oracle_acceptance.md`
  - artifacts: `issue468/artifacts/exactness_small_bundles/summary.json`
  - artifacts: `issue468/artifacts/exactness_small_bundles/summary.csv`
- updated retained plain-baseline matrix for the later prompt corpus:
  - summary: `issue468/summaries/plain_baseline_matrix.md`
  - artifacts: `issue468/artifacts/plain_baseline_matrix/summary.csv`
  - artifacts: `issue468/artifacts/plain_baseline_matrix/summary.json`
- DSpark drafter quantization ceiling (Q4_K vs F16 vs F32) — measured and closed:
  - summary: `issue468/summaries/dspark_quantization_ceiling.md`
  - converter (research-scoped copy): `issue468/dspark_converter/`
  - bulk acceptance harness: `issue468/run_ceiling_bulk.py`
  - F16 ceiling drafter: `issue468/artifacts/dspark_ceiling/dspark_f16.gguf` (gitignored, reproducible)
  - acceptance results: `issue468/artifacts/exactness_small_acceptance/f16_ceiling/` and `q4k_baseline/`
  - diagnostics: `issue468/archive/diagnostics/`

## Current conclusions

- **Q4_K is not the draft-quality bottleneck.** Removing routed-expert quantization
  (F16/F32 drafter from the vendored MXFP4 source) yields no material acceptance
  gain: 3.5% of draft tokens flip vs Q4_K, but net accepted-prefix change is
  +0.38% overall (within noise; sign inconsistent across temperatures).
- The vendored drafter ships as MXFP4 (4-bit); F16 already captures its full
  dequant (F16≡F32 at the weight level), so no available precision beats Q4_K.
- This falsifies the "Q4_K drafter quality causes the verifier-dominated cycle"
  hypothesis. The cycle overhead originates elsewhere (verification / KV-replay /
  scheduling), not draft precision. Research should redirect away from drafter
  quantization.
- This supersedes the sibling `ds4-dspark` dossier notes 22 and 51 (buggy oracle;
  did not measure the HF-source ceiling).

## Next recommended steps

1. Create inventories for:
   - instrumentation in code
   - retained tools/scripts
   - retained artifacts
2. Add compact topic summaries for:
   - accepted findings
   - false leads / exhausted directions
   - current open questions
3. Move bulky or superseded material to archive references rather than keeping it active.

## Canonicality rule

This file is the source of truth for the branch's current research state.
Any deeper notes should support this file, not contradict it.
