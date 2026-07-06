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
- MTP bulk draft verifier bandwidth-binding audit (code audit, not a measurement):
  - summary: `issue468/summaries/mtp_verifier_bandwidth_binding.md`
- DFlash oracle + accepted-prefix comparison vs DSpark (resolved):
  - summary: `issue468/summaries/dflash_oracle_investigation.md`
  - weights: `issue468/dflash_drafter/` (gitignored); forward + harness: `issue468/dflash_oracle/`
  - captures: `issue468/artifacts/dflash_capture/` (30 cells, gitignored); results: `issue468/artifacts/dflash_acceptance/`
  - capture driver: `issue468/run_dflash_capture.py`

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
- **The MTP bulk draft verifier is memory-bandwidth-bound, not compute-bound,**
- **DFlash drafter comparison: DSpark is more attractive on this corpus.**
  DFlash oracle built and validated vs the MLX reference (<=0.08% rel); the only
  bug was a self-inflicted `d2t` token-mapping error (`d2t` is an offset, not an
  absolute map; correct decode `target_id = draft_idx + d2t[draft_idx]`), caught
  by an adversarial codex review after a brief mis-diagnosis as a representation
  blocker. On the same IQ2XXS target and offline per-step protocol, DFlash avg
  accepted prefix is 0.876 (7-token block) vs DSpark q4k's 2.171 (5-token block)
  — DSpark ~2.5x better. DFlash's parallel noise-block drafting rarely extends
  the prefix past position 1. DFlash pos1 acc 0.68 ≈ its val 0.74; positions 2-7
  underperform val (likely IQ2XXS effect on later positions / corpus). See
  `summaries/dflash_oracle_investigation.md`.

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
