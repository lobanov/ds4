# Issue 468 Research Dossier

This directory is the curated home for DSpark research artifacts on the `dspark-research` branch.

The goal is to keep the active research record compact, reviewable, and reproducible.

## Directory contract

- `STATUS.md` — canonical current state, conclusions, and next actions
- `summaries/` — compact topic summaries and negative-result consolidations
- `inventories/` — instrumentation, retained tools, and artifact inventories
- `artifacts/` — machine-readable summaries and a small set of retained evidence
- `ref/` — upstream references used by the research, including the DSpark paper and official source materials
- `archive/` — superseded notes, bulky raw evidence manifests, and historical material not part of the active dossier

## Principles

- Prefer curated summaries over chronological note accumulation.
- Prefer compact `csv` / `json` outputs over large raw logs.
- Retain only artifacts that support a conclusion, repro path, or reusable tool.
- Make supersession explicit: if a note or artifact is no longer canonical, record that in `STATUS.md`.

## Minimum contents expected

Each active research topic should eventually contribute:

1. a short summary of the question asked,
2. the current evidence-backed answer,
3. links to retained artifacts,
4. any remaining open questions.

## Initial structure

This branch starts with the framework only. Populate it by moving or summarizing research material into this layout rather than recreating a long chronological note chain.
