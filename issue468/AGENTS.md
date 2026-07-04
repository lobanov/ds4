# Research Artifact Hygiene

This branch is for compact, high-signal DSpark research curation rooted at `issue468/`.

## Scope

- Keep production code separate from research artifacts.
- Keep only artifacts that materially support a decision, repro, or retained tool.
- Prefer compact summaries over chronological note sprawl.

## Required layout

Research material for this effort lives under `issue468/`.

- `issue468/README.md` — dossier overview and directory contract
- `issue468/STATUS.md` — current canonical state
- `issue468/summaries/` — compact findings and verdicts
- `issue468/inventories/` — instrumentation, tools, retained artifacts
- `issue468/artifacts/` — kept machine-readable outputs only
- `issue468/archive/` — superseded notes and bulky raw material manifests

## Artifact rules

1. Every kept artifact must have an owner note in `README.md`, `STATUS.md`, or an inventory.
2. Prefer `csv` / `json` summaries over full raw logs.
3. Keep representative logs only when they add information not present in summaries.
4. Do not commit local virtualenvs, caches, `__pycache__`, scratch notebooks, or machine-local temp outputs.
5. Large binary captures belong in external storage or `archive/` with a manifest, not in the active dossier.
6. If a result is superseded, record that in `STATUS.md` and move or reference it under `archive/`.
7. If code paths are retained only for research, mark them clearly as research-only instrumentation.

## Documentation rules

- Keep one canonical current-state summary in `issue468/STATUS.md`.
- New findings should update the canonical summary first, then link deeper evidence.
- Negative results should be condensed into topic summaries, not left as scattered terminal notes.
- Avoid multiple conflicting "final verdict" documents.

## Decision standard

A retained artifact should answer at least one of these:

- What is the current conclusion?
- What evidence supports it?
- How is it reproduced?
- What tooling is still useful?

If it answers none of these, archive it or drop it.
