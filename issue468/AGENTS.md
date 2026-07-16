# Research Artifact Hygiene

This branch is for compact, high-signal DSpark research curation rooted at `issue468/`.

## Scope

- Keep production code separate from research artifacts.
- Keep only artifacts that materially support a decision, repro, or retained tool.
- Prefer compact summaries over chronological note sprawl.

## Required layout

Research material for this effort lives under `issue468/`.

- `issue468/README.md` — dossier overview and directory contract
- `issue468/STATUS.md` — the single canonical current-state **narrative** (Bottom line / Investigation arc / Findings by axis / What exists / Next step); update it by rewriting, not appending
- `issue468/summaries/` — compact findings and verdicts
- `issue468/inventories/` — instrumentation, tools, retained artifacts (canonical catalog: `inventories/dossier_inventory.md`)
- `issue468/artifacts/` — kept machine-readable outputs only
- `issue468/archive/` — superseded notes and bulky raw material manifests

## Artifact rules

1. Every kept artifact must have an owner entry in `inventories/dossier_inventory.md` (and, if it changes the current state, a STATUS.md narrative update).
2. Prefer `csv` / `json` summaries over full raw logs.
3. Keep representative logs only when they add information not present in summaries.
4. Do not commit local virtualenvs, caches, `__pycache__`, scratch notebooks, or machine-local temp outputs.
5. Large binary captures belong in external storage or `archive/` with a manifest, not in the active dossier.
6. If a result is superseded, record that in `STATUS.md` and move or reference it under `archive/`.
7. If code paths are retained only for research, mark them clearly as research-only instrumentation.

## Documentation rules

- `issue468/STATUS.md` is the **single canonical current-state narrative** (Bottom line / Investigation arc / Findings by axis / What exists / Next step). Update it as a **narrative, not a log**: a new finding **rewrites** the relevant axis finding, adds one Investigation-arc line, and refreshes the Bottom line if the headline moved. Never append a standalone result section, and never leave a superseded verdict reading as current — demote it to a one-line arc beat.
- The artifact/tool inventory lives in `inventories/dossier_inventory.md` (STATUS's "What exists" is a one-line pointer to it). Add new artifacts/tools/captures/harnesses there under the right axis group — not inside STATUS.
- New findings update STATUS first (per above), then link the deeper `summaries/<topic>.md` evidence.
- Negative results condense into the axis finding + a summaries topic, not scattered terminal notes.
- One current account per question: when a result supersedes an earlier one, rewrite the earlier framing rather than layering an update. Avoid multiple conflicting "final verdict" documents.

## Decision standard

A retained artifact should answer at least one of these:

- What is the current conclusion?
- What evidence supports it?
- How is it reproduced?
- What tooling is still useful?

If it answers none of these, archive it or drop it.
