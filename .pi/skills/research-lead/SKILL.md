---
name: research-lead
description: Playbook for executing a research lead in the issue468 dossier (a `pending/lead_*.md`). Use when assigned or resuming a lead.
---

# Research lead execution

A playbook for taking a `pending/lead_*.md` from "assigned" to a recorded,
decision-grade result in the `issue468/` dossier. It is written from the lessons
of Leads 01 (anchor-reuse falsifier) and 03 (acceptance statistical power) — two
leads where the obvious path was wrong in instructive ways.

This skill is **methodology**, not a script. Pair it with `adversarial-codex-review`
for the gates.

## When to use

- You are assigned a lead (a `pending/lead_*.md`) via `/goals` or `/sisyphus`.
- You are resuming a paused lead.
- You are about to record a lead's verdict (the moment overclaims hide).

## Prerequisites — know the dossier mechanics

- **Layout:** `issue468/{STATUS.md (canonical state), summaries/ (canonical results),
  pending/ (unstarted + in-progress leads), archive/leads/ (resolved leads),
  artifacts/ (kept machine-readable evidence), AGENTS.md (hygiene rules)}.**
- **Lead lifecycle:** `pending/lead_NN_*.md` is the **running worklog** while the lead
  is active (append decisions, calcs, capture logs, per-task outcomes); on completion
  it moves to `archive/leads/` (header → "resolved & archived", link to the canonical
  summary) and all links to it are updated. The canonical *result* goes to a new
  `summaries/*.md` + `STATUS.md`, never just the worklog.
- **Reusable assets accumulate:** check `pending/`, `summaries/`, `artifacts/`,
  `inventories/` before building anything new. Prior leads leave captures, stores,
  harnesses, manifests, and proven capture methods — reuse them.

## Core principles (the load-bearing lessons)

1. **Orient before acting.** Read the lead fully + every summary it references + the
   model/result it feeds. **Verify the assets it claims exist** — paths, binary
   capabilities, the prior numbers — don't trust "the tooling exists." *Lead 03:
   a default model path was off-by-one; `./ds4` is cwd-sensitive (needs the worktree
   root for `metal/`); the engine patch was "in the source tree" but uncommitted and
   the binary unbuilt.*
2. **Lock the decision rule + threshold BEFORE measuring.** "Within a few pp / a
   material drop" is not a verdict. Operationalize it as explicit tiers with CI
   conditions (e.g. SURVIVES if Δ≥0; COLLAPSES if Δ<0 and CI excludes 0; else
   MARGINAL). Use `goal_question`/`goal_questionnaire` if the lead is vague.
3. **The estimator IS the result.** Before trusting any number, ask: *is the metric
   I'm computing the one the downstream decision/model actually consumes?* The
   "obvious" estimator can be wrong by ~3 pp. **Name the estimator in every result.**
   *Lead 03: the sliding (position-uniform) prefix average overestimated per-cycle
   acceptance; the model currency is the cycle-jump (advance by accepted+1/cycle).
   The sliding headline was an overclaim a codex gate caught.*
4. **Fidelity-gate every implementation change.** Before trusting a variant or a new
   implementation, **reproduce a known-good baseline bit-for-bit.** A new code path
   that doesn't reproduce the baseline is wrong, no matter how plausible. *Lead 01:
   the modified harness reproduced retained summaries exactly before the variant was
   trusted. Lead 03: the torch port had to match the numpy oracle at 100% draft-token
   agreement before any powered number was trusted — and the gate initially FAILED,
   leading to a codex bug-hunt that found a real `hc_post` broadcast-axis bug.*
5. **Two codex gates for decision-grade work — setup + verdict.** One gate at setup
   (catch methodology/indexing bugs before you trust the numbers) and one at verdict
   (catch overclaims before you record). **Independently verify every decisive codex
   claim** (re-derive the number yourself). *Lead 01 GATE-1: the KV-window modeling +
   indexing semantics. Lead 03 GATE-1: the sliding-overclaim + the torch bug. Lead 03
   GATE-2: a stale "break-even" framing (2.203 vs the dynamic 2.256).*
6. **Scope the verdict honestly — significance ≠ non-inferiority.** Distinguish "no
   effect detected" from "effect absent." Report CIs and the MDE vs the decision
   margin, not just point estimates. If underpowered, say "non-inferiority NOT
   established," never "preserved/refuted." *Lead 01 GATE-2: "no large collapse
   observed" not "acceptance preserved."*
7. **Corpus representativeness drives the verdict.** Report **per-source/stratified**,
   flag the mix. A tightly-CI'd number on an easy corpus is still biased. If the
   verdict is corpus-limited (not precision-limited), more N from the same mix won't
   help — diversity might. *Lead 03: dolly/codealpaca/jsonex per-source E[a|4] spanned
   2.22–2.51; the old code/synthesis corpus was 2.175 (harder). The verdict was
   corpus-limited, so chasing N was over-powered.*
8. **The regime is part of the result.** For any headline number, record **which
   estimator, which accounting regime, and which policy/sample provenance** produced
   it. "1.04x" is incomplete; "anchor-reuse accounting, frozen threshold chosen on
   eval and replayed on lead3, 1.0375x" is decision-grade. This matters most for
   adaptive-policy leads where shipped vs optimized accounting, in-sample vs frozen
   policy, and deployable vs diagnostic policy can move the headline. *Lead 02:
   the same scheduler family ranged from shipped-losing to anchor-reuse-fragile
   depending on those qualifiers.*
9. **Single-process memory discipline.** This is a shared ~128 GB Apple-Silicon box
   running an 87 GB model. **NEVER run two heavy processes concurrently** (e.g. a
   multi-worker Python measurement + `ds4` loading the model, or two `ds4` captures).
   Know each tool's memory profile (per-call expert dequant spikes; mmap'd GGUFs; the
   87 GB IQ2XXS resident set). **If it crashes once, switch to a lighter path — do not
   retry the same way.** *Lead 03: two machine restarts from oversubscription →
   switched the numpy oracle (per-expert re-dequant) to a torch/MPS port (experts
   loaded once).*
10. **Add a notation/glossary section to every table- or formula-heavy summary.** Define
   every symbol and term up front — estimator-qualified metrics (sliding vs cycle-jump
   `E[a|K]`), `S(K)`, costs (`decode_ms`/`verify_ms`), break-even, `CI`, `P(speed<1)`,
   per-source — so a number like "E[a|4]=2.2" is unambiguous on a later read. The
   glossary is the quick-reference; a separate prose section can hold the reasoning.
11. **Separate deployable evidence from diagnostic ceilings.** For adaptive or policy
   leads, explicitly label each result tier: deployable frozen policy, in-sample tuned
   policy, offline expected-value policy, oracle/upper bound. Do **not** headline a
   weaker tier as if it were stronger. *Lead 02: the frozen threshold was the clean
   evidence; expected-opt was diagnostic; oracle was only a ceiling.*

## Workflow

1. **Orient (read-only).** Read the lead + referenced summaries + the model it feeds.
   `grep` for the lead's name across the dossier to find dependencies. Verify claimed
   assets exist and binaries support the needed flags (`--help`, env vars). Note the
   machine state (free mem; nothing heavy running). Append an "orientation" worklog
   entry recording what you found + any asset discrepancies.
2. **Define the decision.** If the lead's success criteria are vague, run a focused
   `goal_questionnaire` to lock the metric, the threshold, and the estimator (sliding
   vs per-cycle; per-step vs clustered). Record the locked rule in the worklog BEFORE
   measuring.
3. **Build + fidelity-gate the instrument.** Implement/extend the harness. Add a
   fidelity gate that reproduces a known-good baseline (retained summary, or a
   reference implementation) **bit-for-bit / token-for-token**. The gate must pass
   before any new number is trusted. If the gate fails on a port/reimplementation,
   dispatch an `adversarial-codex-review` bug-hunt; fix; re-gate.
4. **Measure (memory-safe).** Run the measurement **single-process**, nothing heavy
   concurrent. Checkpoint per-unit (per-prompt JSON, skip-if-exists) so a crash is
   recoverable. Prefer the lighter/faster implementation that passed the fidelity gate.
   Watch swap; if it grows unbounded, kill and reduce scope (fewer workers, tighter
   cache cap, smaller corpus subset) rather than crash again.
5. **Pick the right estimator + compute CIs.** Compute the metric the downstream
   decision uses (not the easy-to-compute one). Use **prompt-clustered** bootstrap CIs
   (resample prompts, not steps — steps within a prompt are correlated). Report the
   point estimate, CI, MDE vs the decision margin, and per-source/stratified.
6. **Codex gate 1 (setup + measured numbers).** Before propagating, dispatch an
   `adversarial-codex-review` against the instrument, the estimator choice, the
   indexing (the d2t-class off-by-one risk), the CI method, and the headline.
   Independently verify every decisive finding; fix confirmed issues; re-run if
   material.
7. **Codex gate 2 (verdict).** Before recording, dispatch a second review against the
   verdict itself: does it follow from the data? Is it scoped honestly
   (significance vs non-inferiority)? Are stale framings (old break-evens, old
   estimators) creeping in? Independently verify; soften/expand as warranted.
8. **Propagate honestly + completely.** Write a canonical `summaries/<topic>.md`;
   update `STATUS.md` (conclusions + inventory); update `spec_speedup_model.md` (or
   whatever model the lead feeds) with the new numbers; **sweep downstream docs for
   stale numbers the finding changes** and flag them. When a resolved lead changes a
   canonical model/result summary, **rewrite that summary into the best current single
   account** rather than layering a "Lead NN update" overlay on top of superseded
   framing. Keep the lead file as the
   worklog (it stays in `pending/` until the goal completes). On completion, archive
   it to `archive/leads/` (update header + all links). **Commit on `dspark-research`.**

## Adaptive-policy evidence ladder

For any lead that evaluates an adaptive policy / scheduler / controller, report the
result tier explicitly and keep the headline attached to the strongest deployable tier
you actually measured:

1. **Oracle / upper bound** — chooses using realized future information. Ceiling only.
2. **Offline expected-value / expected-opt policy** — chooses from a modeled expectation
   using current-cycle features. Diagnostic, useful for value estimation, but not the
   clean deployment claim.
3. **In-sample tuned policy** — threshold/hyperparameter chosen and evaluated on the same
   slice. Useful for search, not the headline evidence.
4. **Frozen out-of-sample policy** — policy/threshold chosen on one slice and replayed
   unchanged on a fresh slice. This is usually the cleanest offline deployable-ish
   evidence.
5. **Live/implemented measurement** — policy actually executed in the target system.
   Strongest evidence when available.

When several tiers are present, lead with the strongest deployable tier you measured and
place diagnostic tiers after it. Never let oracle or expected-opt displace a weaker but
more decision-relevant frozen-policy result.

## Lead archival (on completion) — checklist

When the lead's goal completes, retire the worklog and consolidate links — do **not**
leave a resolved lead in `pending/` (it should hold only unstarted/in-progress leads):

1. **Move** `pending/lead_NN_*.md` → `archive/leads/lead_NN_*.md` with `git mv`
   (preserve history; mirror the Lead 01/03 layout).
2. **Rewrite its header**: `Status: in progress` →
   `Status: resolved & archived <date> (moved from issue468/pending/). Result: see issue468/summaries/<canonical>.md`.
3. **Sweep every link**: `grep -rn "pending/lead_NN_" issue468/` (exclude `/.venv/`
   and `/archive/`); update each hit to `archive/leads/lead_NN_`. Typical sites:
   `STATUS.md` inventory entry, the canonical `summaries/*.md`, codex-review prompt
   artifacts, and other leads that reference it.
4. **STATUS.md inventory** line: `worklog (running; stays in pending/)` →
   `worklog (resolved, archived): archive/leads/lead_NN_*.md`.
5. **Verify** no `pending/lead_NN_` references remain (the `grep` returns nothing
   outside venvs/archive); confirm `pending/` now holds only unstarted leads.
6. **Commit** on `dspark-research` with a message noting the move + link sweep.

The canonical *result* lives in `summaries/` + `STATUS.md`; the archived worklog is
*provenance*. Both must agree on the numbers.

## Worklog entry template (append to the lead file per task/decision)

```markdown
### YYYY-MM-DD — <task>: <one-line outcome>

<2-4 sentences: what was done, the key number/decision, the artifact path.>
<If a gate ran: what it caught, what you verified, what you fixed.>
<If a path was abandoned: why (e.g. memory), what replaced it.>
```

The worklog is the *provenance*; the `summaries/` file is the *canonical result*.
Don't let them drift — numbers must match.

## Guardrails

- **Never record a powered number from an implementation that hasn't passed a
  bit-for-bit fidelity gate against a known-good reference.** (Lead 03's torch port
  was 43%-wrong until a codex bug-hunt fixed `hc_post`.)
- **Never frame an underpowered result as "established."** Say "no large effect
  observed; non-inferiority not established" with the CI + MDE.
- **Never run the model-measurement and `ds4` (87 GB) concurrently**, nor multi-worker
  heavy Python. Single heavy process at a time; monitor swap.
- **Never let a symbol or estimator go undefined.** "E[a|4]=2.2" is ambiguous;
  "cycle-jump (per-cycle) E[a|4]=2.198" is not — and any summary with tables/formulas
  needs an upfront **Notation & definitions** section defining every symbol
  (estimator-qualified metrics, `S(K)`, break-even, `CI`, costs), not just an in-line
  mention.
- **Never report a headline number without its qualifiers.** At minimum: estimator,
  accounting regime, policy tier/provenance, and sample/split when those can change the
  conclusion.
- **Never commit research/engine work to an upstream feature branch.** Land it on
  `dspark-research`.
- **Never leave stale numbers downstream.** A lead that changes a headline number
  must sweep the docs that cite the old one (other `pending/leads`, `STATUS.md`, the
  model summary) and flag the supersession.
- **Never integrate a resolved lead as an overlay if it changes a canonical model.**
  Rewrite the target summary so it reads as one coherent current account.
- **Never present oracle / expected-opt / in-sample-tuned policy results as if they were
  frozen out-of-sample or deployed evidence.** Label the policy tier explicitly.

## Anti-patterns (from experience)

- **Trusting the "obvious" estimator** → the sliding-overclaim (Lead 03). Always ask
  which metric the downstream decision consumes.
- **Retrying a crashing run unchanged** → the second machine restart (Lead 03). A
  crash is a signal to change the approach, not the patience.
- **One codex gate at the end** → setup bugs poison all the numbers before the verdict
  gate sees them. Gate at setup AND verdict.
- **Pooling a corpus-mixed result into one headline** → hides that the verdict is
  corpus-limited (Lead 03: jsonex positive, dolly negative). Report stratified.
- **Archiving the lead without sweeping links** → dangling `pending/lead_NN_*.md`
  references. On archive, `grep` + update all paths.
