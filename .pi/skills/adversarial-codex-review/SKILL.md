---
name: adversarial-codex-review
description: Dispatches an independent adversarial review of recent work via the codex CLI (default gpt-5.5 at xhigh reasoning). Use when an investigation is blocked, a conclusion feels uncertain, or you want assumptions challenged and bugs caught. The reviewer re-derives claims from code and runs experiments; you must independently verify its findings before accepting them.
---

# Adversarial codex review

Dispatches a second agent (codex) to independently and adversarially review work
you (pi) just did — challenging assumptions, re-deriving claims from code, and
running experiments to find what you missed. Designed for the moment you are about
to conclude "blocked", "infeasible", or "done": a fresh adversarial pass catches
self-inflicted bugs and unjustified leaps before they cost real effort. For the
research-lead workflow, this skill provides the codex gates (setup + verdict).
For torch-specific claims (dtype-invariance, the block-divergence, the offline-vs-live
gap), see `pytorch-numerical-modelling`.

## When to use

- You're about to pause/abort a goal as "blocked" — get a second opinion first.
- A conclusion rests on an assumption you inferred rather than verified (token
  maps, index conventions, dtype/layout, API semantics, encoding formats).
- A sub-validation passed (e.g. "forward matches a reference on random input")
  but the end-to-end result is wrong or untested.
- The user asks for an independent/adversarial/second-opinion review.
- You're about to record a STOP/negative verdict — a negative that's actually a bug
  kills a real lead (the most expensive error).

## Prerequisites

- `codex` CLI installed and authenticated (`codex --version`). Default model is
  `gpt-5.5` at `xhigh` reasoning effort; override per-run.
- The skill runs codex from the current project root; codex can read files and
  (in `read-only`, the default) run read-only commands and inline scripts; escalate
  to `workspace-write` for scratch experiments.

## Workflow

1. **Decide what to review.** From the user's args, or the most recent
   investigation: identify the canonical report/note (e.g. an
   `issue468/summaries/*.md`) plus the key code, captures, and any diagnostic
   scripts. If unclear, ask the user for the target.
2. **Fill the adversarial prompt template** below: paste the concrete paths,
   the claims you want challenged, and the symptom. Keep it concrete — name
   files, line numbers, and the exact claim that "X is ruled out".
3. **Dispatch codex** via the helper script (`./scripts/run_review.sh`). The
   prompt can be a file or piped via stdin; model/effort/sandbox are flags, so
   order is free:
   ```bash
   ./scripts/run_review.sh prompt.md                       # file, all defaults
   ./scripts/run_review.sh -m gpt-5.5 -e xhigh prompt.md   # file + flags
   cat prompt.md | ./scripts/run_review.sh                 # via stdin
   printf 'review X' | ./scripts/run_review.sh -e high -s read-only
   ./scripts/run_review.sh -e high <<'EOF'                 # heredoc
   …
   EOF
   ```
   The script prints the codex log path to stdout (`-h` shows full usage).
   - Defaults: `gpt-5.5`, `xhigh`, `read-only` (safe: codex can run read-only
     commands and inline scripts but cannot modify files under review).
   - Use `-s workspace-write` when the review needs to write scratch experiments;
     `-s danger-full-access` only for sibling repos / `/tmp` and arbitrary
     experiments (the script then passes
     `--dangerously-bypass-approvals-and-sandbox`).
   - Set a generous bash timeout (xhigh + tool use can run 10–25 min).
4. **Read the result.** The script writes codex's final message to
   `<log>.final.md` — read that for the structured output (verdicts, experiments,
   leading hypothesis); the full `<log>` holds the trace.
5. **INDEPENDENTLY VERIFY before accepting.** This is the most important step.
   For any decisive claim — especially "I found the bug" or "assumption X is
   wrong" — re-run the check yourself (a few lines of python/bash) and confirm
   the evidence. codex can be wrong, overconfident, or fooled by the same
   framing you were. Only report a finding as confirmed after your own check.
6. **Triage and report** to the user: confirmed findings (with your verification
   evidence), speculative ones, and the ranked next actions. Apply fixes/next
   steps and update the dossier accordingly.

## Adversarial prompt template

Copy, fill the `{{...}}` placeholders, and write to a temp file. The attack list
is deliberately tilted at the common ways a blocked-conclusion hides a simple bug
(this list was learned the hard way — keep it).

````
You are a skeptical senior engineer doing an ADVERSARIAL independent review. A
previous investigator (another agent) reached a conclusion you must challenge.
Re-derive every claim from the code; run code to verify where you can. Be terse.

WORKTREE: {{cwd}}. Python envs: {{venvs}}. Relevant binaries/paths: {{paths}}.

READ FIRST: {{canonical report/note path}}

RECAP: {{2-3 sentences: what was attempted, the symptom, the claimed conclusion.}}

THE INVESTIGATOR'S CLAIMS (verify, do not trust):
- {{claim 1, e.g. "forward validated vs reference X to <=0.08%"}}
- {{claim 2, e.g. "capture is correct / bit-identical to Y"}}
- {{claim 3, e.g. "ruled out A, B, C; leading hypothesis = Z"}}

PREMISES / SCOPE (exogenous — do NOT flag these as bugs; the investigator was
instructed to assume them. Instead assess (a) realizability and (b) how the
conclusions swing if a premise fails. Phrase results as "under the premise, X;
if the premise fails, Y" — NOT "the model/claim is wrong"):
- {{premise 1, e.g. "the verifier reuses the anchor token across cycles"}}
- {{premise 2, ... — or "none; all claims are the investigator's own"}}

YOUR MANDATE — challenge assumptions, find new avenues. Your decisive claims will
be INDEPENDENTLY re-verified by the dispatcher before acceptance, so prioritize
claims you can back with a runnable check or a precise citation. Not exhaustive:
1. END-TO-END EXISTENCE: was the full pipeline EVER shown to produce a correct
   result on ANY input? Sub-validations (unit A matches unit B) do not prove the
   end-to-end works. If no input yields a correct output, the bug is upstream of
   whatever was "ruled out" — suspect mapping/encoding (token maps, index/base
   conventions, dtype, layout), config, or a non-strict load dropping weights.
2. MAPPING SEMANTICS: are every map/encoding/index assumptions verified
   ALGEBRAICALLY (e.g. against a ground-truth mask or inverse), not inferred from
   names? Off-by-one, offset-vs-absolute, base-0-vs-1, row-vs-column major.
3. THE CONCLUSION ITSELF: is "blocked"/"infeasible" justified, or is it hiding a
   simple bug? Reproduce the one validation that would prove the conclusion;
   does it actually hold?
4. RE-DERIVE THE INTERMEDIATE: is the captured/derived value really what the
   reference expects? Trace the reference's forward and compare, step by step.
5. SYMPTOM READING: does the symptom (e.g. "common-token output", "near-constant
   loss") match the claimed cause, or a different one (degenerate regime,
   dominator input, wrong loss reduction, wrong axis)?
6. QUANTIFY, DON'T ASSERT: where a magnitude is claimed (e.g. "noise dominates"),
   compute it; watch outlier-inflated std vs robust RMS.
7. PREMISE SENSITIVITY: for each PREMISE above, state how tightly the conclusion
   depends on it and the magnitude of the swing if it fails or proves
   unrealizable. A premise that flips the headline IS the headline.
8. THE FALSE NEGATIVE (for a STOP/negative verdict): is the < bar result (a) a bug
   (the implementation, the measurement), (b) a missed signal/θ, or (c) a real
   fundamental limit? Reproduce the one validation that would flip the verdict. A
   negative that's actually a bug kills a real lead — the most expensive error.
   (Lead 11: the equal-length subset removed the eos contamination → still +0.17%;
   the off-by-one signal + the folding loss are fundamental, not bugs.)
9. SIGNAL-AT-THE-DECISION-POINT (for a gate/scheduler): is the signal available AT
   the decision point (not post-decision)? A post-decision signal (circular) forces
   a weaker proxy. (Lead 11: the first-draft margin isn't available pre-draft → the
   weaker anchor margin → no speedup.)

IF THIS IS A MODELING / CONCLUSIONS REVIEW (not a bug hunt), also challenge:
M1. ASSUMPTION REALIZABILITY: is each load-bearing assumption achievable on the
    actual system (cite the implementing code), or merely asserted?
M2. PARAMETER SENSITIVITY: how far does the headline move if each numeric input
    (costs, timings, acceptance) is off by ±10/25%? Name the tightest-margin input.
M3. STATISTICAL SOUNDNESS: are quoted precisions supported by the data (n,
    variance)? Distinguish a mean effect from per-sample spread.
M4. DATA REPRESENTATIVENESS: was the input distribution measured on
    deployment-matching conditions (prompt lengths, temperatures, trajectory
    type) or a convenience sample?
M5. COMPLETENESS: is any cost/overhead term set to zero that is nonzero in
    practice (readback, partial-accept rollback, first-miss waste, state capture)?
M6. OFFLINE-VS-LIVE (runtime/engine leads): is the throughput measured on the REAL
    engine (the live drafter + the real timings) or an offline simulation/oracle? The
    offline drafter may not reproduce the live engine's acceptance (Lead 11:
    D_f32/D_f16 diverged ±50% from the live Metal). If the verdict rests on an
    offline simulation, flag it — the live-engine measurement is the deployable evidence.
    (See research-lead principle 12 + pytorch-numerical-modelling principle 2.)

CONSTRAINTS: do not modify tracked files. You may run read-only commands and
inline scripts (python -c, rg, jq, etc.) to verify claims.

OUTPUT (terse, evidence-based):
## Verdict per claim (sound / questionable / likely-wrong + 1-line why)
## Premise sensitivity (per premise: does the headline survive if it fails?)
## New experiments to try (ranked; each: what to run, expected signal, effort)
## Leading hypothesis after re-examination + the one decisive test for it
````

## Notes & guardrails

- **Security.** NEVER share any secrets/tokens/passwords with codex.
- **Verify, don't defer.** codex is a strong but fallible second pair of eyes,
  not an oracle. The skill's value is the *independent verification* step, not
  the raw codex output. Never report a codex finding as fact without your own
  check.
- **Cost.** gpt-5.5 xhigh with tool use is slow and token-heavy. Use it for
  genuine stuck points, not routine checks. Lower to `high`/`medium` for lighter
  passes.
- **Security.** Default sandbox is `read-only` (codex can run read-only commands
  and inline scripts but cannot modify files — safest for a review). Escalate to
  `workspace-write` for scratch experiments, or `danger-full-access` for sibling
  repos / `/tmp` when the user has effectively authorized it.
- **Moving the skill global:** copy this directory to `~/.pi/agent/skills/` to
  use it across all projects.
