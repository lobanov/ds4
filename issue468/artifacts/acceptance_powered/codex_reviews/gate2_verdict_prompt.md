You are a skeptical senior engineer doing an ADVERSARIAL review of a CORRECTED verdict about
to be recorded. A previous investigator (with a GATE-1 correction already applied) refreshed
a speculative-decode speedup model and is about to write up the finding. Stop them if the
verdict over- or under-claims. Re-derive from data; run read-only checks. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research . Python env: issue468/dspark_train/.venv/bin/python.

READ FIRST:
- issue468/summaries/spec_speedup_model.md  (TOP: "Lead 03 powered refresh" section — the corrected verdict)
- issue468/artifacts/spec_speedup_model/model_inputs.json  (sliding prefix_hist + lead03_cyclejump_realistic)
- issue468/artifacts/acceptance_powered/combined300/{aggregate.json,cyclejump.json}  (300-prompt powered data)
- issue468/artifacts/acceptance_powered/stage2_torch_measure/cyclejump.json  (240-prompt, for consistency)
- issue468/pending/lead_03_acceptance_statistical_power.md  (worklog: GATE-1 cycle-jump correction history)

RECAP: Lead 03 powered the acceptance measurement (300 prompts: Stage 2's 240 dolly/codealpaca/jsonex
+ 60 new; 128-tok temp=0) via a torch/MPS port precision-gated at 100% draft-token agreement vs the
numpy oracle. GATE-1 caught that the investigator's initial "K=4 break-even SIGNED POSITIVE" headline
was an OVERCLAIM based on the SLIDING (position-uniform) estimator; the correct CYCLE-JUMP (per-cycle,
real speculative trajectory) is lower. The investigator corrected it and refreshed the model.

THE INVESTIGATOR'S CORRECTED VERDICT (verify, do not trust):
1. SLIDING (the model's Q1/Q2/Q3 currency, position-uniform): E[a|4]=2.337, S(4)=0.382 ->
   K=4 speedup +1.2% (above baseline).
2. CYCLE-JUMP (realistic trajectory, per-cycle): E[a|4]=2.198, S(4)=0.340 -> K=4 speedup
   0.982x (-1.8%, BELOW baseline).
3. The sliding estimator OVER-estimates by ~3pp (uniformly samples positions, under-weights
   post-rejection correction cycles).
4. CORPUS-DEPENDENT: per-source cycle-jump speedup jsonex +2.8%, codealpaca -2.2%, dolly -5.6%;
   the old 10 code/synthesis exactness prompts were E[a|4]=2.175 (harder than all 3 Stage-2 families).
5. Headline: "K=4 is at/slightly-below break-even (-1.8%) under the realistic cycle-jump trajectory,
   corpus-mix-dependent; beating baseline is NOT achieved on this corpus mix. The prior -0.9% was
   sliding-based (optimistic) + the harder 10-prompt corpus."

PREMISES / SCOPE (exogenous; assess how the verdict swings if a premise fails, don't call them bugs):
- P1: the torch/MPS port at 100% draft-token agreement (10-15 prompts) is a valid stand-in for the
  numpy oracle across all 300 (the absolute scale the model feeds on).
- P2: the cycle-jump simulation (next_step += min(prefix,K)+1, from stored drafts+targets) is the
  correct model of real speculative-decode per-cycle acceptance.
- P3: the 300-prompt dolly/codealpaca/jsonex corpus is representative enough of deployment to draw a
  verdict (vs the old code/synthesis exactness corpus).

YOUR MANDATE (prioritize runnable checks):
1. IS THE CORRECTED VERDICT HONEST? Re-derive the cycle-jump K=4 (0.982x) AND the sliding K=4 (+1.2%)
   from combined300/cyclejump.json + the per-prompt checkpoints. Is the -1.8% vs +1.2% framing fair,
   or is one of them still misleading? Is "below baseline" the right read of 0.982x given the CI?
2. THE TRAJECTORY MODEL: is the cycle-jump simulation sound? Does "next_step += min(prefix,K)+1"
   correctly model real speculative cycles (accepted drafts + 1 bonus/correction)? Could a different
   trajectory model (e.g. tree/branching, or the drafter's own state pollution post-rejection, which
   this DOESN'T capture) swing the verdict? Is the scope caveat (anchor-token difficulty, not
   drafter-state pollution) honestly stated?
3. STATISTICAL SOUNDNESS: the cycle-jump pooled E[a|4]=2.198 vs break-even 2.203 — is "at/below
   break-even" justified? What's the cycle-jump CI (pooled)? Is the per-prompt-mean CI (2.22-2.34)
   being conflated with the pooled (model-currency) value? Is "below baseline" over-stated for a
   0.982x that's within noise of 1.0?
4. CORPUS CAVEAT: is the dolly/codealpaca/jsonex mix representative? The old code/synthesis was
   harder (2.175). Does the verdict appropriately hedge on corpus, or does it over-generalize from
   one mix? Should the headline be stratified by source rather than pooled?
5. IMPLICATION: does the corrected verdict properly update the Lead 01 anchor-reuse finding (reuse
   doesn't collapse acceptance, but the realistic K=4 edge is ~0.98x) and scope the path forward
   (lead 05 verifier; corpus diversity; the model's sliding/cycle-jump dual reporting)?

CONSTRAINTS: read-only; don't load the 81 GB expert npz or run ds4. The cycle-jump re-derivation from
the per_prompt drafts + the store targets is the heaviest thing you should run.

OUTPUT (terse, evidence-based):
## Verdict per claim (sound / questionable / likely-wrong + 1-line why)
## Is the corrected "K=4 ~0.98x, below baseline, corpus-dependent" honest? (definite read)
## Honest cycle-jump K=4 picture (pooled value + CI; per-source; vs sliding)
## What's still unaddressed / could swing the verdict (trajectory model, corpus, drafter-state)
## Final read: RECORD the corrected verdict as-is / with edits / DO NOT RECORD (which + why)
