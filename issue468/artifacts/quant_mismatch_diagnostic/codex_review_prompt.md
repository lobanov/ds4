You are a skeptical senior engineer doing an ADVERSARIAL independent review. A
previous investigator (another agent) reached a conclusion you must challenge.
Re-derive every claim from the code; run code to verify where you can. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research. Python env:
`issue468/dspark_oracle/.venv` (numpy 2.5.1). Relevant paths:
- canonical report: `issue468/summaries/quant_mismatch_diagnostic.md`
- harness (code under review): `issue468/run_stage0_quant_mismatch.py`
- artifacts: `issue468/artifacts/quant_mismatch_diagnostic/{summary.json,per_position.csv,per_position.json}`
- retained measurement it must agree with: `issue468/dspark_oracle/measure_acceptance_bundle.py`
  and per-bundle `issue468/artifacts/exactness_small_bundles/*__t0p0/oracle/acceptance_summary.json`
- the speedup model supplying the numeric anchors: `issue468/summaries/spec_speedup_model.md`
- the drafter forward it reuses: `issue468/dspark_oracle/forward.py` (esp. `forward_head`,
  the markov-head reconstruction)

RECAP: The investigator is testing the hypothesis that the DSpark drafter's poor
acceptance vs the IQ2XXS target is partly because the drafter was distilled on the
FP teacher but is served against a Q2 target (quantization mismatch), which could
motivate a drafter fine-tune. Stage 0 is an OFFLINE diagnostic: re-run the drafter
over 10 temp=0 exactness bundles, capture the full per-position decision score
(base logits + markov bias), and measure the rank of the Q2 target token in the
drafter's distribution on every miss. Claimed conclusion: the error is
"overwhelmingly shallow/perturbative" (median target-rank on p=1 misses = 1.0;
100% within top-10; top-2 coverage 0.8125->0.9125), so Stage 0 "clears" and Stage 1
(raise tap-layer precision, re-measure acceptance) should proceed.

THE INVESTIGATOR'S CLAIMS (verify, do not trust):
- C1 FIDELITY: the re-run forward reproduces the retained `acceptance_summary.json`
  draft tokens at all 80 steps (0 mismatches), so the captured logits are trustworthy.
- C2 MARKOV EXACT: the full decision score reconstructed as
  `base_logits[i] + markov_w2 @ markov_w1[prev_token]` has argmax exactly equal to
  the drafter's rollout token at all 400 positions.
- C3 ANCHORS: p=1 match-rate = 0.8125 == the speedup model's S(1); the k=1 greedy-spine
  counterfactual E[a|5block] = 2.3375 == the model's E[a|5]=2.338; both validate alignment.
- C4 RANK COMPUTATION: `rank_target = count(full_score > full_score[target])` correctly
  gives the 0-indexed rank of the target token; median on p=1 misses is 1.0.
- C5 ALIGNMENT: draft position p (1..5) at measure-step `step` predicts
  `target_tokens[step+p]`, whose Q2 distribution is `target_topk.json` step `step+p`.
- C6 INTERPRETATION: shallow ranks (target is the drafter's rank-2 in the median miss)
  => the error is "perturbative / quant-flip-like / fine-tuning-recoverable", NOT deep
  drafter capacity failure => clears the gate to Stage 1.
- C7 COUNTERFACTUAL: the greedy-spine top-k counterfactual is a valid ACCEPTANCE
  headroom ceiling (k=2 -> E[a|5block]=2.875) and is correctly caveated as NOT realized
  throughput (trees are verify-cost-limited per spec_speedup_model.md Q3).

PREMISES / SCOPE (exogenous — do NOT flag as bugs; assess (a) realizability and
(b) how conclusions swing if a premise fails. Phrase as "under the premise, X; if
the premise fails, Y"):
- P1 The drafter was distilled on the FP teacher (this is the hypothesis, not
  independently verified by the investigator).
- P2 IQ2XXS is the served target; temp=0 exactness bundles are the right data to
  isolate argmax mismatch (no sampling noise).
- P3 The speedup model's retained acceptance numbers (S(1..5)=0.8125,0.65,0.425,
  0.2875,0.1625; E[a|5]=2.338) are valid inputs for the anchors.

YOUR MANDATE — challenge assumptions, find new avenues. Your decisive claims will
be INDEPENDENTLY re-verified by the dispatcher before acceptance, so prioritize
claims you can back with a runnable check or a precise citation. Not exhaustive:
1. END-TO-END EXISTENCE / SHARED-BUG: C1 crosschecks the re-run against the retained
   `measure_acceptance_bundle.py` output — but BOTH share the same drafter forward,
   embed/lm_head load, and markov head. A shared bug (e.g. a wrong tensor name, a
   non-strict load, a transpose) would pass the crosscheck and still produce wrong
   ranks. Is there ANY independent ground truth the ranks are checked against? (The
   drafter's own argmax matching the target's argmax 81% of the time is one weak
   external signal — sanity-check it against any published/expected drafter accuracy.)
2. MAPPING / INDEX SEMANTICS: verify C5 alignment algebraically. Is `target_topk.json`
   step `s` really `target_selected_tokens[s]` at position `pos0+s`? Check the
   `selected.id` chain. Is draft position p=1 really the FIRST token after the anchor,
   i.e. `target_tokens[step+1]`? Trace `measure_acceptance_bundle.measure_bundle`'s
   `target = target_tokens[step+1:step+1+BLOCK]` and confirm `run_stage0` matches.
3. RANK COMPUTATION CORRECTNESS (C4): `count(score > score[target])` — does this
   handle ties correctly? Is `target` ever out of range? Is `full_score` over the
   full 129280 vocab or a truncated set? Re-derive one miss by hand from
   `per_position.csv` + the stored logits if accessible.
4. THE CLEARING DECISION (C6): "shallow rank => fine-tuning-recoverable" is an
   INFERENCE. Challenge it three ways: (a) could the rank-2 misses be a SYSTEMATIC
   drafter bias unrelated to quant (e.g. markov head prior, training-data mix) that
   fine-tuning on Q2 would NOT fix? (b) does "drafter's pick is in Q2's top-3 80% of
   the time" actually support the quant-flip narrative, or does the large median Q2
   gap (2.15 nat) undercut it (Q2 is confident, so FP likely agrees with Q2, so the
   drafter is just wrong, not quant-flipped)? (c) is n=15 misses at p=1 enough to
   support "median 1.0, 100% within top-10" as a robust gate signal? Compute a
   binomial/bootstrap CI if useful.
5. STATISTICAL SOUNDNESS: the corpus is 10 prompts, 80 steps, temp=0 only. Are the
   headline fractions (73% rank<=2, etc.) trustworthy at this n? Is temp=0
   representative of deployment (the speedup model noted temp 0.5/1.0 drops E[a|4])?
6. COUNTERFACTUAL SEMANTICS (C7): "greedy-spine top-k" accepts position p iff the
   target is in the drafter's top-k ALONG THE DRAFTER'S OWN (possibly diverged)
   rollout. Does this over- or under-state what a real fine-tune or tree achieves?
   Is the k=1 row truly identical to current acceptance (it must be, by construction)?
   Verify k=1 == current match distribution from the csv.
7. QUANTIFY, DON'T ASSERT: where the report claims a magnitude (e.g. "+10 pp at k=2"),
   confirm it from the csv.

CONSTRAINTS: do not modify tracked files. You may run read-only commands and inline
scripts (python -c, rg, jq) to verify claims. The venv numpy is available. Do NOT
re-run the full 87GB-model harness (it takes ~1 min); verify from the persisted
csv/json and code inspection.

OUTPUT (terse, evidence-based):
## Verdict per claim (C1..C7: sound / questionable / likely-wrong + 1-line why)
## Premise sensitivity (P1..P3: does the headline survive if it fails?)
## New experiments to try (ranked; each: what to run, expected signal, effort)
## Leading hypothesis after re-examination + the one decisive test for it
## Bottom line: does "Stage 0 clears -> proceed to Stage 1" survive your review?
