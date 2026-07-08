You are a skeptical senior engineer doing an ADVERSARIAL independent review. A previous
investigator (another agent) built and ran an offline falsifier and reached a SURPRISING
result you must challenge. Re-derive every claim from the code; run read-only code to verify
where you can. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research . Python env:
/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/.venv/bin/python (numpy 2.5.1).
Relevant paths (all under issue468/):
- Lead/spec:        archive/leads/lead_01_anchor_reuse_falsifier.md
                   summaries/spec_speedup_model.md  (the anchor-reuse premise + the original adversarial review that proposed this falsifier)
- Implementation:   dspark_oracle/measure_acceptance_bundle.py  (added --reuse mode)
                   run_anchor_reuse_falsifier.py               (fidelity gate + reuse measurement + paired stats)
- Data:             artifacts/exactness_small_bundles/<prompt>__t0p0/oracle/main_hidden_pos{N}.npy
                   artifacts/exactness_small_bundles/<prompt>__t0p0/{bundle_manifest.json,target_selected_tokens.json,oracle/acceptance_summary.json}
- Result:           artifacts/anchor_reuse_falsifier/falsifier_result.json  (aggregate + per_prompt + units)
                   artifacts/anchor_reuse_falsifier/per_prompt.csv , units.csv

READ FIRST: issue468/archive/leads/lead_01_anchor_reuse_falsifier.md , then
issue468/dspark_oracle/measure_acceptance_bundle.py (esp. measure_bundle, the --reuse path,
and the docstring's KV-window modeling note), then issue468/run_anchor_reuse_falsifier.py.

RECAP: The speedup model's optimistic edge (K=4 at -0.9% vs the shipped verifier's -18.9%)
hinges on "anchor reuse" -- the verify forward's correction token anchoring the next draft
without a fresh 26 ms decode. The load-bearing doubt: the correction token was never
processed by the target, so its own hidden doesn't exist; reuse must draft from the
last-accepted-position hidden (which the verify forward did produce) with the correction
token entering only as embedding. The investigator built an offline test: along the greedy
spine the correction token == the greedy target token, so both the "true" hidden at pos and
the "stale" hidden at pos-1 are on disk. The falsifier swaps the drafter's per-step hidden
source from main_hidden[pos] to main_hidden[pos-1] and re-measures acceptance.

THE INVESTIGATOR'S RESULT (temp=0, 10 prompts, 80 paired (prompt,step) units):
- Fidelity gate PASSED: fresh baseline (reuse=False) reproduces the retained
  acceptance_summary.json draft tokens BIT-FOR-BIT across all 10 cells.
- E[a|5block]: baseline 2.3375 -> reuse 2.3625 (delta +0.025); paired bootstrap CI95 [-0.325,+0.375], p_two=0.871.
- p=1 first-token match: 0.8125 -> 0.9000 (delta +0.0875); McNemar discordants b=1 (base+/reuse-), c=8 (base-/reuse+), exact p=0.039 (significant IN FAVOR OF reuse).
- VERDICT: SURVIVES (delta >= 0).

THE INVESTIGATOR'S CLAIMS (verify, do not trust):
1. main_hidden[pos] is the target's hidden at position pos; target_tokens[k] sits at
   position pos0+k (pos0=prompt_tokens). At step t (pos=pos0+t) baseline loads
   main_hidden[pos] (the anchor's own position) and reuse loads main_hidden[pos-1]
   (the "last accepted position"); anchor token stays target_tokens[t] (the greedy
   correction token).
2. main_hidden enters the draft ONLY via the per-step KV-window entry win_kv[s][step] =
   mkv(main_x); the residual stream is seeded from embeddings. So swapping the hidden
   source is the entire, complete change. (See measure_bundle loop + dspark_attn.)
3. The "consistent-lag" KV-window model (reuse for ALL steps, so win_kv is a one-position
   lag of baseline; init mh0 at pos0 unchanged) faithfully simulates a real anchor-reusing
   verifier's drafter input. The accumulated-drift alternative was noted but not run.
4. The fidelity gate (baseline reproduces retained summaries bit-for-bit) validates the
   harness mechanics, so the reuse delta is trustworthy.
5. The SURVIVES verdict (reuse slightly BETTER, not worse) is a real signal: anchor reuse
   is realizable in principle, so the model's optimistic -0.9% edge is legitimate and the
   lead-05 verifier-engineering path is unblocked.

PREMISES / SCOPE (exogenous -- do NOT flag these as bugs; the investigator was instructed to
assume them. Instead assess (a) realizability and (b) how the conclusions swing if a premise
fails. Phrase as "under the premise, X; if the premise fails, Y" -- NOT "the model is wrong"):
- P1: The verify forward produces a usable anchor hidden at the rejection point / last
  accepted position. (This falsifier IS the realizability test of the drafter-acceptance half
  of that premise; codex should assess whether the offline substitution faithfully represents
  what a real verifier would feed the drafter, and how the verdict swings if it does not.)
- P2: The drafter architecture and the IQ2XXS target are fixed.
- P3: The two-tier decision rule (SURVIVES if d(E[a|5block])>=0; COLLAPSES if d<0 and
  bootstrap CI excludes 0; else MARGINAL) is user-specified.

YOUR MANDATE -- challenge assumptions, find new avenues. Decisive claims will be INDEPENDENTLY
re-verified by the dispatcher before acceptance, so prioritize claims you can back with a
runnable read-only check or a precise citation. Not exhaustive:
1. END-TO-END EXISTENCE / DOES THIS TEST THE RIGHT THING: does the offline substitution
   actually exercise the real reuse cost? The original review's doubt was that the verify
   forward yields the correction token's LOGITS but has not run the target on it, so no valid
   KV/hidden exists at the correction position. The investigator's test swaps the drafter's
   SEED hidden to pos-1 but (a) does pos-1 hidden actually equal what a real verify forward
   would produce at the last-accepted position (batched-prefill vs one-at-a-time decode
   hidden equivalence under IQ2XXS)? (b) does the drafter need the correction token's OWN
   hidden anywhere the test still silently provides? Trace exactly what the drafter consumes.
2. MAPPING SEMANTICS (the highest-risk class here -- a sibling investigation was bitten by an
   offset-vs-absolute map bug): is main_hidden[pos] PRE-token (input to the layer processing
   token pos) or POST-token (output after processing token pos)? Verify ALGEBRAICALLY against
   the capture path (dspark_oracle/build_main_hidden_from_captures.py) and the ds4 capture
   convention, not from names. If main_hidden[pos] is post-token-at-pos, then pos-1 is the
   hidden after the token BEFORE the anchor -- is that "the last accepted position" or off by
   one? Re-derive target_tokens[k] -> position mapping from the manifest + the harness loop.
3. THE SURPRISE: reuse slightly BEATS baseline. Is this (a) real (e.g. the drafter was trained
   on verify-produced / stale-by-design hiddens, so fresh decode hiddens are slightly OOD and
   the baseline measurement has been mildly pessimistic all along), or (b) an artifact of the
   substitution NOT capturing the real reuse cost (e.g. the test still gives the drafter a
   perfectly valid, fully-decoded hidden from a real prefix, which a real mid-rejection
   verifier would NOT have)? Distinguish "stale hidden from a valid prefix" (what the test
   feeds) from "hidden produced by a verify forward over a partially-rejected suffix" (what a
   real verifier feeds). Quantify if you can.
4. KV-WINDOW MODELING: is consistent-lag the right model, or does a real reusing verifier give
   the drafter true-KV-for-prior-cycles + stale-only-for-current? Does the choice swing the
   verdict? (Note win_kv accumulates across steps; reuse shifts it to a lag. Is the duplicate
   at pos0 between init and step-1 a problem?)
5. STATISTICS: n=80 paired units, 10 prompts x ~8 steps. Is the McNemar on p=1 (b=1,c=8) sound
   given possible within-prompt correlation (steps are NOT independent)? Should units be
   per-prompt-averaged first? Does the bootstrap CI properly account for the paired/clustering
   structure? Is "SURVIVES" with delta=+0.025 and CI [-0.325,+0.375] being over-read?
6. THE CONCLUSION ITSELF: is "SURVIVES -> optimistic edge legitimate -> lead 05 unblocked"
   justified from THIS evidence, or does it overclaim? What is the minimum additional check
   that would make the verdict load-bearing (e.g. an accumulated-drift variant, a
   verify-produced-hidden capture, a per-prompt-clustered stat)?

CONSTRAINTS: do not modify tracked files. You may run read-only commands and inline scripts
(python -c, rg, jq, numpy on the .npy arrays) to verify claims. Loading the 87 GB target +
11 GB drafter to re-run the harness is slow (~70 s); prefer verifying against the retained
arrays and the result json, and only re-run if decisive.

OUTPUT (terse, evidence-based):
## Verdict per claim (sound / questionable / likely-wrong + 1-line why)
## Premise sensitivity (per premise: does the headline survive if it fails?)
## Does the test exercise the real reuse cost? (the central question -- give a definite read)
## New experiments to try (ranked; each: what to run, expected signal, effort)
## Leading hypothesis after re-examination + the one decisive test for it
