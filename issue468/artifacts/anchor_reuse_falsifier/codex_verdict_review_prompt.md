You are a skeptical senior engineer doing an ADVERSARIAL independent review of a VERDICT.
A previous investigator ran an offline anchor-reuse falsifier and is about to RECORD a conclusion.
Your job is to stop them if the verdict overclaims or the stats don't support it. Re-derive from the
data; run read-only checks where you can. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research . Python env:
/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/.venv/bin/python (numpy 2.5.1).
Relevant paths (all under issue468/):
- Result:           artifacts/anchor_reuse_falsifier/falsifier_result.json  (temps->{t0p0,t0p5,t1p0} -> reuse_modes->{lag,backfill}->aggregate + per_prompt + units)
                   artifacts/anchor_reuse_falsifier/per_prompt_*.csv , units_*.csv
- GATE 1 review:   artifacts/anchor_reuse_falsifier/codex_setup_review.md  (already addressed; do not re-litigate)
- Spec/model:      summaries/spec_speedup_model.md  (the anchor-reuse premise + the -0.9%/-18.9% band)
- Lead:            archive/leads/lead_01_anchor_reuse_falsifier.md

READ FIRST: issue468/artifacts/anchor_reuse_falsifier/falsifier_result.json , then
issue468/summaries/spec_speedup_model.md (the "Adversarial review" section that proposed this falsifier).

RECAP: The falsifier tests whether DSpark drafter acceptance survives drafting from the
one-position-stale target hidden (last-accepted-position) instead of the true anchor hidden --
the realizability test for anchor reuse (the model's optimistic -0.9% at K=4 vs the shipped
verifier's -18.9% hinge on it). Method: offline, reusing retained greedy-spine captures; along
the spine the correction token == greedy target token, so both main_hidden[pos] and
main_hidden[pos-1] are on disk. The harness swaps the drafter's per-step hidden source. Two
KV-window models tested: "lag" (consistent lag, = the lead's specified method) and "backfill"
(current slot stale, prior slots true; added per GATE 1). 10 prompts x ~8 steps = 80 paired units
per (temp, mode); 3 temps (t0p0 greedy, t0p5/t1p0 sampled-stream). Decision rule (user-specified,
two-tier): SURVIVES if d(E[a|5block])>=0; COLLAPSES if d<0 and per-step bootstrap CI95 excludes 0;
else MARGINAL.

THE INVESTIGATOR'S RESULT (all 6 cells):
  t0p0  lag      d_E[a|5]=+0.025  CI95[-0.325,+0.375]  clustered[-0.15,+0.20]  d_p1=+0.0875  McN_p=0.039  SURVIVES
  t0p0  backfill d_E[a|5]=-0.075  CI95[-0.400,+0.250]  clustered[-0.21,+0.10]  d_p1=+0.075  McN_p=0.109  MARGINAL
  t0p5  lag      d_E[a|5]=+0.138  CI95[-0.200,+0.487]  clustered[-0.05,+0.30]  d_p1=+0.1125 McN_p=0.012  SURVIVES
  t0p5  backfill d_E[a|5]=+0.113  CI95[-0.225,+0.450]  clustered[-0.09,+0.34]  d_p1=+0.1125 McN_p=0.023  SURVIVES
  t1p0  lag      d_E[a|5]=+0.113  CI95[-0.237,+0.463]  clustered[-0.13,+0.33]  d_p1=+0.0625 McN_p=0.227  SURVIVES
  t1p0  backfill d_E[a|5]=+0.113  CI95[-0.200,+0.425]  clustered[-0.04,+0.28]  d_p1=+0.075  McN_p=0.146  SURVIVES
  fidelity_gate_passed_all_temps: true (fresh baseline reproduces retained summaries bit-for-bit).
  p=1 delta >= 0 in ALL 6 cells (McNemar discordants c >> b everywhere: c in [8..11], b in [1..3]).

THE INVESTIGATOR'S VERDICT (about to be recorded):
- "Anchor reuse does NOT collapse drafter acceptance in any sampling regime or KV-window model.
  5/6 cells SURVIVE; the lone MARGINAL (t0p0/backfill, -0.075) is within noise and is NOT
  replicated at t0p5/t1p0 (both backfill SURVIVE). p=1 robustly non-negative in all 6 cells."
- Implication: "The anchor-reuse accounting in spec_speedup_model is NOT invalidated on
  acceptance grounds. Lead 05 (verifier engineering) is unblocked on the acceptance axis -- but
  must still prove the full cycle economics (verify-produced-hidden equivalence under IQ2XXS
  prefill-vs-decode, residual cycle overhead, actual timing), which this offline test does NOT
  cover. The optimistic -0.9% edge at K=4 remains the optimistic edge of the band, conditional on
  those verifier-side factors."

THE INVESTIGATOR'S CLAIMS (verify, do not trust):
1. "No collapse" is the correct framing: every CI straddles 0, so reuse is statistically
   indistinguishable from baseline (never significantly worse). The pessimistic concern (drafter
   cannot draft from stale hidden) is refuted on the representational axis.
2. p=1 robustly non-negative (c>>b in all 6 cells) is a real, consistent signal that the first
   draft token does not degrade under reuse.
3. The t0p0/backfill MARGINAL is isolated noise, not a systematic cost, because t0p5/t1p0 backfill
   SURVIVE.
4. The implication is properly scoped: it claims only that acceptance is preserved (necessary
   condition), NOT that the full -0.9% optimistic edge is realizable; verifier-side factors are
   explicitly deferred to lead 05.

PREMISES / SCOPE (exogenous -- do NOT flag as bugs; assess how the conclusion swings):
- P1: The falsifier covers only the drafter-acceptance representational condition (does the
  drafter tolerate stale-from-valid-prefix hidden?), NOT end-to-end verifier cycle economics.
- P2: The two-tier decision rule is user-specified.
- P3: main_hidden[pos-1] (post-token, last-accepted position) is a valid proxy for a real verify
  forward's hidden at the last accepted position (GATE 1 accepted this as a scope boundary; the
  verify-batch vs fresh-decode hidden difference under IQ2XXS is untested).

YOUR MANDATE -- this is a VERDICT review, so the bar is "would a careful reader accept this
conclusion as honestly supported?" Prioritize decisive, runnable checks:
1. STATISTICAL SOUNDNESS: are the bootstrap CIs and McNemar tests computed correctly? Re-derive
   at least one from units_*.csv. Is "every CI straddles 0 => no collapse" sound, or does
   straddling-0 just mean UNDERPOWERED to detect a real effect (could a real -0.075 cost be
   hiding in the noise)? What is the minimum detectable effect at n=80 / n_clusters=10? Is the
   per-prompt-clustered CI the one that should govern (steps are NOT independent within a prompt)?
2. p=1 SIGNAL: is "p=1 robustly non-negative" being overweighted? p=1 is just the FIRST token of
   the block; E[a|5block] (the metric the decision rule uses) is what matters for speedup. Is the
   p=1 improvement possibly an artifact (e.g. the stale hidden shifts the drafter toward more
   common/likely first tokens that happen to match)? Does p=1 improving while E[a|5block] is flat
   imply later positions are WORSE under reuse?
3. THE "NO COLLAPSE" FRAMING: the original concern (spec_speedup_model Adversarial review) was
   realizability of anchor REUSE, framed as a binary risk. Does "no statistically-significant
   collapse at n=80" actually refute that risk, or does it just say "we're underpowered to see a
   modest cost"? Is the verdict "SURVIVES" vs the more honest "indeterminate / underpowered"?
4. THE t0p0/backfill MARGINAL: is dismissing it as "isolated noise, not replicated at t0p5/t1p0"
   justified? Note t0p5/t1p0 are SAMPLED-STREAM (different semantics -- the "correction token" is
   a sampled token, not greedy argmax); is comparing across temps even apples-to-apples for this
   falsifier? Could the sampled-stream cells be LESS sensitive (and thus hide a real greedy-regime
   cost)?
5. IMPLICATION SCOPE: does "lead 05 unblocked on the acceptance axis" overclaim given (a) the
   verify-produced-hidden proxy is untested, (b) the corpus is 10 short prompts (~67-165 tok)?
   What is the smallest additional check that would make this verdict load-bearing for the
   model's -0.9% claim?
6. COMPLETENESS: is any cost term relevant to acceptance being silently set to zero? Is the
   "consistent-lag" vs "backfill" KV choice the only modeling degree of freedom, or are there
   others (e.g. does the drafter's own draft-window KV, separate from win_kv, also need to be
   re-examined under reuse)?

CONSTRAINTS: do not modify tracked files. Read-only commands and inline scripts only.

OUTPUT (terse, evidence-based):
## Verdict per claim (sound / questionable / likely-wrong + 1-line why)
## Is "SURVIVES / no collapse" honestly supported, or should it be "indeterminate/underpowered"?
## Minimum detectable effect at this n (quantify) -- could a real modest cost be hiding?
## Smallest check that would make the verdict load-bearing for the -0.9% claim
## Final read: RECORD the verdict as-is / RECORD with edits / DO NOT RECORD (which + why)
