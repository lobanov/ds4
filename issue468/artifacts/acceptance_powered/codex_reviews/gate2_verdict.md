## Verdict Per Claim

1. **Sliding +1.2%: sound, but diagnostic only.** Rebuilt hist exactly: E[a|4]=2.3374, S4=0.3821, speed=1.0121x.
2. **Cycle-jump 0.982x: sound.** Recomputed from drafts+targets: 10,982 cycles, E[a|4]=2.1976, S4=0.3402, speed=0.98219x.
3. **Sliding overestimates: sound if “~3pp” means speedup gap.** Sliding +1.21% vs cycle-jump -1.78% = 2.99pp. Mechanism is credible.
4. **Corpus caveat: sound, with numeric edits.** Combined300 per-source cycle-jump: jsonex +2.3%, codealpaca -1.9%, dolly -5.6%. The +2.8/-2.2 numbers are 240-prompt/stale.
5. **Headline: mostly honest, but edit break-even wording.** “Below baseline” is statistically supported for this corpus/model; “E=2.198 vs break-even 2.203” is wrong/stale.

## Is The Corrected Verdict Honest?

**Yes, with edits.** The corrected “K=4 ~0.98x, below baseline, corpus-dependent” is the right read. Do not record any statement implying the cycle-jump E gap is only `2.198 vs 2.203`; with S4=0.340, the dynamic break-even E is **2.256**, deficit **0.058 accepted drafts/cycle**.

## Honest Cycle-Jump K=4 Picture

Pooled cycle-jump: **E[a|4]=2.1976, S4=0.3402, speed=0.9822x (-1.78%)**.

Prompt-cluster bootstrap, pooled model currency:
- E[a|4] CI: **[2.1495, 2.2459]**
- speed CI: **[0.9712, 0.9931]**
- delta vs dynamic break-even E CI: **[-0.0935, -0.0226]**
- bootstrap P(speed < 1): **0.9989**

Per-source:
- jsonex: **1.0227x**
- codealpaca: **0.9805x**
- dolly: **0.9445x**

Per-prompt-mean CI **[2.228, 2.333]** is not the pooled cycle-weighted model currency. Do not use it to imply positive K=4.

## Still Unaddressed

Cycle-jump transition `s += accepted + 1` is correct for linear speculative decode; see [run_lead03_cyclejump.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_cyclejump.py:42). It does not cover tree/branching, adaptive policies, residual verifier overhead, or drafter-state pollution after rejected drafts.

The scope caveat exists in the trajectory script: anchor-token difficulty, **not** drafter-state pollution; see [run_lead03_trajectory.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_trajectory.py:109). Put that caveat in the recorded verdict too.

Downstream docs still have stale -0.9/0.99 language, e.g. [lead_06_verifier_engineering.md](/Users/lobanov/Projects/ds4-dspark-research/issue468/pending/lead_06_verifier_engineering.md:17). Sweep those if recording dossier-wide.

## Final Read

**RECORD with edits.** Keep the core verdict: realistic K=4 is **0.982x, signed below baseline on this corpus, corpus-mix-dependent**. Edit: remove stale `2.203` break-even framing, label sliding as optimistic diagnostic only, use combined300 per-source numbers, and explicitly scope out drafter-state pollution / Lead 06 verifier reality.