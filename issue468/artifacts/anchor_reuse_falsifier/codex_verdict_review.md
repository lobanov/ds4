## Verdict Per Claim

1. **Questionable**: CIs straddling 0 means “no significant harm detected,” not “acceptance preserved.” This is underpowered for modest losses.
2. **Sound narrowly, overweighted broadly**: p=1 is consistently non-negative, but it is only first-token acceptance; later positions offset it.
3. **Likely wrong**: t0p0/backfill is not dismissible as isolated noise: 6 prompts negative, 3 flat, 1 positive. t0p5/t1p0 are sampled-stream cells, not clean replications.
4. **Mostly sound but wording too strong**: The caveats are present, but “unblocked on acceptance axis” should mean “not blocked by this proxy falsifier,” not “closed.”

## SURVIVES / No Collapse?

**Supported only as “no large collapse observed under the specified proxy/rule.”**

Not supported as “acceptance is preserved” or “representational risk is refuted.” The labels are mechanically correct under the user rule, but this is not a non-inferiority test.

Re-derived example from `units_t0p0_backfill.csv`: `d_E[a|5]=-0.075`, per-step CI `[-0.400,+0.250]`, clustered CI `[-0.212,+0.113]`, McNemar `b=2,c=8,p=0.109`. Matches result.

p=1 is misleading for speedup: t0p0/backfill position deltas are `[+0.075,+0.0375,0,-0.100,-0.0875]`; all 6 cells have negative position-5 delta.

## Minimum Detectable Effect

Using observed variance:

- Per-step bootstrap scale: 95% half-width about `0.33-0.36`; 80% MDE about `0.46-0.50` accepted tokens per 5-block.
- Prompt-clustered scale, which should govern: 95% half-width about `0.20-0.28`; 80% MDE about `0.28-0.39`.

So yes, a real modest cost like `-0.05` to `-0.15` can easily be hiding. For the K=4 model edge, capped `E[a|4]` point estimates are non-negative, but greedy backfill still has clustered CI about `[-0.113,+0.175]`, far wider than the ~`0.03` accepted-token budget implied by the fragile `-0.9%` edge.

## Smallest Load-Bearing Check

Run a real K=4 reuse-verifier path, not the proxy: verify-produced hidden/KV under IQ2XXS, folded correction, actual cache transition, and measured cycle timing. Then do a prompt-clustered non-inferiority test with margin tied to the speedup budget, roughly `d_E[a|4] > -0.03` unless timing overhead reduces the margin further.

## Final Read

**RECORD with edits.**

Do not record as-is. Replace “does NOT collapse / refuted / acceptance preserved” with “no large collapse observed in this offline proxy; non-inferiority is not established.” Replace “lead 05 unblocked” with “lead 05 can proceed, but must still prove real verifier hidden equivalence, residual overhead, and a better-powered K=4 acceptance bound.”