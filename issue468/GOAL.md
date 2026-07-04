# Issue 468 Goal

## Research goal

Prove or disprove that a DSpark-style speculative path can deliver a **material local decode speedup** on the `ds4` engine architecture.

This is a research goal, not a product commitment. The first objective is not full DSpark parity with the paper. The first objective is to determine whether `ds4` can realize a real local speedup from:

- a parallel draft pass,
- a lightweight sequential correction head,
- and eventually a scheduled verification prefix.

The output of this work should be a decision:

- proceed with DSpark integration,
- narrow the scope,
- or stop because `ds4`'s architecture does not admit a worthwhile speedup.

## Success criteria

Primary gate:

- demonstrate `>= 20%` greedy decode throughput improvement versus baseline `ds4` target-only decoding on at least one realistic local machine/backend setup, with exact greedy output preservation.

Secondary gates:

- beat or clearly match the current `--mtp` speculative path on the same workload,
- preserve exact greedy output stream relative to target-only decode,
- avoid pathological memory overhead or replay overhead,
- show a plausible path to server-side gains after local single-request gains are proven.

## Working hypothesis

DSpark can only beat baseline locally if:

`(target step to get anchor + dspark draft pass + sequential head + verification) / accepted_tokens`

is materially lower than:

`plain target decode cost per emitted token`

for at least one meaningful prompt class on at least one intended local machine.

The main risk is that `ds4`'s verifier and state-management overhead erase the benefit of longer accepted prefixes.

## Non-goals for the first research cycle

- full DSpark training pipeline
- full serving scheduler integrated into `ds4-server`
- RNN head support
- support for every backend from day one
- polished CLI or stable public flags

## Active dossier intent

Within this branch, `issue468/` should keep a compact, trustworthy version of this research goal:

- one canonical current status,
- compact summaries of accepted findings and false leads,
- inventories of instrumentation and retained tools,
- and only the artifacts needed to support conclusions or reproduction.

## Source

This goal is distilled from `PLAN.md` in the source branch and should stay aligned with that plan's core question and success gates.
