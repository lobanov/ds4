# DSpark prefix-limited batched scheduler post-benchmark adversarial review

Date: 2026-07-12
Reviewer: `gpt-5.5` (`xhigh`)
Scope: current prefix-limited batched scheduled DSpark path after the first
throughput/profile comparison

## Verdict on the current evidence

The current prefix-limited batched scheduler is worth a **narrow follow-up**, not
a positive optimization claim yet.

- Semantics look credible: the implementation uses prefix-limited intra-block
  visibility and the strengthened schedule-parity test now checks draft ids,
  confidence logits, `verify_n`, accepted tokens, and accepted chunking.
- The benchmark evidence is mixed:
  - the short retained `ds4-spec-bench` sample is slightly **negative** on mean
    throughput
  - the retained 8k profile is **positive** on draft-side cost and slightly
    positive on end-to-end generation rate

## Safe claims

- The current prefix-limited path is still opt-in and passes stronger
  source-level schedule-parity coverage than the earlier rejected full-block
  attempt.
- On the retained 8k sched-only profile, batched scheduling lowers measured
  draft time while preserving measured `verify_n` / `verified` means.
- On the short 6-prompt retained sample, accepted chunk metrics are unchanged
  while throughput is slightly worse on average.

## Unsafe claims

- “Batched scheduling is faster.”
- “This optimization is validated.”
- “Acceptance economics are identical” without qualifying that the short sample
  only records chunk-level `accepted_mean`, not full per-cycle draft/verify
  details.
- “GPU migration is now proven to be the best next step.”

## Measurement weaknesses / confounders

- Harness mismatch:
  - `ds4-spec-bench` reports throughput from aggregate `decode_ms`
  - the long profile uses CLI `gen_tps` plus parsed DSpark timing lines
- Small samples:
  - 6 short prompts in the retained `ds4-spec-bench` sample
  - 3 long prompts in the retained profile
- The short JSONL artifacts do not yet record `schedule_batched`, `draft_ms`,
  `verify_n`, `verified`, rows computed, or confidence values directly.
- Batched scheduling can compute more rows than the retained serial scheduler
  even while `verify_n` stays fixed, so “same economics” is too broad.

## Highest-value next measurement step

Extend `ds4-spec-bench` into the unified retained DSpark benchmark substrate for
this question: one interleaved paired A/B JSONL harness across short, medium,
and 8k prompts that records, per run and ideally per cycle:

- `schedule_batched`
- rows computed / drafted
- `verify_n`
- `verified`
- `draft_ms`
- `verify_ms`
- `total_ms`
- confidence values or a stable hash
- output hash

## GPU path judgement

For DSpark-specific work, GPU migration of the prefix-limited full-block
microbatch still looks like the most coherent DSpark design direction. For the
overall runtime, it is not yet proven to be the best next optimization lever,
because verifier decode remains a major ceiling and the short prompt sample did
not show an end-to-end win from the CPU batched path.
