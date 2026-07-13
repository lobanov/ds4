# Lead 09 — Agentic / tool-call decode regime (region-selective DSpark)

Date: 2026-07-13. Status: **pending (new, not yet executed).** Independent of
leads 05 / 07 / 08 for its offline phase; the cheapest decisive experiment can
run in parallel with all of them.

## Execution contract

Pursue this lead under the general research methodology in
`.pi/skills/research-lead/SKILL.md`:

- orient and verify all claimed assets before acting;
- lock the decision rule and estimator before trusting measurements;
- fidelity-gate any new acceptance path against retained oracle /
  powered-acceptance tooling;
- report prompt-clustered CIs and **per-region** (not just corpus-mean) results;
- propagate the canonical result through `summaries/` and `STATUS.md`;
- archive the lead correctly on completion.

Treat this as decision-grade work with two adversarial review gates following
`.pi/skills/adversarial-codex-review/SKILL.md`:

- **setup gate:** after the trajectory-capture + per-region acceptance path
  exists, but before trusting measurements; and
- **verdict gate:** before recording the conclusion.

Each gate should use an independent `gpt-5.5` `xhigh` subagent review that
re-derives claims from code and artifacts. Any decisive subagent finding must
be independently verified before acceptance.

## Rationale

Every acceptance number in this dossier is measured on a corpus of **one-shot
general reasoning / generation**: `dolly`, `codealpaca`, `jsonex`, and the
`code` / `grounded` / `synthesis` × {4k, 8k, 16k} baseline prompts
(`prompts/baseline_corpus/`, `prompts/lead3_corpus/`). None of it reflects
**ds4's actual intended workload**, which is agentic AI-assisted software
engineering. In that regime a large fraction of decoded tokens are **structured
tool-call emission** — function names, argument keys, JSON/XML delimiters, enum
values, repetitive boilerplate inside the chat template's tool-call wrappers —
not free-form reasoning. Those tokens have a very different, much more
predictable distribution than the measured corpus.

Two existing facts make this lead more than speculation:

1. **The most structured source already wins.** On the 300-prompt powered
   corpus, per-source cycle-jump K=4 speedup is
   `jsonex +2.3% / codealpaca −1.9% / dolly −5.6%`
   (`summaries/acceptance_statistical_power.md`). `jsonex` — the only
   structured-output source and the closest proxy for tool-call emission — is
   the **only** cell above baseline. That is direct *prima facie* evidence that
   structural regularity raises drafter acceptance on this target/drafter.
2. **Selective enablement is cheap and greedy-exact in this runtime.** The
   Lead 06 / milestone-2 measurements show DSpark support-state pushes cost
   only ~0.6–1.5 ms/cycle, so toggling DSpark on/off at region boundaries is
   not obviously expensive, and plain decode in low-acceptance reasoning
   regions *avoids* paying draft+verify for tokens the drafter would mostly
   miss. On exactness: the **default `ds4 --dspark` path uses the exact
   sequential verifier and is greedy-exact** (milestone-1: 10/10 byte-identical
   on the retained corpus); the faster **batch verifier**
   (`verify_suffix_tops`) is sublinear but **not greedy-exact** — it flips
   ~0.64% of argmax tokens, though it is distribution-exact on temp>0
   (milestone-2). Lead 09 uses the **greedy / exact path**: the primary gate
   requires exact greedy output, and Phase 1 capture needs *exact* token
   traces — batch-verifier flips would corrupt the tool-call syntax
   measurement. Under the exact path, region-selective enablement (DSpark on
   in tool regions, plain decode elsewhere) preserves greedy-identical output
   **by construction**.

The specific doubt: does the retained Q4_K drafter produce **materially higher
acceptance on real agent tool-call regions** than on the measured corpus mean
(`E[a|4]=2.198`, `0.982×` baseline, dynamic break-even `2.256`), high enough
that **region-selective DSpark** over a realistic agent trajectory clears the
+20% primary gate — even though the always-on corpus-average path does not?

This is the one remaining route to the local primary gate that has **never been
measured** in this dossier, and unlike drafter quality (Lead 07) or kernel work
(Lead 08) it is testable with a cheap offline acceptance experiment before any
engineering investment.

Note: the Phase-1 per-region acceptance measurement is **regime-defining
regardless** of whether the eventual deployable path is region-selective
enablement or simply always-on DSpark on agent trajectories. Phase 1 decides
both; the selective-enable framing is the user's stated default carrier and is
the one modeled in Phase 2.

## Content of work

Phase 1 is the cheap decisive test; Phases 2 and 3 run only if it passes.

### Phase 1 — Offline per-region acceptance on real agent trajectories

The cheapest decisive test reuses two things that already exist in-tree: the
**engine's per-position layer-mean hidden capture** (the `DS4_METAL_GRAPH_DUMP`
/ `hc_ffn_post` path behind `--capture-dataset`, used for Stage 2), and the
**server's own per-token region classification** — during streaming decode
every emitted token is already tagged
`openai_stream_mode ∈ {THINKING, TEXT, TOOL, SUPPRESS}`, where `TOOL` is
precisely the inside of a DSML tool-call span
(`<｜tool▁calls▁begin｜> … <｜tool▁calls▁end｜>`, etc.). So region labels are
**free**; the only new work is persisting them next to token ids and hiddens
for a live agent request.

Pi already speaks to ds4-server as its `ds4` provider
(`http://127.0.0.1:8000/v1`, model `deepseek-v4-flash`, OpenAI-compatible,
tools enabled), so a Pi agent session is the natural trajectory source.

**Trajectory carrier — Pi (diverse tools) on a borrowed SWE-Bench Verified
instance in docker.** The carrier choice is load-bearing: the point is a
*representative* tool-call token distribution, so the agent must emit *diverse*
structured tool calls on a *real* task.
- The existing `~/Projects/swe-bench` rig is already wired to ds4-server
  (`.env` → `LLM_BASE_URL=http://127.0.0.1:8000/v1`) and **proven on ds4**
  (`runs/full-metal-20-40/report.txt` = 14/20, 70% resolved), with `mini-extra`
  v2.4.0, `swebench`, and ~60 cached eval images. **But its default agent is
  mini-swe-agent, which is bash/submit-only** — a single repeated tool, far too
  narrow a distribution for this lead.
- **Pi is the diverse-tool replacement.** Pi's built-in tools are
  `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`; `pi -p` runs the full
  agent loop non-interactively, and Pi documents a supported "Plain Docker"
  containerization pattern (run the whole `pi` process in a container).
- So the carrier is: **borrow one SWE-Bench eval container**
  (`ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>` — Ubuntu 22.04, apt-able,
  conda + the repo at `/testbed`, the bug's env), **install Node + Pi
  in-container**, point Pi's `ds4` provider at `host.docker.internal:8000`
  (`temperature=0`), feed it the instance's problem statement, and run
  `pi -p "<problem statement>" -t read,edit,write,bash,grep,find,ls` from
  `/testbed`. ds4-server stays host-side (the single ~87 GB heavy process);
  the container is light. Pi's diverse tool calls flow through ds4-server, which
  captures the per-token traces. **SWE-bench grading is optional bonus**
  (copy the patch back and run the harness) — Lead 09 needs the trajectory, not
  the score.
- **Feasibility risks to confirm in orientation:** linux/amd64 emulation on
  arm64 Mac slows container-side tool ops (LLM decode is host-side, unaffected);
  Node/Pi install in-container (npm network, or copy the pure-JS host bundle —
  verify node version + no native modules); `host.docker.internal` resolution
  (Docker Desktop Mac default = yes).

1. **Add a capture mode to `ds4-server`** (env-gated, off by default): during a
   greedy (`temperature: 0`) chat decode, append one record per sampled token
   — `(request_id, token_id, region_label, layer_40/41/42_mean)` — to a
   per-request JSONL file. Piggyback on the engine's existing layer-mean
   capture (the same `3*4096` hidden DSpark feeds the drafter); take
   `region_label` from the live `openai_stream_mode`, optionally keeping the
   raw DSML delimiter token ids (`tool_calls_start` / `invoke` / `parameter`)
   for sub-region granularity. DSpark stays **OFF** during capture — the
   target-only greedy trajectory is the ground truth the drafter is judged
   against and the baseline we later speed up. Greedy is already the
   speculation gate (`server_should_speculate()` requires `temperature ≤ 0`).
2. **Rig Pi-in-container + capture (single-instance smoke first).** Start a
   borrowed SWE-Bench eval container (`docker run -d … --add-host=
   host.docker.internal:host-gateway`, kept alive), install Node + Pi
   in-container, configure the `ds4` provider → `host.docker.internal:8000`
   (`temperature=0`). Smoke gate: one `pi -p` tool call (e.g. `read`/`bash`)
   round-trips through ds4-server with capture ON, yielding records with the
   expected `TOOL`/`THINKING`/`TEXT` `region_label`. Then run `pi -p "<problem
   statement>"` on **one instance** first, and expand to a powered slice across
   repos only if the single-instance trajectory shows a region signal worth
   powering — assistant turns (thinking + text + tool spans) captured as
   per-token JSONL. Memory-safe: ds4-server is the only heavy process; the
   container is light; checkpoint per-instance so a crash is recoverable.
3. **Offline per-region acceptance replay.** Feed the captured layer-means to
   the retained numpy / torch-MPS drafter oracle (Lead-03 style, fidelity-gated
   against the live target tokens); compare draft tokens to captured target
   tokens **partitioned by `region_label`**; report `p=1`, `E[a|4]`, `S(4)`,
   and cycle-jump speedup per region (`TOOL` vs `THINKING` vs `TEXT`) with
   prompt-clustered CIs.
4. **Decision:** if `TOOL`-region acceptance is not materially above the
   corpus mean (e.g. `E[a|4]` not clearly above the ~2.256 dynamic break-even),
   record the agentic regime as closed for this drafter and stop — no
   selective-enable engineering is warranted.

### Phase 2 — Selective-enable modeling (only if Phase 1 passes)

5. Add a per-region selective-enable replay to `model_spec_speedup.py`: DSpark
   on in tool-call regions, plain decode elsewhere, weighted by the measured
   region-token fractions from the real agent sessions. Feed the measured
   per-region acceptances and the ~1 ms region-switch cost into the existing
   cycle model.
6. Report the modeled trajectory-average speedup vs plain decode, with a
   region-fraction sensitivity sweep (how tool-call-heavy must a session be to
   clear the gate).

### Phase 3 — Live selective-enable measurement (only if Phase 2 projects ≥ +20%)

7. Add a region-gated mode to `ds4 --dspark` (enter spec only when the live
   `openai_stream_mode` indicates a tool-call region; plain-decode otherwise),
   on the **exact sequential verifier** (greedy-exact). Measure end-to-end with
   `ds4-spec-bench` on the real agent tasks, byte-diffed vs target-only decode.
   The batch verifier is **out of scope** here: its ~0.64% argmax flips would
   break both the exact-output gate and the tool-call syntax itself.

Estimated effort: Phase 1 ~3–6 days (ds4-server capture mode + fidelity gate;
rig Pi-in-container on a borrowed SWE-Bench instance; offline replay);
Phases 2/3 contingent on the prior phase's decision.

## Success criteria

- **Primary gate cleared (Phase 3):** ≥ +20% generation t/s vs the same-setup
  plain baseline on at least one realistic agentic SWE workload, exact greedy
  output preserved. This satisfies `GOAL.md`'s primary gate on the
  **intended** workload — arguably the gate's true spirit.
- **Projection clears (Phase 2):** the per-region selective-enable model
  projects ≥ +20% on a realistic tool-call fraction → commit to Phase 3.
- **Acceptance signal real but not gate-clearing (Phase 1 / 2):** tool-call
  regions show materially higher acceptance but the selective-enable projection
  lands in [secondary, +20%) → record as a workload-conditioned secondary-gate
  lever and a scope-narrowing input.
- **No material tool-call advantage (Phase 1):** record the agentic regime as
  closed for this drafter and refocus on drafter quality (Lead 07) /
  server-side batching.

## Non-goals

- a new drafter or drafter fine-tuning (Lead 07 owns quality);
- verifier / kernel engineering (Lead 08);
- server-side multi-request batching (a separate deployment-shape lead);
- a polished CLI or stable public flag;
- building a general agent harness — Pi is reused as-is for trajectory capture.

## Deliverables

1. A compact summary answering: how much higher is drafter acceptance in real
   tool-call regions than the corpus mean; whether region-selective DSpark
   clears the +20% gate on the intended agentic workload; and whether the
   agentic regime revives the local primary gate.
2. A retained real-agent trajectory corpus + per-region acceptance artifact.
3. If Phase 2 / 3 run, the selective-enable model and the live measurement with
   an exact-output gate.

## Exit conditions

- **Proceed:** Phase 1 shows a material tool-call-region acceptance advantage
  and Phase 2 projects ≥ +20% → run Phase 3.
- **Narrow:** the acceptance advantage is real but the selective-enable
  projection stays in [secondary, +20%) → record as workload-conditioned
  secondary material and a scope input.
- **Stop:** no material tool-call-region advantage → close the agentic regime
  for this drafter.

## One-line verdict

The dossier's "acceptance-limited, 0.98× baseline" conclusion rests on a
general-reasoning corpus that does not reflect ds4's intended agentic SWE
workload; Lead 09 tests whether real tool-call regions have materially higher
drafter acceptance, making region-selective DSpark the one unmeasured route to
the local +20% gate.
