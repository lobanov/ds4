# Lead 09 — Agentic / tool-call decode regime (region-selective DSpark)

Date: 2026-07-13. **Revised 2026-07-20: add a second trajectory carrier —
SCBench (SlopCodeBench) cp1 via SCBench's managed `pi` adapter (3 random
problems × cp1 only) — and refresh the baseline framing against the M3
always-on path (which now beats plain by +4.9%, so selective-enable must beat
not only plain but also the current best always-on config).** Status:
**pending (revised, not yet executed).** Independent of leads 05 / 07 / 08 for
its offline phase; the cheapest decisive experiment can run in parallel with
all of them.

## Execution contract

Pursue this lead under the general research methodology in
`.pi/skills/research-lead/SKILL.md`:

- orient and verify all claimed assets before acting;
- lock the decision rule and estimator before trusting measurements;
- fidelity-gate any new acceptance path against retained oracle /
  powered-acceptance tooling;
- report prompt-clustered CIs and **per-region** (not just corpus-mean)
  results, **stratified by carrier** (SWE-Bench vs SCBench cp1);
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

Three existing facts make this lead more than speculation:

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
3. **M3 has moved the always-on baseline, but not the gate.** As of
   2026-07-15, the full DSpark stack (committing batched verify + anchor-reuse
   + prefix-checkpoint + GPU Metal drafter + STS re-tune) **beats plain ds4 by
   +4.9 %** (40.04 vs 38.16 t/s; score-neutral on the 92Q) — the first config
   to beat plain locally (`summaries/dspark_runtime_milestone_3_progress.md`).
   This **raises the selective-enable bar**: region-selective DSpark must beat
   not only plain decode (GOAL.md's primary gate) but also the **M3 always-on
   path**, otherwise selective-enable is dominated by simply leaving the
   always-on path on. The +20 % gate vs plain is unchanged; Phase 2/3 report
   against **both** baselines.

The specific doubt: does the retained Q4_K drafter produce **materially higher
acceptance on real agent tool-call regions** than on the measured corpus mean
(`E[a|4]=2.198`, `0.982×` baseline, dynamic break-even `2.256`), high enough
that **region-selective DSpark** over a realistic agent trajectory clears the
+20 % primary gate against plain **and** beats the M3 always-on path — even
though the always-on corpus-average path clears neither the old 0.98× framing
nor (by enough) the +20 % gate?

This is the one remaining route to the local primary gate that has **never been
measured** in this dossier, and unlike drafter quality (Leads 07 / 10, both
closed) or kernel work (Lead 08) it is testable with a cheap offline acceptance
experiment before any engineering investment.

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

Phase 1 captures **two complementary agent regimes**, reported stratified:

- **Carrier 1 — SWE-Bench Verified (bug-fix trajectories on real repos,
  diverse tools).** Pi-in-container on a borrowed eval container. The
  general-purpose regime: open-ended debugging on a real codebase.
- **Carrier 2 — SCBench / SlopCodeBench cp1 (from-scratch code generation,
  SCBench-managed `pi` adapter).** Added 2026-07-20: SCBench already ships a
  first-class `pi` agent and a Docker runtime, so this arm reuses SCBench's
  own `slop-code run --agent pi` orchestration; we only add a `ds4` provider
  entry and the ds4-server capture mode. Phase 1 captures **cp1 only** (the
  from-scratch initial implementation) for **3 random problems** — the
  iterative-extension checkpoints (cp2..N) are a Phase 1.5 expansion only if
  cp1 shows a region signal.

#### Carrier 1 — Pi (diverse tools) on a borrowed SWE-Bench Verified instance

The carrier choice is load-bearing: the point is a *representative* tool-call
token distribution, so the agent must emit *diverse* structured tool calls on a
*real* task.

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
  the container is light. Pi's diverse tool calls flow through ds4-server,
  which captures the per-token traces. **SWE-bench grading is optional bonus**
  (copy the patch back and run the harness) — Lead 09 needs the trajectory,
  not the score.
- **Feasibility risks to confirm in orientation:** linux/amd64 emulation on
  arm64 Mac slows container-side tool ops (LLM decode is host-side,
  unaffected); Node/Pi install in-container (npm network, or copy the pure-JS
  host bundle — verify node version + no native modules);
  `host.docker.internal` resolution (Docker Desktop Mac default = yes).

#### Carrier 2 — SCBench cp1 via the managed `pi` adapter

SCBench (problems: `gabeorlanski/scb-problems`; harness:
`SprocketLab/slop-code-bench`) is purpose-built for iterative feature
introduction — each problem is a sequence of checkpoints, the agent implements
cp1 from scratch then extends its own solution. Its harness already supports
Pi as a first-class agent and runs the agent inside a Docker environment it
manages, so the "Pi-in-container" discipline is satisfied by SCBench's own
runner: no manual Pi rig is needed for this carrier.

- **Setup:** clone `SprocketLab/slop-code-bench`; add a `ds4` provider entry
  to `configs/providers.yaml` (`base_url: http://host.docker.internal:8000/v1`,
  OpenAI-compatible, no API key) and a `deepseek-v4-flash` model entry to
  `configs/models/`; point the `pi` agent at it. ds4-server stays host-side
  (the single ~87 GB heavy process); the SCBench Docker env is light.
- **Scope:** sample **3 random problems** from the 36-problem v1.0 set;
  capture **cp1 only** per problem — the from-scratch initial implementation,
  the cleanest signal of the from-scratch code-generation regime before any
  iterative extension. cp2..N (iterative feature introduction proper, the
  even-more-structured extension regime that the user's motivation centers on)
  is a **Phase 1.5 expansion** only if cp1 shows a region signal worth
  powering.
- **Capture:** identical to Carrier 1 — ds4-server capture mode ON, DSpark
  OFF, `temperature=0`, one per-token JSONL record per sampled token.
- **Eval/erosion skipped:** `slop-code eval` (correctness) and
  `slop-code metrics judge` (erosion) are **out of scope** — Lead 09 needs the
  trajectory, not the score. An optional sanity `eval` is fine only if a
  checkpoint's tool-call traffic looks malformed.
- **Feasibility risks to confirm in orientation:** the `ds4` provider config
  is accepted by SCBench's `pi` adapter (high likelihood — OpenAI-compatible —
  but verify the agent actually invokes `pi` and not a stub); the exact CLI
  flag to run cp1 only (e.g. `--checkpoint 1` / `--max-checkpoints 1` —
  confirm against `slop-code run --help`); Docker Desktop's
  `host.docker.internal` resolution (Mac default = yes); the `temperature=0`
  setting flows through SCBench's provider to ds4-server (greedy is the
  speculation gate).

#### Phase 1 shared steps

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
2. **Rig both carriers (single-instance smoke first, each).**
   - **Carrier 1 (SWE-Bench):** start a borrowed SWE-Bench eval container
     (`docker run -d … --add-host=host.docker.internal:host-gateway`, kept
     alive), install Node + Pi in-container, configure the `ds4` provider →
     `host.docker.internal:8000` (`temperature=0`). Smoke gate: one `pi -p`
     tool call (e.g. `read`/`bash`) round-trips through ds4-server with
     capture ON, yielding records with the expected `TOOL`/`THINKING`/`TEXT`
     `region_label`. Then run `pi -p "<problem statement>"` on **one instance**
     first; expand to a powered slice across repos only if the single-instance
     trajectory shows a region signal worth powering.
   - **Carrier 2 (SCBench):** smoke gate = one
     `slop-code run --agent pi --model ds4/deepseek-v4-flash --problem <easy>`
     round-trips through ds4-server with capture ON, yielding records with the
     expected `region_label`s. Then run cp1 on the 3 sampled random problems.
   - Memory-safe either way: ds4-server is the only heavy process; both
     containers are light; checkpoint per-instance / per-problem so a crash is
     recoverable.
3. **Offline per-region acceptance replay.** Feed the captured layer-means to
   the retained numpy / torch-MPS drafter oracle (Lead-03 style,
   fidelity-gated against the live target tokens); compare draft tokens to
   captured target tokens **partitioned by `region_label`**; report `p=1`,
   `E[a|4]`, `S(4)`, and cycle-jump speedup per region (`TOOL` vs `THINKING`
   vs `TEXT`) with prompt-clustered CIs, **stratified by carrier** (SWE-Bench
   vs SCBench-cp1) and never pooled into a single headline.
4. **Decision:** if `TOOL`-region acceptance is not materially above the
   corpus mean on **either carrier** (e.g. `E[a|4]` not clearly above the
   ~2.256 dynamic break-even), record the agentic regime as closed for this
   drafter and stop — no selective-enable engineering is warranted. If the
   signal is real on one carrier only, record that carrier's regime as the
   workload-conditioned lever and treat the other as the negative control.

### Phase 2 — Selective-enable modeling (only if Phase 1 passes)

5. Add a per-region selective-enable replay to `model_spec_speedup.py`: DSpark
   on in tool-call regions, plain decode elsewhere, weighted by the measured
   region-token fractions from the real agent sessions (per carrier). Feed the
   measured per-region acceptances and the ~1 ms region-switch cost into the
   existing cycle model.
6. Report the modeled trajectory-average speedup **dual-baselined**: vs plain
   decode (GOAL.md's gate) **and** vs M3 always-on DSpark (the current best).
   Include a region-fraction sensitivity sweep (how tool-call-heavy must a
   session be to clear each baseline).
7. **Phase 1.5 expansion (optional, only if cp1 shows signal):** extend the
   SCBench capture to cp2..N on the sampled problems (iterative feature
   introduction proper — the regime the user's motivation centers on) and
   re-report per-region plus a new per-checkpoint-position axis (cp1
   from-scratch vs cpN iterative extension). This is the natural follow-on
   when cp1 alone clears the signal bar.

### Phase 3 — Live selective-enable measurement (only if Phase 2 projects ≥ +20 % vs plain AND > M3 always-on)

8. Add a region-gated mode to `ds4 --dspark` (enter spec only when the live
   `openai_stream_mode` indicates a tool-call region; plain-decode otherwise),
   on the **exact sequential verifier** (greedy-exact). Measure end-to-end
   with `ds4-spec-bench` on the real agent tasks, byte-diffed vs target-only
   decode **and** vs M3 always-on. The batch verifier is **out of scope**
   here: its ~0.64 % argmax flips would break both the exact-output gate and
   the tool-call syntax itself.

Estimated effort: Phase 1 ~3–6 days (ds4-server capture mode + fidelity gate;
rig both carriers — the SWE-Bench Pi-in-container rig is the longer pole,
SCBench is near-free given its `pi` adapter; offline replay); Phases 2/3
contingent on the prior phase's decision.

## Success criteria

- **Primary gate cleared (Phase 3):** ≥ +20 % generation t/s vs the same-setup
  plain baseline on at least one realistic agentic workload (SWE-Bench or
  SCBench), exact greedy output preserved, **and** beating the M3 always-on
  path on the same workload. This satisfies `GOAL.md`'s primary gate on the
  **intended** workload — arguably the gate's true spirit.
- **Projection clears (Phase 2):** the per-region selective-enable model
  projects ≥ +20 % vs plain **and** > M3 always-on on a realistic tool-call
  fraction → commit to Phase 3.
- **Acceptance signal real but not gate-clearing (Phase 1 / 2):** tool-call
  regions show materially higher acceptance but the selective-enable
  projection lands in [secondary, +20 %) or below M3 always-on → record as a
  workload-conditioned secondary-gate lever and a scope-narrowing input.
- **No material tool-call advantage (Phase 1):** record the agentic regime as
  closed for this drafter and refocus on Lead 08 (fused verify kernel) /
  server-side batching.

## Non-goals

- a new drafter or drafter fine-tuning (Leads 07 / 10 — both closed);
- verifier / kernel engineering (Lead 08);
- server-side multi-request batching (a separate deployment-shape lead);
- a polished CLI or stable public flag;
- building a general agent harness — Pi is reused as-is (via SCBench's
  adapter for Carrier 2; via the manual in-container rig for Carrier 1);
- **SCBench erosion / correctness scoring** — `slop-code eval` and
  `slop-code metrics judge` are out of scope; Lead 09 uses SCBench for
  trajectory capture only, not for benchmark scoring.

## Deliverables

1. A compact summary answering: how much higher is drafter acceptance in real
   tool-call regions than the corpus mean; whether the advantage holds across
   both agent regimes (SWE-Bench bug-fix vs SCBench from-scratch cp1);
   whether region-selective DSpark clears the +20 % gate vs plain **and**
   beats M3 always-on; and whether the agentic regime revives the local
   primary gate.
2. A retained real-agent trajectory corpus + per-region acceptance artifact,
   stratified by carrier.
3. If Phase 1.5 / 2 / 3 run: the iterative-extension capture, the
   selective-enable model, and the live measurement with an exact-output gate.

## Exit conditions

- **Proceed:** Phase 1 shows a material tool-call-region acceptance advantage
  on at least one carrier and Phase 2 projects ≥ +20 % vs plain **and** > M3
  always-on → run Phase 3.
- **Narrow:** the acceptance advantage is real but the selective-enable
  projection stays in [secondary, +20 %) or below M3 always-on → record as
  workload-conditioned secondary material and a scope input.
- **Stop:** no material tool-call-region advantage on either carrier → close
  the agentic regime for this drafter.

## One-line verdict

The dossier's "acceptance-limited, 0.98× baseline" conclusion rests on a
general-reasoning corpus that does not reflect ds4's intended agentic SWE
workload; Lead 09 tests whether real tool-call regions (SWE-Bench bug-fix +
SCBench from-scratch cp1) have materially higher drafter acceptance, making
region-selective DSpark the one unmeasured route to the local +20 % gate — now
benchmarked against the M3 always-on path that already beats plain by +4.9 %.

## Worklog

### 2026-07-20 — revision: add SCBench cp1 arm; refresh M3 baseline framing

Revised the lead before execution to reflect two decisions: (1) add a second
trajectory carrier — **SCBench (SlopCodeBench) cp1 via SCBench's managed `pi`
adapter** (depth A: reuse `slop-code run --agent pi`, add a `ds4` provider
entry, no manual Pi rig), sampling **3 random problems × cp1 only**; (2)
refresh the baseline framing against **M3 always-on DSpark** (now +4.9 % vs
plain), so Phase 2/3 report dual-baselined (vs plain = GOAL.md gate; vs M3
always-on = current best). Phase 1 now reports per-region acceptance
**stratified by carrier** (SWE-Bench vs SCBench-cp1), never pooled; the SCBench
cp2..N iterative-extension regime (the user's stated motivation) is deferred
to a Phase 1.5 expansion only if cp1 shows signal. SCBench erosion/correctness
scoring explicitly out of scope (carrier only). Leads 07/10 (drafter quality)
marked closed in the rationale, since both closed negative post-original-date.
Not yet started; pending goal assignment.

### 2026-07-20 — orientation (task lead09-orient): assets verified, 3 feasibility findings, decision rule LOCKED

Orientation complete before any measurement (goal `mrszcq1x-h5ls26`, task
`lead09-orient`). Findings recorded for the codex setup gate to audit.

**Assets verified present (re-confirmed from pre-goal orientation + new
SCBench specifics):**

- `ds4_server.c:5047+` — the `openai_stream_mode ∈ {THINKING, TEXT, TOOL,
  SUPPRESS}` region tag + the DSML tool-call parser state machine
  (`DSML_TOOL_BETWEEN_INVOKES` / `_BETWEEN_PARAMS` / `_PARAM_VALUE` / `_DONE`).
  Region labels are free during streaming decode.
- `ds4.c:11098–16532` — the engine layer-mean capture path
  (`DS4_METAL_GRAPH_DUMP_PREFIX`, `hc_ffn_post`, layers 40/41/42 → `3*4096`),
  the same hidden DSpark feeds the drafter; used by Stage 2 / Lead 03.
- Binaries built: `ds4` (Jul 19), `ds4-server` (Jul 19), `ds4-spec-bench`
  (Jul 18). All on PATH.
- Oracle + replay: `issue468/dspark_oracle/measure_acceptance_bundle.py`
  (numpy + torch-MPS) + `issue468/run_lead03_*.py` (cycle-jump + trajectory +
  aggregate). These are the fidelity-gate references.
- SWE-Bench rig: `~/Projects/swe-bench` (`.env → LLM_BASE_URL=
  http://127.0.0.1:8000/v1`, `SWEBENCH_SUBSET=verified`, metal run dirs present).
- **SCBench (NEW):** cloned to `~/Projects/slop-code-bench` (depth-1, main).
  `uv sync --frozen` succeeds. `configs/agents/pi.yaml` confirms `pi` is a
  first-class agent (`type: pi`, `binary: pi`, `version: 0.74.0`). `--no-evaluate`
  flag available (skips eval/erosion per the goal's non-goal).
  `slop-code problems ls` lists the 36-problem v1.0 set (the sampling frame:
  `code_search`, `migrate_configs`, `textdrop`, `xjq` easy; `trajectory_api`,
  `pwd_manager`, `mvvault` medium; `circuit_eval`, `test_translator`, `recli`,
  `sheeteval`, `sith`, `rejector`, `mocked_http`, `metric_transform_lang` hard;
  etc.). Problems are fetched dynamically from `gabeorlanski/scb-problems` via
  `src/slop_code/problem_catalog.py` (no separate clone needed).

**Three feasibility findings (rig-affecting; resolve in `lead09-rig-smoke`):**

1. **No cp1-only CLI flag.** `slop-code run --help` exposes no `--checkpoint`
   / `--max-checkpoints` / `--only-cp1`. The only checkpoint-shaping config is
   `one_shot`, and `apply_one_shot_mode()` (`one_shot.py`) **collapses** all
   checkpoints into one (combined spec joined in checkpoint order, evaluated
   against the final cp's tests) — the opposite of cp1-only. The run config
   is `extra="forbid"`, so no hydra override key for checkpoint slicing.
   **Candidate cp1-only mechanisms** (pick at start of rig-smoke):
   (a) **local stripped problem config** — fetch the problem YAML from
   `scb-problems`, rewrite `checkpoints: {checkpoint_1: ...}` dropping cp2..N,
   point SCBench at the local copy (cleanest IF SCBench accepts local problem
   paths — verify first);
   (b) **kill after cp1's solve** — run normally, terminate once cp1 completes
   (robust IF checkpoints are separate API conversations, which the
   worker/driver architecture implies but must be confirmed);
   (c) capture-all + filter cp1 by `request_id` in the offline replay
   (easiest, but runs cp2..N decode which the user's scope excludes — last
   resort only).
   **Default plan:** try (a) first; fall back to (b); use (c) only if both
   blocked. If all three blocked → goal blocker (b) (carrier rig fails smoke).
2. **Provider shape confirmed; acceptance TBD at smoke.** `configs/providers.yaml`
   supports an `endpoints: {openai: {api_base, api_format: openai}}` block
   (see `openrouter`, `zhipu-coding-plan`). The `ds4` provider entry will be:
   `api_base: http://host.docker.internal:8000/v1`, `api_format: openai`, no
   real API key (ds4-server is open). Model entry `configs/models/deepseek-v4-flash.yaml`
   with `provider: ds4`. **Load-bearing smoke question:** does SCBench's pi
   adapter actually consume the provider `endpoints.openai.api_base` (vs
   relying on pi's own provider config)? Confirm by smoke run.
3. **Pi version mismatch.** SCBench pins `pi` at 0.74.0; installed is 0.80.3.
   Likely backward-compatible (the adapter calls the `pi` binary with
   stable flags), but verify the smoke `pi -p` invocation behaves; override
   via `agent.pi.version=0.80.3` or edit `configs/agents/pi.yaml` if it pins.

**Machine state:** ~9.7 GB free + ~81 GB inactive (reclaimable); no `ds4` /
heavy Python running. Single-process discipline still applies (no concurrent
ds4-server + heavy measurement).

**=== Phase 1 DECISION RULE — LOCKED BEFORE MEASUREMENT ===**

*Primary metric (per region, per carrier):* **per-region cycle-jump speedup**
= the speedup DSpark would achieve if every cycle landed in that region,
computed at corpus-level decode/verify/draft costs (the model currency from
`issue468/model_spec_speedup.py`). Reported for TOOL / THINKING / TEXT with
**prompt-clustered bootstrap 95% CIs** (resample prompts, not steps — steps
within a prompt are correlated; per skill principle 5/7).

*Rationale for speedup over raw `E[a|4]`:* the dynamic break-even `E[a|4]`
(2.256 at the corpus `S(4)=0.340`) depends on `S(4)`, which varies by region.
Raw `E[a|4]` against a single 2.256 threshold conflates the acceptance level
with the region's `S(4)` shift. Per-region speedup folds in the region-specific
`S(4)` via the model → clean apples-to-apples comparison. (Raw `E[a|4]`,
`S(4)`, `p=1` still reported as diagnostics; speedup is the decision metric.)

*Tiers (per carrier — SWE-Bench, SCBench-cp1 — evaluated independently):*

- **PROCEED:** TOOL-region cycle-jump speedup **point estimate ≥ 1.20×** AND
  **CI lower bound > 1.05×**. (1.20× aligns with the +20% gate spirit at the
  per-region level; 1.05× CI lower ensures the signal clears 1.0× with margin.)
- **NARROW:** TOOL-region speedup **CI lower > 1.0×** but below PROCEED
  (point < 1.20× OR CI lower ≤ 1.05×). Real signal, sub-gate → record as
  workload-conditioned secondary material.
- **STOP:** TOOL-region speedup **point < 1.05× AND CI lower ≤ 1.0×**. No
  material signal even at the point estimate → record the regime as closed
  *at the measured sample power*.

*Overall Phase 1 verdict:*
- **≥1 carrier PROCEEDs** → run Phase 2 (`lead09-phase2-model`) on the
  proceeding carrier's data; the other carrier is the stratified comparison.
- **All carriers NARROW** → record as workload-conditioned secondary; skip
  Phase 2; complete at the verdict gate.
- **All carriers STOP** → record the regime as closed at sample power; skip
  Phase 2; complete at the verdict gate.

*Honesty caveats (per skill principle 6/7, pre-registered):*
- **Never pool carriers.** SWE-Bench and SCBench-cp1 are reported
  independently; a pooled headline is forbidden.
- **Sample-power disclosure.** SCBench arm = n=3 problems (smoke-grade);
  SWE-Bench arm starts at n=1 (smoke) and expands only on signal. With these
  Ns, CIs will be wide. A STOP records **"no signal at smoke power;
  non-inferiority NOT established"**, NOT "regime definitively closed." A
  definitive close would need a powered slice — out of scope this cycle
  (goal's scope-expansion blocker rule → user check-in).
- **MDE reported** vs the 1.20× / 1.05× / 1.0× bars, not just point estimates.
- **M3 baseline carried from goal:** even on PROCEED, Phase 2's projection
  must clear ≥+20% vs plain AND > M3 always-on to recommend Phase 3 (out of
  scope this cycle regardless). PROCEED is necessary, not sufficient, for a
  "revive the gate" verdict.

Next: task `lead09-instrument` — build the env-gated ds4-server capture mode
+ extend the Lead-03 replay for per-region × per-carrier stratification, then
fidelity-gate against a retained Lead-03 baseline.

### 2026-07-20 — instrument design locked (task lead09-instrument, in progress)

Investigated the capture-path options before writing code. **Design decision:
reuse the existing `metal_graph_debug_dump_tensor` env-driven path — ZERO engine
surgery.** Findings:

- `ds4.c:11116 metal_graph_debug_wants()` reads `getenv("DS4_METAL_GRAPH_DUMP_PREFIX")`
  every call, filters by `_NAME` (substring), `_LAYER` (comma list), `_POS`.
  `ds4.c:11157 metal_graph_debug_dump_tensor()` then GPU-reads-back + writes
  `<prefix>_<pathtag><name>-<il>_pos<pos>.bin`. This is the EXACT mechanism
  Stage 2 used (`run_stage2_capture.py` set `DS4_METAL_GRAPH_DUMP_NAME=hc_ffn_post`,
  `_LAYER=40,41,42`). It fires from the release Metal decode path every token,
  gated only on the env — so it works during a `ds4-server` run too.
- The DSpark capture path (`ds4.c:16225`, `g->dspark_capture_hc[capture]`)
  WAS the alternative — but it's gated on `g->dspark_enabled` and exposing it
  with DSpark off means engine surgery (alloc + gate-flip + readback API) with
  real risk to the production decode path. **Rejected** in favor of the env-dump
  path, which needs no engine changes and is fidelity-trivial to gate (env unset
  → `metal_graph_debug_wants` returns false → zero dumps, zero overhead).
- Server hook point: the decode loop at `ds4_server.c:10401`, token sample at
  `:10422`, per-token region available right after `dsml_decode_tracker_update`
  (`:10507`) + `thinking_state_feed` (`:10505`). Region =
  {TOOL if `dsml_decode_state_is_tool(dsml_tracker.decode)`, THINKING if
  `thinking.inside`, else TEXT}. This is per emitted token, in the inner loop.
- Concurrency: the model is single-instance (~87 GB, instance lock); decode is
  serialized, so `setenv("DS4_METAL_GRAPH_DUMP_PREFIX", "<dir>/<request_id>_", 1)`
  at request-start is safe (no concurrent decode to race).
- Capture config: DSpark OFF + no `--mtp` + temp=0 → `server_should_speculate`
  returns false (no draft tokens) → plain `ds4_session_eval` → 1 token per loop
  iteration → clean 1:1 `completion ↔ engine_pos` mapping for the offline join.

**Build plan (task lead09-instrument, next slice):**
1. `ds4_server.c` capture mode: env `DS4_CAPTURE_TRAJECTORY=1` +
   `DS4_CAPTURE_TRAJECTORY_DIR=<dir>`. At request-start (serialized):
   `setenv` the dump prefix to `<dir>/<request_id>_` + `_NAME=hc_ffn_post` +
   `_LAYER=40,41,42`; open `<dir>/<request_id>_tokens.jsonl`. In the decode
   loop: append `{request_id, completion, abs_pos, token, region}` per token.
   Env unset → no setenv, no JSONL, zero overhead (fidelity-trivial).
2. Rebuild `ds4-server`.
3. Smoke the env-gate: off → no dump files, baseline t/s; on → dump files +
   JSONL with expected region labels.
4. Replay harness: extend `run_lead03_cyclejump.py` (or a new
   `run_lead09_region_replay.py`) to load the dump `.bin`s + JSONL, run the
   drafter oracle per-region × per-carrier, compute cycle-jump speedup with
   prompt-clustered bootstrap CIs.
5. Fidelity gate: the extended oracle reproduces a retained Lead-03 baseline
   bit/token-for-bit (100% draft-token agreement on a known-good slice).

**Risk noted:** per-token-per-layer dump files mean ~3 small files/token. For
Phase 1's smoke scope (3 SCBench × cp1 + SWE-Bench n=1, ~10-30k tokens) that's
~90k files — manageable with per-request subdirs. If a powered slice later
pushes file volume into inode-pressure territory, revisit (the engine-surgery
path remains a fallback).

### 2026-07-20 — capture mode built + smoked (task lead09-instrument, partial)

Built the env-gated capture mode in `ds4_server.c` (3 edits, zero engine
changes): preamble setup (per-request `setenv` of `DS4_METAL_GRAPH_DUMP_PREFIX`
to `<dir>/<id>_` + open `<dir>/<id>_tokens.jsonl`), per-token record after
`dsml_decode_tracker_update` (`{req,completion,abs_pos,token,region}`, region =
TOOL if `dsml_decode_state_is_tool`, THINKING if `thinking.inside`, else TEXT),
teardown (`fflush`/`fclose` + `unsetenv`). Env unset → `capture_active=false`
→ block skipped → baseline unchanged (one NULL-check/token). Rebuilt clean
(`-Wall -Wextra`, no warnings). Committed-pending (lands with the task).

**Env-gate smoke (capture ON, one model load, ds4-server on M5 Max):** a 20-token
non-tool chat request produced:
- **60 dump files** = 3 layers (40/41/42) × 20 tokens, named
  `<id>_hc_ffn_post-<il>_pos<pos>.bin`, pos 12–31 (= 12-token prompt + 20
decoded).
- **20 JSONL records** (`chatcmpl-1_tokens.jsonl`), 5 fields each, all region
  = **THINKING** (deepseek-v4-flash reasons by default; `thinking.inside`
  drove the tag → THINKING path confirmed). TEXT/TOOL paths exercise in
  rig-smoke with real tool-call carriers.
- **.bin content validated** (venv numpy): 16384 floats/layer (65536 bytes),
  all finite + nonzero; layers 40/41/42 differ from each other; consecutive
  tokens differ (pos12≠pos13) → real per-layer per-token hiddens.
- The new binary served the chat completion correctly with capture ON → the
  serving path is intact; the OFF path (capture block skipped) is
  code-inspection-verified (trivially correct).

**Two indexing notes for the replay harness (codex setup gate will audit):**
1. **dump_pos ↔ abs_pos offset.** Dumps at pos 12–31; JSONL abs_pos 13–32 →
   observed `dump_pos = abs_pos − 1`. The replay join must map each TOKEN to
   its INPUT hidden (the hidden that predicted it), not its OWN-position
   hidden. Resolve with a ground-truth probe (decode a deterministic token,
   find which dump's hidden the drafter maps to it) before trusting any
   acceptance number — this is the d2t-class off-by-one risk (skill principle 5).
2. **`after_ffn_hc` = 16384 floats/layer**, not the naïve 4096. The drafter
   input is the concatenation of layers 40/41/42 — confirm the exact drafter
   input dim (3 × 16384? or a projected sub-dim?) against `drafter_body.py` /
   the oracle loader before building the replay.

Server stopped after the smoke (freed 87 GB; next slice is offline). Smoke
artifacts left in `/tmp/lead09_smoke/` (throwaway; not retained).

**Task lead09-instrument NARROWED, not complete:** capture mode done + smoked;
the replay harness (step 4) + fidelity gate (step 5) remain as the next slice
(offline — reads the dump `.bin`s + JSONL, runs the drafter oracle per-region ×
per-carrier with prompt-clustered CIs, fidelity-gates vs a retained Lead-03
baseline).
