## Verdict per current claim
- **“GPU drafter body/head is the next major lever” — likely-wrong.** Current exact sequential verifier caps a zero-draft-cost path around **35-38 t/s**, still under ~40 t/s baseline; verifier/state correctness comes first.
- **“Bottleneck picture: decode ~28 ms, fixed draft ~9-10 ms, scheduled verify ~27-56 ms” — sound but mislabeled.** `decode_ms` includes DSpark hidden push; `verify_ms` is serial top-only target decodes plus readback, near-zero on first miss.
- **“Keep verifier intact, move DSpark body/head to GPU” — questionable.** Keeping this verifier intact likely cannot win; also DSpark state update for accepted speculative drafts appears incomplete.
- **“Confidence scheduling remains part of DSpark design” — sound as secondary.** It should not drive the first GPU slice; Lead 02 says it is marginal under shipped verifier economics.

## Runtime decomposition as implemented now
CLI samples `first_token` from host logits, then enters DSpark only for greedy decode. Server path appears not DSpark-enabled because it checks only `ds4_engine_mtp_draft_tokens(...) > 1`, not `ds4_engine_has_dspark(...)`.

DSpark branch in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28362):
- `decode_ms`: optional initial DSpark push, then normal target `ds4_session_eval(first_token)`. Normal eval runs target GPU decode, reads full logits, pushes checkpoint, then calls `dspark_session_push_graph_hidden()` ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:27788)).
- DSpark hidden push is only partly GPU-first: it computes mean/`main_proj`/stage KV on GPU, then reads stage KV to host DSpark windows ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19874)).
- `draft_ms`: CPU DSpark drafter. Fixed path linearizes the 3 stage KV rings, runs 3 block bodies, then CPU head/Markov/confidence/argmax ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28069)). Scheduled path does token-serial `n_tok=1` drafting and stops by STS threshold ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28230)).
- `verify_ms`: compares row 0 against existing target logits, then for each accepted draft runs `metal_graph_eval_token_raw_swa_top()` sequentially, pushes checkpoint, and finally reads full logits if needed ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28428)).

Important caveat: accepted speculative drafts do **not** appear to call `dspark_session_push_graph_hidden()`. Normal eval does ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:27798)); verifier accepted drafts only call `token_vec_push` ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28442)). That means DSpark’s support-state window likely lags/skips accepted draft tokens.

## Ranked optimization opportunities
1. **Fix/prove DSpark state advancement for accepted drafts.** High impact, medium risk. Current acceptance/bottleneck numbers may be distorted if DSpark skips verified tokens.
2. **Fix/prove GPU hidden push semantics.** `main_proj` is documented as `3*4096 -> 4096`, but GPU push applies it to one 4096 layer mean at a time and overwrites the same slot; this needs parity against CPU/oracle before building on it.
3. **Verifier economics.** High impact, high risk. Current verifier is serial target decode; removing all draft time still gives only ~37.5 t/s on the 6-prompt sample and ~35.7 t/s on long prompts.
4. **CPU drafter head cleanup.** Full vocab base projection plus full Markov bias per draft row is still expensive; top-only/head fusion or GPU head is a narrower win than full body migration.
5. **CPU body cleanup.** Persistent Q8 activation scratch, less KV-ring linearization, and avoiding scheduled token-serial body when threshold rarely stops early.
6. **Confidence threshold retuning.** Only after corrected state/verifier timings.

## Best GPU migration path
Do **not** start with full 3-stage body/head while preserving the current verifier.

Better path:
1. Correct DSpark hidden/state update on GPU: concat or accumulate layers 40/41/42, run one `main_proj`, one norm, update persistent device DSpark KV.
2. Use that same path for every verified accepted draft, not only the normal `first_token`.
3. Build a low-K verifier that returns row tops, continuation logits, and DSpark state updates without serial full target decodes.
4. Then port DSpark head/confidence/argmax, then body.

Parity gates should compare against CPU/oracle on DSpark window state, draft ids, and confidence logits, not only final generated text.

## Confidence-scheduling implications for design
GPU should produce confidence logits if the drafter moves to GPU. STS thresholding and verify-length choice can remain host policy: it is five scalars.

Do not force GPU early-stop first. Full-block GPU draft plus host `verify_n` is simpler and avoids turning scheduling into many tiny GPU launches. Device-side early-stop only matters if profiling after verifier fixes shows body cost still dominates.

## Benchmark / methodology risks
- Temp parity forces `DS4_DSPARK_VERIFY_K=0`; it proves entry/logit parity, not multi-token speculative parity.
- Greedy byte exactness cannot catch bad drafter state; exact verifier masks bad drafts.
- Corpus benchmark artifact is only 6 prompts and appears all `dolly_0000..0005`.
- Harness skips existing per-prompt JSON, so reused artifact dirs can mix runs.
- Corpus/exactness artifacts do not record critical env like `N`, `LIMIT`, commit, threshold.
- `decode_ms` and `verify_ms` are not pure buckets; current labels can mislead roadmap decisions.
- Long-prompt profile has no paired baseline rows in the artifact.

## One recommended next implementation step
Implement a parity-gated DSpark state-update fix: after every accepted verifier draft, update `dspark_win_kv`; correct GPU `main_proj` to consume the 3-layer concatenation; then re-run draft-id/confidence/window parity and the same profile before starting full GPU body/head work.