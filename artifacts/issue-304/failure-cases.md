# Issue #304 Failure Cases

This artifact records mismatches that should remain rejected during distributed-prefill-to-local-decode work.

## Current Status

- Phase 3 recorded a real same-backend rejection case.
- Phase 2 observed `CUDA -> Metal` forced-token logit drift after resumed eval, but that is no longer the main blocker.
- The stronger rejection is: distributed-prefill -> merged `DSV4` -> local Metal decode does not stay close enough to a fresh local Metal baseline after resumed evaluation begins.
- Phase 4 added two new failure classes that must stay rejected:
  - unsafe second-process accelerator startup when free memory is already below the residency budget,
  - reused-worker lifecycle contamination where a later rerun can inherit stale state from an earlier coordinator connection.

## Recorded Rejection Cases

- `official ctx=16384 / long_code_audit`
  - Official gate: pass on all five runs.
  - Same-backend parity worst@5: `top5=4/5`, `top20=15/20`, `top64=55/64`, `rms=0.876981199`, `max_abs=4.63468361`.
  - Why rejected: the handoff path diverges materially from a fresh local Metal baseline even though selected tokens/top-logprobs stay inside the official API envelope.

- `local-golden ctx=5000 / long_story_4096`
  - Local-golden fixture: pass on all five runs.
  - Same-backend parity worst@5: `top5=3/5`, `top20=12/20`, `top64=46/64`, `rms=2.1708498`, `max_abs=12.3051796`, `top20_max_abs=7.73109055`.
  - Why rejected: the resumed checkpoint stays within the coarse local-golden regression envelope, but the next eight greedy steps do not match a fresh local Metal baseline closely enough.

- `phase3.5 official / short_code_completion / route1 pure local CUDA`
  - Official gate: fail at worst@5.
  - First selected mismatch step: `1`.
  - Max logprob delta: `1.42368603`.
  - Why rejected: this route misses the authoritative official selected-token/logprob surface before any distributed or resumed-handoff behavior is involved.

- `phase3.5 official / short_code_completion / route3 distributed generation CUDA -> Metal`
  - Official gate: fail at worst@5.
  - First selected mismatch step: `1`.
  - Max logprob delta: `0.85852015`.
  - Why rejected: direct distributed generation on this backend pairing already misses the official acceptance gate, so later resumed-route drift on the same pairing cannot be attributed only to payload resume.

- `phase3.5 official / short_code_completion / route5 resumed CUDA -> Metal`
  - Official gate: fail at worst@5.
  - First selected mismatch step: `1`.
  - Resumed-route parity: `top5=3/5`, `top20=14/20`, `top64=54/64`, `rms=1.25698996`, `max_abs=6.82095432`.
  - Why rejected: this cell misses both the official gate and resumed-route parity.

- `phase3.5 local-golden / long_story_4096 / route2 pure local Metal`
  - Local-golden fixture: fail at worst@5.
  - Golden overlap: `top5=3/5`, `top20=16/20`, `top64=46/64`, `top20_max_abs=5.2168293`.
  - Why rejected: the stored local-golden vector is not a reliable “pure local Metal canonical” surface for Phase 3.5 route attribution.

- `phase4 DGX startup / second CUDA residency request while stale worker is still live`
  - Observed guard rejection:
    - free `18.94 GiB`
    - request `42.03 GiB`
    - reserve `10.14 GiB`
    - total `121.69 GiB`
  - Why rejected: without an explicit guard this class previously exhausted DGX memory badly enough to freeze the box.

- `phase4 CUDA -> Metal / reused Metal worker rerun`
  - First unstable rerun: first mismatch at generated step `11`.
  - Second unstable rerun: first mismatch at generated step `0`; `handoff_first_token=5`, reference `2337`.
  - Why rejected: repeated handoff validation on the same long-lived Metal worker process is not trustworthy until reconnect/session cleanup is fixed.

- `phase4 CUDA -> Metal / lifecycle-hardening follow-up`
  - Fresh-worker rerun after session-id, stale-channel, and session-free fixes: first mismatch at generated step `12`.
  - Immediate next rerun on the same worker: first mismatch at generated step `0`; `handoff_first_token=5`, reference `2337`.
  - Why rejected: the lifecycle fixes improved the original stale-worker symptom, but they did not make repeated reused-worker parity reliable enough to use as a strict acceptance surface.

- `phase5 ds4-eval CUDA -> Metal local-decode / old 60s socket timeout`
  - Explicit DGX coordinator command:
    `./ds4-eval --ctx 16384 --plain --temp 0 --seed 1 --nothink --tokens 4096 --questions 6 --role coordinator --layers 0:21 --listen 192.168.1.98 1241`
  - Local Metal worker:
    `./ds4 --ctx 16384 --role worker --layers 22:output --local-decode --coordinator 192.168.1.98 1241`
  - Observed failure:
    - coordinator trace error:
      `failed to read frame header: Resource temporarily unavailable`
    - worker stderr:
      `distributed worker: protocol error: failed to read frame header: Resource temporarily unavailable`
  - Why rejected: the old default distributed socket timeout (`60s`) was shorter than a healthy one-shot local-decode answer generation in `ds4-eval`.
  - Current status: fixed by raising the default distributed socket timeout to `600s`; keep this recorded as the historical failure class that explained the reported “dies after 5-6 questions” blocker.

## Phase 5 user-visible negative cases

- `--local-decode` on a coordinator process
  - Diagnostic: `--local-decode requires --role worker`
  - Status: reject at distributed option validation before runtime.

- `--local-decode` on a worker without `N:output`
  - Diagnostic: `--local-decode requires --layers N:output`
  - Status: reject before runtime because the worker would not own the output head.

- local-decode handoff without an advertised local-decode worker
  - Diagnostic: `distributed handoff requires worker local-decode capability`
  - Status: reject at handoff time; the route exists, but the final worker did not advertise the needed capability.

- local-decode handoff to a route whose final worker does not own output logits
  - Diagnostic: `distributed handoff requires worker-owned output head`
  - Status: reject at handoff time; the workflow only supports final-worker decode.

- pushed-shard token timeline mismatch
  - Diagnostics seen in protocol handlers:
    - `worker snapshot token hash mismatch`
    - `snapshot load token hash mismatch`
    - `worker local generate token hash mismatch`
  - Status: fail closed; stale or mismatched worker state must not be reused silently.

- worker/coordinator payload layout mismatch
  - Diagnostic class: snapshot layout validation rejects mismatched `prefill_cap`, layer range, or payload metadata before accepting the shard.
  - Status: preserve as a hard rejection; this is an integrity boundary, not a tolerable drift case.

## Phase 4 closeout classification

- The two Phase 4 `CUDA -> Metal` reused-worker rerun failures remain recorded here because they fail strict token-parity expectations.
- They do not block Phase 4 closeout because the diagnostic traces now show:
  - coherent decoded text rather than gibberish,
  - near-matching logits with full `top5`/`top10` overlap at the first divergent step,
  - and no evidence that the final-worker handoff workflow itself is invalid.
- Treat them as the same practical class of backend-sensitive near-top1 variance already documented in Phase 3.5 unless a later phase proves a concrete stale-state or payload-integrity defect.

## Phase 5 closeout classification

- The fresh-worker worker-owned `--local-decode` path is now considered valid for Phase 5 closeout.
- The remaining reusable-session follow-up drift does not stay in the Phase 5
  blocker set unless it reappears as:
  - stale coordinator KV after catch-up,
  - token-hash/session-integrity failure,
  - or a broken subsequent prompt control-flow path.
- Treat score variance in the full `ds4-eval` runs as evaluation noise for
  Phase 5 purposes, not as a new rejection case for the handoff workflow.

## Phase 5.5 closeout classification

- Phase 5.5 no longer carries any open rejection case under Issue 304.
- The previously failing reusable-session cells are reclassified as bounded
  prefill-vs-decode variance because:
  - reused distributed sessions matched fresh decode-replay references
    exactly on the replay diagnostics,
  - and pure local Metal/CUDA reproductions showed the same prefill-vs-decode
    split without any distributed handoff path.
- Keep the remaining integrity failures as hard blockers:
  - token-hash mismatch
  - payload layout mismatch
  - stale KV/session contamination
  - missing output-head ownership
  - or protocol-level corruption
- Do not reopen Issue 304 merely because fresh full-transcript prefill and
  decode replay land on different logits or different greedy continuations.
  That behavior is now accepted as bounded numerical trajectory variance,
  subject to future benchmarking rather than correctness rejection.

## Phase 3 Rejection Criteria

A drift case should be recorded here as rejected if it:

- makes distributed prefill -> merged `DSV4` -> local decode differ from fully local inference on the same decode backend,
- fails the selected-token/top-logprob envelope used by `./ds4_test --logprob-vectors`,
- fails the local-golden drift envelope used by `./ds4_test --local-golden-vectors`,
- changes greedy continuation where the same-backend fully-local baseline stays stable,
- or localizes to payload ABI, shard metadata, token hash, context, layer range, output-head, or cache-state corruption.

Cross-engine forced-logit differences alone are not a rejection case if same-backend handoff parity and official/local gates pass.

## Additional Layout Constraint

- Merged `DSV4` staging also rejects worker/coordinator `prefill_cap` mismatch.
- Phase 3 required worker `DS4_METAL_PREFILL_CHUNK` alignment with the local environment:
  - `2048` for the official-vector runs
  - `4096` for the local-golden runs

## Future Failure Classes To Preserve

- Token hash mismatch.
- Layer range mismatch.
- Context/raw-window/compression-cap mismatch.
- Model id or quant profile mismatch.
- Missing output head when logits are requested.
- Worker reconnect before shard fetch.
- Stale worker session after route rebuild.
- Stale data-connection thread surviving a coordinator reconnect.
- Metal runtime scratch or command-state reuse across worker reconnects.
- Incomplete or stale incremental KV return chunks.
