## Verdict Per Claim

1. **moe-equality: questionable.**  
   I recomputed row0 dumps: `ffn_norm` max diff `0.01413`, gate `6.736` = `476.7x`, up `7.553` = `534.5x`, swiglu `0.02996`, out `0.01765`. That is **consistent with inherited input amplification**, not proof. A real same-stage reduction mismatch could be masked. Source supports near-equivalence (`metal/moe.metal` shared/duplicated IQ2XXS pair path: [impl](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:680), [decode swiglu](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1022), [batch addr call](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1308)), but the verdict should say **“consistent with,” not “confirmed,”** unless the identical-input harness was actually run.

2. **attention-residual: sound for swap-only.**  
   I found no config/env that routes K=4 batch HC/compressor/attention through decode reductions. Batch verify calls `metal_graph_encode_layer_batch` → attention batch + FFN batch ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19652)); those functions are separate batch paths ([attention](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:17704), [FFN](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19171)). The `DS4_METAL_DISABLE_*_FUSION` flags select reference/unfused helpers, not decode reroute ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:13620)). `--quality` / `DS4_MTP_STRICT` selects `decode2_exact` only when `draft_n == 2` ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29017)).

3. **bit-exact-k4: conclusion sound, numbers overpessimistic.**  
   The `code_topk 0.41x` is not representative if measured with `DS4_DSPARK_VERIFY_DIST_PROBE=1`: that probe runs the batched verifier in addition to real sequential verify ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28660)). I would not base the verdict on it. The retained 8k exact-reuse numbers are better: `30.81/36.34 = 0.848`, `31.69/37.46 = 0.846`, `28.85/33.65 = 0.857`, mean ~`0.85x`, not `0.78x` as written. Full 300 is `32.673/39.020 = 0.837x` ([summary](/Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/mtp_verifier_engineering_and_phaseA.md:17)). Still NO-GO: below baseline and below +20%.

4. **“commit to novel-kernel build”: overclaiming.**  
   The threshold is real: `verify_ms(4) <= 50.45 ms`; floor `39-43 ms` would clear. But that assumes the future verifier is sublinear, bit-exact, reaches decode bandwidth, and lands with GPU drafter + anchor reuse stack. Those are not established; even the headroom doc calls K=4 decode-bandwidth reachability unproven ([headroom](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/lead08_stage_divergence/headroom_decomposition.md:63)).

## Missed Swap?

I do **not** see a missed swap that yields sublinear + bit-exact K=4.

`verify_suffix_tops` is sublinear but uses batch kernels and is not exact ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21631)). `decode2_exact` is exact but hardcoded N=2 and linear decode-kernel work ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21805)). Extending it to K=4 is a code change and remains linear; it is not the missing gate-clearing swap.

## Honest Framing

“NO-GO via swaps” survives.  
“Necessary and sufficient path; commit” does not.

Better verdict: **no existing config/kernel-selection swap clears the gate; a sublinear bit-exact verifier is plausible and worth a bounded build attempt, but GO remains unconfirmed until an end-to-end K=4 bit-exact verifier profiles at `<=50.5 ms`.**

## Final Assessment

The decision should be: **NO-GO via swaps → attempt exact-sublinear verifier build with hard exit gate.**  
Not: **gate cleared / commit to novel-kernel build as sufficient.**