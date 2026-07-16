## Verdict Per Claim

- **89.5% verify share: sound.** From `full_stack.jsonl`, prompt mean `verify=62.066 ms`, `cycle=69.380 ms`, so `62.066/69.380 = 89.46%`. A fixed `X` ms saving gives `X/(69.38-X)` relative gain; `X=10` gives `+16.84%` over current, `46.78 t/s`, or `+22.6%` over plain 38.16 t/s.

- **“Saving stacks on +4.9%”: arithmetically sound, scheduler model questionable.** STS chooses `verify_n` from confidence before verify in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29793), so cheaper verify may change `verify_n` and `verified_mean`. Sensitivity is material: around baseline, `+0.25` accepted tokens/cycle adds about `+8.9%` relative throughput; `+2 ms` extra cycle cost removes about `3.4%`.

- **Dense weight sharing but expert dequant not shared: sound.** Dense has multi-token r1 kernels in [metal/dense.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/dense.metal:912). Routed expert batch dispatches `z = n_tokens * n_selected`, then calls the IQ2XXS impl per `(token, expert-slot)` in [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1257) and [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:20945).

- **“190 GB/s means scattered access”: unsupported.** The profiling byte counter is **unique union expert bytes**, not physical per-pair traffic: see [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:12401). But the kernel still reloads per pair. So `1.14 GiB / 6.14 ms ~= 186 GiB/s` is an effective union-byte slope, not measured DRAM bandwidth.

- **“Expert stream is ~28 ms at 190 GB/s”: questionable.** The K3→K4 slope gives ~186 GiB/s on `code_8k`, but K4→K5 gives only ~56 GiB/s. Other prompts were ~88-135 GiB/s. That is not a stable bandwidth diagnosis, and current instrumentation does not separate IQ2XXS ALU/dequant, address divergence, occupancy, and memory traffic.

- **Two fusion mechanisms are incomplete.** Dequant de-dup and coalesced access are real possibilities, but missed/underweighted items are readahead overlap, physical pair-vs-unique duplication ceiling, down/sum6 reduction cost, STS reoptimization, and exactness fallback cost.

- **Exactness scope: correctly not solved.** The single-stage routed prototype does not make full greedy verify exact because attention/hidden state still diverge; `decode2_exact` is sequential N=2 only in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:22001). The 0.64% flip rate does not prove a cheap margin fallback; fallback frequency needs a top1-top2 margin histogram.

## Missed Avenues

1. **Batch selected readahead/overlap sweep.** Decode has selected readahead machinery in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:14596); batch FFN also has hooks around [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19466). Test flag sweeps before writing kernels. Prize: medium/high. Effort: low.

2. **Unique-expert grouped routed kernel including down, not just gate/up.** Current duplicate factor is roughly `pairs/unique`: K4 code case `24/19 = 1.26`, K5 `30/21 = 1.43`. Perfect de-dup ceiling is only ~21-30% of routed cost unless coalescing also improves. Prize: high if memory-bound. Effort: high.

3. **STS retune after cheaper verify.** Current fixed-X math assumes static acceptance. Rerun/simulate STS thresholds with altered verify cost curve. Prize impact: several percentage points. Effort: low/medium.

4. **Margin-guarded exact fallback.** Measure top1-top2 margins and batched-vs-exact deltas; simulate fallback thresholds. This may compose with the cost work better than full exact fusion. Effort: low/medium.

5. **Shared Q8 expert is mostly not a lever.** Shared expert already goes through batched Q8 paths in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19508) and [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:12795). Profile it, but do not expect large win.

6. **Attention→FFN activation fusion is low prize.** HC traffic is tiny: `2 * 4096 * 4 * 3.78 * 43 ~= 5.3 MiB` for one read/write boundary. Even with several intermediates, this is sub-ms scale, not a 10 ms lever.

7. **On-GPU argmax compare is already mostly done.** Verify top rows use GPU argmax/topk in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21907); remaining CPU compare/readback is tiny.

## Better/Cheaper Experiments

1. **Run existing stage profiles at fixed K=2..5.** Same benchmark as full-stack, add `DS4_MTP_VERIFY_PROFILE=1`, `DS4_MTP_VERIFY_EXPERT_PROFILE=1`, `DS4_METAL_LAYER_STAGE_PROFILE=1`, `DS4_METAL_MOE_ONE_STAGE_PROFILE=1`. Signal: routed gate/up/down ms, attention share, dense/share cost. This bounds the real prize before kernel work.

2. **Correlate routed stage time with physical pairs and unique experts.** Use selected profile stats plus `pairs = verify_n * 6`. If time tracks pairs, de-dup helps; if it tracks unique bytes poorly, coalescing/compute/counters needed.

3. **Use Instruments/Metal counters on current addr kernel vs decode id kernel.** Need real memory throughput, L2 hit rate, occupancy, ALU utilization. In-tree timings cannot prove bandwidth-bound.

4. **Readahead/overlap flag sweep.** Toggle selected batch addr/readahead/shared overlap flags and compare routed stage time. If this wins ms, it is cheaper than a fused kernel.

5. **Exactness margin probe.** Log top1/top2 margins for rows where batched and exact disagree or nearly disagree. Expected signal: fallback rate at margins like `0.25`, `0.5`, `1.0`, `1.75`.

6. **Only then build M=2 grouped prototype.** Prototype must include down/sum6 behavior or it will overstate speed and understate exactness risk.

## Probe-Design Issues

- The proposed “synthetic contiguous union-expert load” is not clearly runnable without a new kernel. Existing batch addr path still computes/dequants per pair. A Metal blit copy path exists, but that measures copy bandwidth, not shader gather/dequant behavior.

- Compute-vs-bandwidth split is not available from current logs. It needs Instruments/Xcode GPU counters or a deliberately altered kernel/data path.

- The `380/250 GB/s` thresholds are arbitrary. If current routed cost is really `28 ms @190 GB/s`, then `300 GB/s` already saves `~10.3 ms`; `250 GB/s` saves `~6.7 ms`; if compute-bound, bandwidth thresholds are irrelevant.

- Probe 1 being fast is not sufficient. A real fused verifier must dequant, apply SwiGLU, reduce down/sum6, preserve enough ordering/exactness, and interact with STS.

- Probe 4 should move earlier. Attention+dense share is the hard upper bound on routed-expert fusion prize.

## Leading Re-Assessment

The arithmetic prize is real, and the per-pair routed IQ2XXS dequant duplication is real. The weak link is the attribution: “190 GB/s because scattered access, recoverable to decode bandwidth” is not established by the current evidence.

I would downgrade this from GO to **conditional HOLD**. The one decisive experiment I would run first is the existing fixed-K stage profile sweep: prove routed gate/up/down is actually a ~25-30 ms isolated cost and that it scales with physical pair duplication or poor shader memory behavior. If that holds, build the grouped M=2 routed kernel. If not, kill the fused-kernel lead or redirect to scheduler/overlap/exactness work.