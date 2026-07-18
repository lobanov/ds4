# Lead 08 iteration 20: Metal counter capability gate

Date: 2026-07-18. Status: **REDESIGN / tooling BLOCKED; audit COMMIT.** No
bandwidth-versus-compute attribution is made.

## Contract and preflight

T1 asks whether installed Instruments can distinguish DRAM/cache pressure,
dequant/ALU throughput, occupancy/stalls, or dispatch gaps on the production
fixed-K4 M3 verifier. The independent preflight returned **REDESIGN**:

- do not treat exported schemas as populated counters;
- compare an untraced fixed-K4 control with an identical trace;
- require resolved shader intervals and at least one populated discriminating
  counter family;
- require traced steady verifier timing within 5% of control; and
- use K1-to-K4 same-path scaling only after the capability gate passes. Raw M1
  decode is contextual and cannot establish the verifier's binding regime.

The default `Metal GPU Counters` instrument was first smoke-tested on
`/usr/bin/true`. It selected `Performance Limiters` and warned: `Selected
counter profile is not supported on target device`. The supported `Metal
System Trace` template was therefore tested on the real workload.

## Matched workload

Both arms use the same binary, model, prompt, fixed K=4, and full M3 runtime
composition. Dumps, stage synchronization, MTP profiling, and expert profiling
are disabled. The first K4 cycle is excluded as cold; the final K2 tail cycle
is excluded by selecting only `verify_n==4`. This retains 21 cycles per arm.

Untraced control:

```sh
env DS4_DSPARK_VERIFY_K=4 DS4_DSPARK_VERIFY_BATCHED=1 \
  DS4_DSPARK_ANCHOR_REUSE=1 DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 \
  DS4_DSPARK_DRAFT_METAL=1 DS4_DSPARK_DRAFT_METAL_STS=1 \
  DS4_DSPARK_TIMING=1 ./ds4-spec-bench --metal \
  -m /Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /Users/lobanov/Projects/ds4/gguf/dspark.gguf \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter20_viability.jsonl
```

Trace arm:

```sh
xcrun xctrace record --template 'Metal System Trace' --time-limit 15s \
  --output /tmp/lead08_iter20_k4_system.trace --no-prompt \
  --env DS4_DSPARK_VERIFY_K=4 --env DS4_DSPARK_VERIFY_BATCHED=1 \
  --env DS4_DSPARK_ANCHOR_REUSE=1 \
  --env DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT=1 \
  --env DS4_DSPARK_DRAFT_METAL=1 --env DS4_DSPARK_DRAFT_METAL_STS=1 \
  --env DS4_DSPARK_TIMING=1 --launch -- ./ds4-spec-bench --metal \
  -m /Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /Users/lobanov/Projects/ds4/gguf/dspark.gguf \
  --bulk-config issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl \
  --jsonl-out /tmp/lead08_iter20_k4_trace.jsonl
xcrun xctrace export --input /tmp/lead08_iter20_k4_system.trace --toc \
  --output /tmp/lead08_iter20_k4_toc.xml
xcrun xctrace export --input /tmp/lead08_iter20_k4_system.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-info"]' \
  --output /tmp/lead08_iter20_gpu-counter-info.xml
```

The other table exports use the same XPath with `gpu-counter-value`,
`metal-gpu-intervals`, `metal-shader-profiler-shader-list`, and
`metal-shader-profiler-intervals`.

## Capability result

| Gate | Result | Verdict |
|---|---:|---|
| discriminating counter names | only `RT Unit Active` | fail |
| counter value rows | 822,505; all formatted values `0.0000` | fail |
| resolved shader names | 119 | supporting only |
| shader interval rows | 0 | fail |
| GPU interval rows | 12,408 across system processes | not kernel attribution |
| command-buffer submission rows | 287 | supporting only |
| steady K4 verify median | 62.082 -> 66.754 ms | +7.526%, fail >5% |
| end-to-end throughput | 37.803 -> 35.703 t/s | -5.556% |

The trace contains real tables, but not the counter family or per-shader
intervals required by the contract. Its only counter is a ray-tracing activity
counter and every value is zero, as expected for this compute workload. The
timing perturbation independently fails the capability gate.

The 84 MiB trace and 400+ MiB bulk XML exports remain temporary and are not
retained. The compact CSV retains the decision-grade values.

## Decision and next falsifier

T1 is **BLOCKED** for hardware binding attribution through the available CLI
profiles. Full Xcode installation alone does not resolve the question. A custom
Instruments GUI template selecting an M5-supported counter set could unblock
it, but no claim may be made until DRAM/cache, ALU, or occupancy/stall values
are populated and trace perturbation is controlled.

Do not spend another iteration on K1/K4 traces with this template: the
capability prerequisite failed. The next self-contained discriminator is T2, a
same-kernel arithmetic-intensity separator that holds selected weights,
addresses, dispatch geometry, and physical rows fixed while changing only the
amount of register-resident dequant/FMA work. It requires its own preflight and
must be designed against compiler elimination before implementation.

The mandatory audit independently reproduced the unsupported-profile warning,
all counter/table counts, both 21-cycle medians, and the +7.526% / -5.556%
deltas. It confirmed the bounded T1/T2 wording and returned **COMMIT**.
