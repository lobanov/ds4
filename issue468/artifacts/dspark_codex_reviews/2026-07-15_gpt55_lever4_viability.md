**Verdict:** anchor-reuse-for-Metal is viable. I do not see a remaining issue that invalidates the `38.02 t/s` result or the core `dspark_n_real` fix.

Findings:

1. The benchmark result looks sound for the valid paired subset. All compared files have the same 93 successful IDs. Filtering out every row where any mode ended early via EOS leaves 89 rows, and fixed Metal anchor-reuse is still essentially break-even: `37.92 t/s` vs plain `38.19 t/s`, paired ratio `0.993`. So the EOS caveat is not creating the win.

2. The 93-row subset is not performance-biased across modes, but it is not the full “300 prompt” corpus. The dropped rows are deterministic `frontier > prompt` errors shared by modes. Good for a paired lever decision; not enough to claim broad corpus coverage without fixing the manifest/frontiers.

3. The `dspark_n_real` fix is correct in the main path. With anchor reuse, `target_top` is seeded to `first_token` at [ds4.c:29818](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29818), so `drafts[0]` is the anchor and increments `verified` in the batched commit paths at [ds4.c:29888](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29888) and [ds4.c:29918](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29918). Therefore `metal_base_real + verified` at [ds4.c:30130](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:30130) is the right advance; the old `+ 1` double-counted the anchor.

4. Remaining state caveat: pure prefill does not appear to populate `dspark_metal_main_hidden`; it is normally filled by decode capture or fast-commit. The fixed artifact includes this first-cycle tax: first cycle averages `verified=1.02`, later cycles `verified=2.56`. It self-heals after the first accepted anchor, and correctness is protected by target verification, but for a polished default I would initialize `dspark_metal_main_hidden` after prefill or skip Metal anchor-reuse for the very first cycle.

5. Lever 5 is lower priority, but the “verify_n rose from 2.54 to 3.56” argument is mostly accounting. Anchor-reuse adds the always-verified anchor, so continuation scheduling is about `2.56`, basically the old Metal STS `2.54`. Draft-6 still looks marginal at this acceptance/cost point; I would keep lever 4, fix the first-cycle initialization robustness, and only then test draft-6/recalibration as an optional sweep.