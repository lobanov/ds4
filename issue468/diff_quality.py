#!/usr/bin/env python3
"""Branch B precision diff — robust to Metal temp=0 nondeterminism.

Greedy argmax is not bit-reproducible on Metal (GPU reduction order), so a
single run is one sample. This tool reads MULTIPLE runs per config and reports:

  - per-config overall pass count (each run + mean)
  - per-case consensus (the modal answer across runs) and per-case stability
  - run-to-run flip rate per config  (the noise floor for that path)
  - batch-vs-exact-vs-reference per-case answer comparison

Interpretation:
  exact path (--quality) uses the same single-token decode kernel as target-only,
  so its flip rate IS the Metal nondeterminism noise floor. The batch path adds
  the batched verifier's logit drift. Branch B is quality-neutral for this
  metric iff batch's behavior is within exact's noise band.

Usage:
  diff_quality.py issue468/baseline/quality/*.out
Auto-groups files by config (target_only/batch/exact) and run index (rN).
The reference log is parsed the same way; name it target_only.* or pass it in.
"""
import re, sys, glob, os
from collections import Counter, defaultdict

CASE_RE = re.compile(
    r"^\s*(\d+)\s+(PASSED|FAILED|ERROR|SKIPPED)\s+\d+\s+\d+\s+\d+"
    r"\s+(\S+)\s+(\S+)\s+(.+)$"
)

def parse(path):
    cases = {}
    overall = None
    with open(path, errors="replace") as f:
        for line in f:
            m = CASE_RE.match(line)
            if m:
                cases[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
            mp = re.search(r"(\d+)/(\d+)\s+passed", line)
            if mp and overall is None:
                overall = (int(mp.group(1)), int(mp.group(2)))
    return overall, cases

def config_of(path):
    """batch.r1.out -> batch; exact.r2.log -> exact; target_only.out -> target_only"""
    base = os.path.basename(path).split(".")[0]
    base = re.sub(r"(_nothink_4096|_quality)$", "", base)  # normalize reference names
    base = base.replace("reference", "target_only")
    return base

def main():
    paths = sorted(sys.argv[1:]) if len(sys.argv) > 1 else \
            sorted(glob.glob("issue468/baseline/quality/*.out") +
                   glob.glob("issue468/baseline/quality/*.log"))
    if not paths:
        print("no input files"); sys.exit(1)

    # group by config
    by_cfg = defaultdict(list)  # config -> [(run_path, overall, cases)]
    for p in paths:
        cfg = config_of(p)
        ov, cases = parse(p)
        if cases:
            by_cfg[cfg].append((p, ov, cases))

    print("=" * 72)
    print("PER-CONFIG OVERALL PASS COUNT")
    print("=" * 72)
    print(f"{'config':<14} {'runs':<5} {'pass per run':<24} {'mean':<6}")
    for cfg in sorted(by_cfg):
        runs = by_cfg[cfg]
        passes = [ov[0] for (_, ov, _) in runs if ov]
        per = ", ".join(str(x) for x in passes) or "?"
        mean = f"{sum(passes)/len(passes):.1f}" if passes else "?"
        n = ov[1] if runs and runs[0][1] else "?"
        print(f"{cfg:<14} {len(runs):<5} {per:<24} {mean:<6} / {n}")

    print()
    print("=" * 72)
    print("RUN-TO-RUN ANSWER FLIPS PER CONFIG  (= noise floor for that path)")
    print("=" * 72)
    print(f"{'config':<14} {'runs':<5} {'cases w/ >1 distinct answer':<30} {'flip rate'}")
    cfg_flips = {}
    for cfg in sorted(by_cfg):
        runs = by_cfg[cfg]
        if len(runs) < 2:
            print(f"{cfg:<14} {len(runs):<5} (need >=2 runs to measure flips)")
            continue
        all_idx = sorted(set().union(*[set(c) for (_, _, c) in runs]))
        n_flip = 0
        for idx in all_idx:
            ans = [c.get(idx, ("?","?","?"))[1] for (_, _, c) in runs]
            if len(set(ans)) > 1:
                n_flip += 1
        cfg_flips[cfg] = (n_flip, len(all_idx))
        print(f"{cfg:<14} {len(runs):<5} {n_flip:<30} {n_flip}/{len(all_idx)} ({100*n_flip/len(all_idx):.0f}%)")

    print()
    print("=" * 72)
    print("THE BRANCH B TEST: batch drift vs exact noise floor")
    print("=" * 72)
    exact = by_cfg.get("exact", [])
    batch = by_cfg.get("batch", [])
    if "exact" in cfg_flips and "batch" in cfg_flips:
        ef, en = cfg_flips["exact"]; bf, bn = cfg_flips["batch"]
        print(f"  exact run-to-run flips (noise floor): {ef}/{en} ({100*ef/en:.0f}%)")
        print(f"  batch run-to-run flips (noise+drift): {bf}/{bn} ({100*bf/bn:.0f}%)")
        if bf <= ef + 1:
            print("  -> batch drift is WITHIN the Metal nondeterminism noise band.")
            print("     Branch B is QUALITY-NEUTRAL for this metric (on this sample).")
        else:
            print("  -> batch flip rate EXCEEDS the noise floor.")
            print("     The batch verifier's logit drift is likely answer-changing. Investigate.")
    else:
        print("  (need >=2 runs each of exact and batch to compute this)")

    # per-case consensus comparison: reference vs batch consensus vs exact consensus
    print()
    print("=" * 72)
    print("PER-CASE: target_only vs batch(consensus) vs exact(consensus)")
    print("=" * 72)
    if not ("target_only" in by_cfg and batch and exact):
        print("  (needs target_only + batch + exact to compare; skipping)")
        return
    def consensus(runs):
        all_idx = sorted(set().union(*[set(c) for (_, _, c) in runs]))
        out = {}
        for idx in all_idx:
            ans = Counter(c.get(idx, ("?","?","?"))[1] for (_, _, c) in runs)
            out[idx] = ans.most_common(1)[0][0]
        return out

    tgt = consensus(by_cfg["target_only"]) if "target_only" in by_cfg else {}
    bat_c = consensus(batch) if batch else {}
    ex_c = consensus(exact) if exact else {}
    all_idx = sorted(set(tgt) | set(bat_c) | set(ex_c))
    diffs = [(i, tgt.get(i,"?"), bat_c.get(i,"?"), ex_c.get(i,"?"))
             for i in all_idx
             if not (tgt.get(i,"?")==bat_c.get(i,"?")==ex_c.get(i,"?"))]
    if diffs:
        print(f"  {'case':>4} {'target':<10} {'batch':<10} {'exact':<10}")
        for i, t, b, e in diffs:
            mark = ""
            if b != t and e == t: mark = "  <- batch-only flip (drift?)"
            elif e != t and b == t: mark = "  <- exact-only flip (noise)"
            elif b != t and e != t and b == e: mark = "  <- both flip same way"
            print(f"  {i:>4} {t:<10} {b:<10} {e:<10}{mark}")
    else:
        print("  (all three paths agree on every case)")

if __name__ == "__main__":
    main()
