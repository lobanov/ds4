#!/usr/bin/env python3
"""Parse ds4 MTP speculative-decode stderr logs produced under
DS4_MTP_TIMING / DS4_MTP_SPEC_LOG / DS4_MTP_CONF_LOG and emit the Phase 0
T3/T4/T5 aggregate tables.

Usage:
    parse_spec_log.py LOGFILE [LOGFILE ...]

Each log is summarized independently. Output is TSV to stdout.
"""
import re
import sys
from collections import defaultdict

# e.g. "ds4: mtp timing micro drafted=2 committed=2 draft=2.197 ms snapshot=0.000 ms verify=42.653 ms total=44.946 ms"
TIMING_RE = re.compile(
    r"mtp timing (?P<path>\S+)"
    r"(?:\s+drafted=(?P<drafted>\d+))?"
    r"(?:\s+committed=(?P<committed>\d+))?"
    r"(?:\s+verified=(?P<verified>\d+))?"
    r"(?:\s+margin=(?P<margin>[\d.]+))?"
    r"(?:\s+threshold=(?P<thresh>[\d.]+))?"
    r"(?:\s+draft=(?P<us_draft>[\d.]+)\s+ms)?"
    r"(?:\s+snapshot=(?P<us_snapshot>[\d.]+)\s+ms)?"
    r"(?:\s+verify=(?P<us_verify>[\d.]+)\s+ms)?"
    r"(?:\s+prefix=(?P<us_prefix>[\d.]+)\s+ms)?"
    r"(?:\s+exact_replay=(?P<us_exact_replay>[\d.]+)\s+ms)?"
    r"(?:\s+replay=(?P<us_replay>[\d.]+)\s+ms)?"
    r"(?:\s+total=(?P<us_total>[\d.]+)\s+ms)?"
    r"(?:\s+noreplay=(?P<noreplay>\d+))?"
)

CONF_RE = re.compile(
    r"mtp conf drafted=(?P<drafted>\d+) committed=(?P<committed>\d+)\s+"
    r"mtp_top=(?P<top>-?\d+) runner=(?P<runner>-?\d+) margin=(?P<margin>[\d.]+)"
)

SUMMARY_RE = re.compile(r"generation:\s+([\d.]+)\s+t/s")
PREFILL_RE = re.compile(r"prefill:\s+([\d.]+)\s+t/s")


def parse(path):
    paths = defaultdict(int)
    drafts = []
    committed = []
    us_draft = us_verify = us_total = 0.0
    us_snapshot = us_prefix = us_replay = us_exact = 0.0
    n_timing = 0
    # acceptance by position (within drafts[])
    pos_trials = defaultdict(int)
    pos_accepts = defaultdict(int)
    conf_margins = []
    full = partial = miss_first = 0
    gen_tps = prefill_tps = None

    with open(path) as f:
        for line in f:
            m = TIMING_RE.search(line)
            if m:
                n_timing += 1
                p = m.group("path")
                paths[p] += 1
                d = int(m.group("drafted")) if m.group("drafted") else 0
                c = int(m.group("committed")) if m.group("committed") else 0
                drafts.append(d)
                committed.append(c)
                if m.group("us_draft"): us_draft += float(m.group("us_draft"))
                if m.group("us_verify"): us_verify += float(m.group("us_verify"))
                if m.group("us_total"): us_total += float(m.group("us_total"))
                if m.group("us_snapshot"): us_snapshot += float(m.group("us_snapshot"))
                if m.group("us_prefix"): us_prefix += float(m.group("us_prefix"))
                if m.group("us_replay"): us_replay += float(m.group("us_replay"))
                if m.group("us_exact_replay"): us_exact += float(m.group("us_exact_replay"))
                # classify accept outcome (committed = accepted draft tokens)
                if d >= 1:
                    if c >= d:
                        full += 1
                    elif c >= 1:
                        partial += 1
                    else:
                        miss_first += 1
                    # position-wise: positions 0..c-1 accepted, c..d-1 rejected
                    for k in range(d):
                        pos_trials[k] += 1
                        if k < c:
                            pos_accepts[k] += 1
                continue
            cm = CONF_RE.search(line)
            if cm:
                conf_margins.append(float(cm.group("margin")))
                continue
            if "spec miss first" in line:
                miss_first = miss_first  # counted from timing too; keep max below
            sm = SUMMARY_RE.search(line)
            if sm and gen_tps is None:
                gen_tps = float(sm.group(1))
            pm = PREFILL_RE.search(line)
            if pm and prefill_tps is None:
                prefill_tps = float(pm.group(1))

    if n_timing == 0:
        return None
    mean_drafted = sum(drafts) / n_timing
    mean_committed = sum(committed) / n_timing
    rows = []
    rows.append(("gen_tps", f"{gen_tps:.2f}" if gen_tps else "-"))
    rows.append(("prefill_tps", f"{prefill_tps:.2f}" if prefill_tps else "-"))
    rows.append(("cycles", str(n_timing)))
    rows.append(("mean_drafted", f"{mean_drafted:.2f}"))
    rows.append(("mean_committed", f"{mean_committed:.3f}"))
    rows.append(("full_accept", str(full)))
    rows.append(("partial_accept", str(partial)))
    rows.append(("miss_first", str(miss_first)))
    rows.append(("mean_us_draft", f"{us_draft/n_timing:.3f}"))
    rows.append(("mean_us_verify", f"{us_verify/n_timing:.3f}"))
    rows.append(("mean_us_snapshot", f"{us_snapshot/n_timing:.3f}"))
    rows.append(("mean_us_prefix", f"{us_prefix/n_timing:.3f}"))
    rows.append(("mean_us_replay", f"{us_replay/n_timing:.3f}"))
    rows.append(("mean_us_exact_replay", f"{us_exact/n_timing:.3f}"))
    rows.append(("mean_us_total", f"{us_total/n_timing:.3f}"))
    # acceptance by position
    pos_str = " ".join(
        f"{pos_accepts[k]}/{pos_trials[k]}" for k in sorted(pos_trials)
    )
    pos_rate = " ".join(
        f"{pos_accepts[k]/pos_trials[k]:.2f}" for k in sorted(pos_trials)
    )
    rows.append(("accept_by_position (acc/trial)", pos_str))
    rows.append(("accept_by_position (rate)", pos_rate))
    if conf_margins:
        rows.append(("mean_conf_margin", f"{sum(conf_margins)/len(conf_margins):.3f}"))
    rows.append(("path_mix", " ".join(f"{k}={v}" for k, v in sorted(paths.items()))))
    return rows


def main():
    for path in sys.argv[1:]:
        rows = parse(path)
        print(f"=== {path} ===")
        if rows is None:
            print("  (no mtp timing lines)")
            print()
            continue
        for k, v in rows:
            print(f"  {k:32s} {v}")
        print()


if __name__ == "__main__":
    main()
