#!/usr/bin/env python3
"""Run the retained temp>0 logit/distribution parity check for anchor reuse.

This wraps `ds4_test --mtp-temp-logit-parity`, captures its per-prompt summary
lines, and writes a compact JSON artifact for the dossier.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "issue468/artifacts/mtp_temp_distribution_compare"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_RE = re.compile(
    r"ds4-test: mtp-temp-logit-parity prompt=(\S+) temp=([0-9.]+) steps=(\d+) eligible=(\d+) "
    r"max_abs=([0-9.eE+-]+) rms=([0-9.eE+-]+) sampled_lp_diff=([0-9.eE+-]+)")
TOTAL_RE = re.compile(
    r"ds4-test: mtp-temp-logit-parity total_steps=(\d+) eligible=(\d+) "
    r"max_abs=([0-9.eE+-]+) rms=([0-9.eE+-]+) sampled_lp_diff=([0-9.eE+-]+)")


def main() -> int:
    cmd = ["./ds4_test", "--mtp-temp-logit-parity"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (OUT / "stdout.txt").write_text(proc.stdout)
    (OUT / "stderr.txt").write_text(proc.stderr)

    rows = []
    total = None
    for line in proc.stderr.splitlines():
        m = SUMMARY_RE.search(line)
        if m:
            rows.append({
                "prompt_label": m.group(1),
                "temperature": float(m.group(2)),
                "steps": int(m.group(3)),
                "eligible_steps": int(m.group(4)),
                "max_abs": float(m.group(5)),
                "rms": float(m.group(6)),
                "sampled_lp_diff": float(m.group(7)),
            })
            continue
        m = TOTAL_RE.search(line)
        if m:
            total = {
                "total_steps": int(m.group(1)),
                "eligible_steps": int(m.group(2)),
                "max_abs": float(m.group(3)),
                "rms": float(m.group(4)),
                "sampled_lp_diff": float(m.group(5)),
            }

    out = {
        "returncode": proc.returncode,
        "rows": rows,
        "summary": total,
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
