#!/usr/bin/env python3
"""Lead 03 — sample 60 NEW prompts (20/source) from corpus_source for the broader capture.

Non-overlapping with Stage 2's 240 (excludes Stage 2's source_idx; ids 0080-0099 per
source vs Stage 2's 0000-0079). Same families (dolly/codealpaca/jsonex) + stratification
as sample_stage2_corpus.py, seed=43. Writes prompts/lead3_corpus/<id>.txt + manifest.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "corpus_source"
OUT = HERE / "prompts" / "lead3_corpus"
STAGE2_MAN = HERE / "prompts" / "stage2_corpus" / "manifest.json"
SEED = 43
PER_SOURCE = 20
ID_OFFSET = 80  # Stage 2 used 0000-0079; Lead 3 uses 0080-0099
MIN_CHARS, MAX_CHARS = 30, 8000
RNG = random.Random(SEED)


def filt(t):
    t = (t or "").strip()
    return bool(t) and MIN_CHARS <= len(t) <= MAX_CHARS


def form_dolly(r):
    i = (r.get("instruction") or "").strip(); c = (r.get("context") or "").strip()
    return f"{i}\n\n{c}" if c else i


def form_codealpaca(r):
    return (r.get("prompt") or "").strip()


def form_jsonex(r):
    i = (r.get("instruction") or "").strip(); t = (r.get("text") or "").strip()
    return f"{i}\n\n{t}" if t else i


def pick(rows, stratum_key, n, form_fn, used_idx):
    by = {}
    for i, r in enumerate(rows):
        if i in used_idx:
            continue
        text = form_fn(r)
        if not filt(text):
            continue
        s = str((r.get(stratum_key) or "none") if stratum_key else "all")
        by.setdefault(s, []).append(i)
    strata = sorted(by)
    per = n // len(strata); extra = n - per * len(strata)
    picked = []
    for k, s in enumerate(strata):
        take = per + (1 if k < extra else 0)
        idxs = by[s]; RNG.shuffle(idxs)
        for i in idxs[:take]:
            picked.append((i, form_fn(rows[i])))
    RNG.shuffle(picked)
    return picked[:n]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s2 = json.loads(STAGE2_MAN.read_text())
    used = {}
    for r in s2["prompts"]:
        used.setdefault(r["source"], set()).add(r["source_idx"])
    dolly = [json.loads(l) for l in (SRC / "databricks-dolly-15k.jsonl").open()]
    ca = pq.read_table(SRC / "codealpaca_python_train.parquet").to_pylist()
    je = pq.read_table(SRC / "json_extraction_train.parquet").to_pylist()
    picks = {
        "dolly": pick(dolly, "category", PER_SOURCE, form_dolly, used.get("dolly", set())),
        "codealpaca": pick(ca, None, PER_SOURCE, form_codealpaca, used.get("codealpaca", set())),
        "jsonex": pick(je, "topic", PER_SOURCE, form_jsonex, used.get("jsonex", set())),
    }
    records = []
    for source, pk in picks.items():
        for j, (src_idx, text) in enumerate(pk):
            pid = f"{source}_{ID_OFFSET + j:04d}"
            (OUT / f"{pid}.txt").write_text(text + "\n", encoding="utf-8")
            records.append({"prompt_id": pid, "source": source, "split": "lead3",
                            "file": f"prompts/lead3_corpus/{pid}.txt", "source_idx": src_idx,
                            "chars": len(text)})
    man = {"description": "Lead 03 broader capture (60 new prompts, non-overlapping with Stage 2).",
           "seed": SEED, "generation_length_tokens": 128, "n_prompts": len(records),
           "prompts": records}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"sampled {len(records)} prompts; by source:", dict(Counter(r["source"] for r in records)))
    print("ids:", sorted(set(r["prompt_id"] for r in records))[:3], "...",
          sorted(set(r["prompt_id"] for r in records))[-3:])
    # overlap check
    s2_ids = {r["prompt_id"] for r in s2["prompts"]}
    new_ids = {r["prompt_id"] for r in records}
    print("overlap with Stage 2 ids:", len(s2_ids & new_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
