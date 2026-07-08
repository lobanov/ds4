#!/usr/bin/env python3
"""Activity 1 — sample the Stage 2 prompt corpus (RESEARCH INSTRUMENTATION).

Samples ~60 train + ~20 eval prompts from each of the three downloaded datasets
(Dolly-15k, CodeAlpaca-Python, json-extraction) into issue468/prompts/stage2_corpus/,
stratified (Dolly by category, json-extraction by topic, CodeAlpaca random), seeded
for reproducibility. Prompt formation (raw text; ds4 applies the chat template at
capture time): Dolly = instruction(+context); CodeAlpaca = prompt; json-extraction =
instruction + text. Writes one .txt per prompt + a manifest.json.

The 10 exactness prompts are referenced as a locked legacy benchmark (unchanged).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent  # issue468/
SRC = HERE / "data" / "corpus_source"
OUT = HERE / "prompts" / "stage2_corpus"
SEED = 42
PER_SOURCE = 80          # 60 train + ~20 eval
TRAIN_FRAC = 0.75
MAX_PROMPT_CHARS = 8000  # keep prompt well under ctx (leave room for 128 gen)
MIN_PROMPT_CHARS = 30    # drop junk / trivially-short prompts (e.g. typos, 1-liners)
RNG = random.Random(SEED)


def filt(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and MIN_PROMPT_CHARS <= len(t) <= MAX_PROMPT_CHARS


def form_dolly(row: dict) -> str:
    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("context") or "").strip()
    return f"{instr}\n\n{ctx}" if ctx else instr


def form_codealpaca(row: dict) -> str:
    return (row.get("prompt") or "").strip()


def form_jsonex(row: dict) -> str:
    instr = (row.get("instruction") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{instr}\n\n{text}" if text else instr


def stratified_pick(rows: list[dict], stratum_key, n: int, form_fn) -> list[tuple[int, str]]:
    """Pick n rows (evenly across strata), returning (original_index, text)."""
    by_stratum: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        text = form_fn(r)
        if not filt(text):
            continue
        s = str((r.get(stratum_key) or "none") if stratum_key else "all")
        by_stratum.setdefault(s, []).append(i)
    strata = sorted(by_stratum)
    per = n // len(strata)
    extra = n - per * len(strata)
    picked: list[tuple[int, str]] = []
    for k, s in enumerate(strata):
        take = per + (1 if k < extra else 0)
        idxs = by_stratum[s]
        RNG.shuffle(idxs)
        for i in idxs[:take]:
            picked.append((i, form_fn(rows[i])))
    RNG.shuffle(picked)
    return picked[:n]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # --- Dolly (jsonl) ---
    dolly = [json.loads(l) for l in (SRC / "databricks-dolly-15k.jsonl").open()]
    dolly_pick = stratified_pick(dolly, "category", PER_SOURCE, form_dolly)
    # --- CodeAlpaca (parquet) ---
    ca = pq.read_table(SRC / "codealpaca_python_train.parquet").to_pylist()
    ca_pick = stratified_pick(ca, None, PER_SOURCE, form_codealpaca)
    # --- json-extraction (parquet) ---
    je = pq.read_table(SRC / "json_extraction_train.parquet").to_pylist()
    je_pick = stratified_pick(je, "topic", PER_SOURCE, form_jsonex)

    def emit(source: str, stratum_key, picks, rows):
        n_train = n_eval = 0
        n_train_target = round(len(picks) * TRAIN_FRAC)  # deterministic count split
        for idx, (src_idx, text) in enumerate(picks):
            split = "train" if idx < n_train_target else "eval"
            pid = f"{source}_{idx:04d}"
            fpath = OUT / f"{pid}.txt"
            fpath.write_text(text + "\n", encoding="utf-8")
            row = rows[src_idx]
            stratum = (row.get(stratum_key) or "none") if stratum_key else None
            records.append({
                "prompt_id": pid,
                "source": source,
                "split": split,
                "file": f"prompts/stage2_corpus/{pid}.txt",
                "source_idx": src_idx,
                "stratum": stratum,
                "chars": len(text),
                "tok_est": max(1, len(text) // 4),
            })
            if split == "train":
                n_train += 1
            else:
                n_eval += 1
        return n_train, n_eval

    stats = {}
    stats["dolly"] = emit("dolly", "category", dolly_pick, dolly)
    stats["codealpaca"] = emit("codealpaca", None, ca_pick, ca)
    stats["jsonex"] = emit("jsonex", "topic", je_pick, je)

    manifest = {
        "description": "Stage 2 fine-tune PoC prompt corpus (Activity 1).",
        "seed": SEED,
        "generation_length_tokens": 128,  # applied at capture (Activity 3)
        "per_source_target": PER_SOURCE,
        "train_frac": TRAIN_FRAC,
        "prompt_formation": {
            "dolly": "instruction (+ context if present)",
            "codealpaca": "prompt",
            "jsonex": "instruction + text",
        },
        "legacy_benchmark": {
            "dir": "prompts/exactness_small_corpus/",
            "manifest": "prompts/exactness_small_corpus/manifest.json",
            "note": "10 exactness prompts; locked legacy benchmark (unchanged).",
        },
        "counts": {
            "total": len(records),
            "train": sum(1 for r in records if r["split"] == "train"),
            "eval": sum(1 for r in records if r["split"] == "eval"),
            "by_source": {s: {"train": t, "eval": e} for s, (t, e) in stats.items()},
        },
        "prompts": records,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["counts"], indent=2))
    # stratification check
    from collections import Counter
    for source, key in [("dolly", "category"), ("jsonex", "topic")]:
        c = Counter(r["stratum"] for r in records if r["source"] == source)
        print(f"{source} strata ({key}):", dict(c))
    # split + length sanity
    lens = [r["chars"] for r in records]
    import statistics as st
    print(f"prompt chars: min {min(lens)} median {int(st.median(lens))} max {max(lens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
