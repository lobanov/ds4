#!/usr/bin/env bash
# Activity 2 — golden-parity test for the multi-layer capture patch (RESEARCH INSTRUMENTATION).
# Verifies that a single multi-layer pass (LAYER=40,41,42) produces byte-identical
# per-layer hidden-state dumps and topk to dedicated single-layer passes, on the
# SAME ds4 binary (isolates the comma-list patch from any binary drift).
# Usage: ./run_stage2_capture_golden.sh
set -u
DS4=/Users/lobanov/Projects/ds4/ds4
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
cd /Users/lobanov/Projects/ds4   # ds4 loads metal/*.metal relative to cwd
CORPUS=/Users/lobanov/Projects/ds4-dspark-research/issue468/prompts/stage2_corpus
LIST=/tmp/gold_list.txt
OUT=/tmp/gold
TOK=8; CTX=4096; SEED=2; TOPK=128
printf 'dolly_0000\t%s/dolly_0000.txt\ncodealpaca_0000\t%s/codealpaca_0000.txt\n' "$CORPUS" "$CORPUS" > "$LIST"
rm -rf "$OUT"; mkdir -p "$OUT"

run () {  # $1=label $2=layers
  echo "=== capture arm: $1 (layers=$2) ===" >&2
  "$DS4" --metal -m "$MODEL" --capture-dataset "$LIST" --capture-out "$OUT/$1" \
    --capture-layers "$2" --tokens "$TOK" --ctx "$CTX" --temp 0 --seed "$SEED" \
    --logprobs-top-k "$TOPK" 2>"$OUT/$1.err" >/dev/null
}

run multi 40,41,42
run l40   40
run l41   41
run l42   42

python3 - "$OUT" <<'PY'
import sys, pathlib, json, hashlib
out = pathlib.Path(sys.argv[1])
def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
prompts = ["dolly_0000", "codealpaca_0000"]
layers = [40, 41, 42]
fails = []
# topk parity: multi == each single (deterministic greedy)
for pid in prompts:
    multi_tk = md5(out/"multi"/f"{pid}.topk.json")
    for arm in ("l40","l41","l42"):
        h = md5(out/arm/f"{pid}.topk.json")
        if h != multi_tk:
            fails.append(f"topk {pid} multi vs {arm}: {multi_tk} != {h}")
# per-layer .bin parity: multi's layer L == single L's layer L, every generated position
for pid in prompts:
    for L in layers:
        multi_files = sorted((out/"multi").glob(f"{pid}_hc_ffn_post-{L}_pos*.bin"))
        sing_files  = sorted((out/f"l{L}").glob(f"{pid}_hc_ffn_post-{L}_pos*.bin"))
        names_m = [f.name for f in multi_files]
        names_s = [f.name for f in sing_files]
        if names_m != names_s:
            fails.append(f"layer {L} {pid}: file-set mismatch multi={len(names_m)} single={len(names_s)}")
            continue
        for fm, fs in zip(multi_files, sing_files):
            if fm.name != fs.name or md5(fm) != md5(fs):
                fails.append(f"layer {L} {pid} {fm.name}: content mismatch")
print(f"compared: {len(prompts)} prompts x {len(layers)} layers; failures: {len(fails)}")
for f in fails[:20]:
    print("  FAIL:", f)
print("GOLDEN PARITY:", "PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
PY
