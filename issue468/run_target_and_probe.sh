#!/usr/bin/env bash
# Fresh target-only control + determinism probe (issue468/07_branch_b_options.md §5.1).
#
# Two non-MTP target-only runs of the 92-case eval, identical conditions to the
# batch/exact baseline (greedy, --nothink, -n 4096, --seed 1). These are the
# cleanest control:
#   1. Reproduce the reference 67/92 -> rules out model/code drift.
#   2. The two runs' per-case differences = the pure target-only noise floor
#      (no MTP loaded at all, so no MTP buffers touching the path).
#
# Then the token-level determinism probe: two identical target-only temp=0 runs
# of one chat prompt, cmp the token streams.
set -u
cd "$(dirname "$0")/.."
MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
OUT=issue468/baseline/quality
DET=issue468/baseline/determinism
mkdir -p "$OUT" "$DET"
COMMON=(--plain --nothink -n 4096 --seed 1 --questions 92 --power 100)

echo "=== [target.r1] $(date +%H:%M:%S) ==="
./ds4-eval -m "$MODEL" "${COMMON[@]}" --trace "$OUT/target.r1.trace" \
     > "$OUT/target.r1.out" 2> "$OUT/target.r1.log"
echo "[target.r1] $(grep -E 'passed' "$OUT/target.r1.out" | tail -1)"

echo "=== [target.r2] $(date +%H:%M:%S) ==="
./ds4-eval -m "$MODEL" "${COMMON[@]}" --trace "$OUT/target.r2.trace" \
     > "$OUT/target.r2.out" 2> "$OUT/target.r2.log"
echo "[target.r2] $(grep -E 'passed' "$OUT/target.r2.out" | tail -1)"

echo
echo "=== DETERMINISM PROBE $(date +%H:%M:%S) ==="
PROMPT="Explain how gradient descent works, step by step, then give two concrete examples."
for r in 1 2; do
  ./ds4 -m "$MODEL" --plain --temp 0 -n 512 --seed 1 --power 100 -p "$PROMPT" \
        > "$DET/run$r.txt" 2> "$DET/run$r.log"
  echo "  probe run $r: $(wc -c < "$DET/run$r.txt") bytes"
done
if cmp -s "$DET/run1.txt" "$DET/run2.txt"; then
  echo "  IDENTICAL (bit-reproducible this pair; non-determinism can be rare)"
else
  echo "  DIFFERENT (confirms Metal temp=0 non-determinism at token level)"
  echo "  first divergence: $(cmp "$DET/run1.txt" "$DET/run2.txt" | head -1)"
fi
echo "=== ALL DONE $(date +%H:%M:%S) ==="
