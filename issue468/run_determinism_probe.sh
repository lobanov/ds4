#!/usr/bin/env bash
# Determinism probe (issue468/07_branch_b_options.md §1, §5.1).
#
# temp=0 greedy argmax is NOT bit-reproducible on Metal: GPU reduction order is
# nondeterministic, so two identical runs can diverge at near-tied tokens. This
# probe confirms/quantifies that at the token level, which feeds the B1-vs-B2
# choice:
#   - if baseline IS non-deterministic  -> B1's drift bound δ becomes δ+ε_baseline
#     (unmeasurable in isolation) -> strengthens the case for B2.
#   - the quality-baseline diff (diff_quality.py) separately uses the EXACT path
#     (--quality, same decode kernel as target-only) as the noise floor.
#
# Method: two target-only temp=0 runs of the SAME chat prompt, same seed, plain
# text to a file. Compare token streams with cmp/diff. Report divergence point
# and how many tokens agree before divergence (if any).
#
# Runs after the quality baseline releases the instance lock.
set -u
cd "$(dirname "$0")/.."
MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
OUT=issue468/baseline/determinism
mkdir -p "$OUT"
PROMPT="Explain how gradient descent works, step by step, then give two concrete examples."
# greedy, no MTP, plain text only, fixed token budget
COMMON=(--plain --temp 0 -n 512 --seed 1 --power 100)

echo "=== determinism probe $(date +%H:%M:%S): two identical target-only temp=0 runs ==="
for r in 1 2; do
  ./ds4 -m "$MODEL" "${COMMON[@]}" -p "$PROMPT" > "$OUT/run$r.txt" 2> "$OUT/run$r.log"
  echo "  run $r: $(wc -c < "$OUT/run$r.txt") bytes, $(wc -l < "$OUT/run$r.txt") lines"
done

echo
echo "=== byte-level comparison ==="
if cmp -s "$OUT/run1.txt" "$OUT/run2.txt"; then
  echo "IDENTICAL — target-only temp=0 reproduced bit-for-bit on this run pair."
  echo "(Caveat: non-determinism can be rare; identical here does not prove determinism.)"
else
  echo "DIFFERENT — confirms Metal temp=0 non-determinism."
  echo "  first differing byte:"
  cmp "$OUT/run1.txt" "$OUT/run2.txt" | head -1
  AGREE=$(cmp "$OUT/run1.txt" "$OUT/run2.txt" 2>/dev/null | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+')
  echo "  tokens/bytes in agreement before divergence: ${AGREE:-?}"
  echo "  diff (unified, first 30 lines):"
  diff -u "$OUT/run1.txt" "$OUT/run2.txt" | head -30
fi
echo "=== probe done $(date +%H:%M:%S) ==="
