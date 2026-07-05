# Small exactness-debug corpus

This corpus is for fast-turnaround DSpark/plain exactness testing.

Contract:
- 10 prompts total
- prompts are intentionally much smaller than the long-context proof corpus
- most prompts land near the 100–200 token target range under the local ds4 tokenizer
- a few structured JSON prompts are somewhat larger because schema instructions add token overhead
- this corpus is for debugging and faster harness iteration only
- it does not replace the 9-prompt long-context proof corpus

Files:
- `code_sort_pairs.txt`
- `code_topk.txt`
- `code_histogram.txt`
- `grounded_observatory.txt`
- `grounded_archive.txt`
- `grounded_repair.txt`
- `synthesis_ops_json.txt`
- `synthesis_timeline_json.txt`
- `synthesis_incident_json.txt`
- `mixed_exactness_smoke.txt`

See `manifest.json` for retained token counts.
