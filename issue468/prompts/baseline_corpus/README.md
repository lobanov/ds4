# Long-context prompt corpus

Current active prompt corpus for long-input DSpark vs baseline benchmarking.

Families:
- `code_{4k,8k,16k}.txt` — repository-style code completion with ~500-token target output
- `synthesis_{4k,8k,16k}.txt` — long operations archive summarized as minified JSON output
- `grounded_{4k,8k,16k}.txt` — grounded long-form continuation with ~500-token target output

Lengths are trimmed against actual token counts measured with the local `ds4` tokenizer.
Files preserve natural section and line boundaries for review.
See `manifest.json` for retained sizes.
