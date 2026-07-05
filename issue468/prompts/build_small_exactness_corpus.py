from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exactness_small_corpus"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT.parent.parent / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DS4 = ROOT.parent.parent / "ds4"
TARGET_MIN = 100
TARGET_MAX = 200


def count_tokens(text: str) -> int:
    with NamedTemporaryFile("w", suffix=".txt", delete=True) as tmp:
        tmp.write(text)
        tmp.flush()
        proc = subprocess.run(
            [str(DS4), "--backend", "metal", "-m", str(MODEL), "--dump-tokens", "--prompt-file", tmp.name],
            check=True,
            capture_output=True,
            text=True,
        )
    nums = re.findall(r"-?\d+", proc.stdout)
    return len(nums)


def write_prompt(name: str, text: str, manifest: dict[str, dict[str, object]]) -> None:
    path = OUT / f"{name}.txt"
    path.write_text(text)
    manifest[name] = {
        "file": str(path.relative_to(ROOT.parent)),
        "tokens": count_tokens(text),
        "chars": len(text),
        "words": len(text.split()),
    }


def main() -> None:
    manifest: dict[str, dict[str, object]] = {}

    prompts = {
        "code_sort_pairs": """Repository note:
Implement `sort_pairs(pairs)`.

Rules:
- input: list of `(name, count)` tuples
- output: a new list sorted by descending count, then ascending name
- keep the implementation standalone
- avoid custom classes
- include a short docstring and two tiny tests in the file
- output code only
""",
        "code_topk": """Task:
Write a clean implementation of `top_k_frequent(nums, k)`.

Requirements:
- use plain Python only
- deterministic tie-breaking by ascending integer value
- include a short module docstring
- include a tiny `__main__` demo
- output code only
""",
        "code_histogram": """Repository task:
Add `build_histogram(values)` that returns a list of `(value, count)` pairs.

Style constraints:
- prefer readability over cleverness
- return plain dict/list data only
- stable deterministic ordering by value
- include two compact tests
- output Python code only
""",
        "grounded_observatory": """Source notes:
- Mara returned to the east dome before dawn.
- Hard wind scraped the copper catwalk.
- The signal lamp stayed dark through the sleet.
- The frost on the lens housing had thickened overnight.
- She carried a canvas tool roll and a frosted logbook.

Task:
Write the next 2 short paragraphs of the expedition log.
Constraints:
- third-person past tense
- no dialogue
- keep the observatory setting grounded in the notes
- end with a small physical action
""",
        "grounded_archive": """Source notes:
- The lower stairwell smelled of cold iron and wet rope.
- A brass eyepiece had been wrapped in cloth near the ledger desk.
- The mountain lights appeared only in brief gaps in the fog.
- A faint vibration had been marked above bracket 4.

Task:
Continue the scene in 2 short paragraphs.
Constraints:
- grounded descriptive prose
- no new characters
- no dialogue
- keep all details consistent with the notes
""",
        "grounded_repair": """Source notes:
- Wind pressed snow against the snow gate.
- Mara checked the dome gearing by hand.
- The maintenance ledger mentioned intermittent drag at the meridian stair.
- The oak transit case had been left shut beside the wall.

Task:
Write a short continuation of the repair log in 2 paragraphs.
Constraints:
- third-person past tense
- concrete physical details only
- no dialogue
""",
        "synthesis_ops_json": """Operations notes:
facility=North Basin status=open issue=label mismatch units_held=18
facility=Atlas Dock status=contained issue=carrier delay units_held=9
facility=Lumen Hub status=open issue=sensor drift units_held=14
facility=Cedar Spur status=open issue=packing backlog units_held=11

Task:
Return one minified JSON object only.
Schema:
{"overview": string, "facility_status": [{"site": string, "status": string, "issue": string}], "recommended_actions": [string]}
Constraints:
- valid minified JSON only
- no markdown
- keep `overview` concise
- include all listed facilities
""",
        "synthesis_timeline_json": """Archive:
2026-04-03T05:10Z North Basin label mismatch found on lane 2
2026-04-03T07:40Z Atlas Dock held westbound pallet group for recount
2026-04-03T09:15Z Lumen Hub flagged sensor drift on scanner rack
2026-04-03T11:05Z Cedar Spur backlog reduced after relabel pass

Task:
Return one minified JSON object only.
Schema:
{"timeline": [{"stamp": string, "event": string}], "summary": string}
Constraints:
- valid minified JSON only
- preserve timestamp order
- no extra keys
""",
        "synthesis_incident_json": """Incident snippets:
- Juniper Yard: power fluctuation, manual review in progress
- Morrow Annex: seal failure, isolate affected cartons
- Atlas Dock: carrier delay, reroute coastal lane
- North Basin: label mismatch, second count requested

Task:
Produce one minified JSON object only.
Schema:
{"sites": [{"site": string, "issue": string, "action": string}], "open_questions": [string]}
Constraints:
- valid minified JSON only
- no markdown
- concise values
""",
        "mixed_exactness_smoke": """Small mixed prompt for exactness smoke testing.

Part A:
Write a tiny Python helper `pair_counts(items)` that returns sorted `(item, count)` pairs.

Part B:
After the code block, add 2 sentences describing a technician crossing a frozen observatory catwalk in hard wind.

Constraints:
- keep the code simple
- deterministic ordering for ties
- prose must be grounded and concrete
- no dialogue
""",
    }

    for name, text in prompts.items():
        write_prompt(name, text, manifest)

    readme = """# Small exactness-debug corpus

This corpus is for fast-turnaround DSpark/plain exactness testing.

Contract:
- 10 prompts total
- each prompt is intended to stay roughly in the 100-200 token range when tokenized by the local ds4 tokenizer
- this corpus is for debugging and faster harness iteration only
- it does not replace the 9-prompt long-context proof corpus

See `manifest.json` for retained token counts.
"""
    (OUT / "README.md").write_text(readme)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
