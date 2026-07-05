from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "baseline_corpus"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT.parent.parent / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DS4 = ROOT.parent.parent / "ds4"

TARGET_TOKENS = {
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
}
TOLERANCE = 64


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


def join_sections(header: str, sections: list[str], footer: str, n: int) -> str:
    body = "\n\n".join(sections[:n]).rstrip()
    text = header.rstrip() + "\n\n" + body + "\n\n" + footer.strip() + "\n"
    return text


def trim_sections_to_token_target(header: str, sections: list[str], footer: str, target_tokens: int) -> tuple[str, int, int]:
    lo, hi = 1, len(sections)
    best_text = join_sections(header, sections, footer, len(sections))
    best_count = count_tokens(best_text)
    best_n = len(sections)
    best_err = abs(best_count - target_tokens)

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = join_sections(header, sections, footer, mid)
        count = count_tokens(candidate)
        err = abs(count - target_tokens)
        if err < best_err:
            best_text, best_count, best_n, best_err = candidate, count, mid, err
        if err <= TOLERANCE:
            return candidate, count, mid
        if count > target_tokens:
            hi = mid - 1
        else:
            lo = mid + 1

    return best_text, best_count, best_n


def build_code_parts() -> tuple[str, list[str], str]:
    header = '''Repository snapshot follows. Use it as the only source of style, naming, and local conventions.

Goal:
Write the missing implementation for the final target module.

Output contract:
- output Python code only
- write a complete implementation, not notes
- include a short module docstring
- include 3 compact tests in the same file
- include a tiny __main__ demo
- target roughly 90-130 lines of code
- prefer readability over cleverness
- stay consistent with the utility patterns shown in the repository snapshot

Repository snapshot:
'''
    sections: list[str] = []
    module_names = [
        "cache_keys", "render_pairs", "histograms", "window_stats", "scan_rules",
        "normalize_tags", "priority_queue", "text_slices", "dedupe_index", "audit_rows",
    ]
    for i in range(1, 900):
        name = module_names[i % len(module_names)]
        sections.append(f'''### file: analytics/{name}_{i:03d}.py
"""Helper utilities for batch reporting task {i}."""
from __future__ import annotations

def stable_key_{i}(record):
    parts = [record.get("group", ""), record.get("region", ""), record.get("sku", "")]
    return "::".join(str(p).strip().lower() for p in parts)

def choose_rank_{i}(count, score, penalty):
    base = (count * 7) + (score * 3) - penalty
    return max(base, 0)

def summarize_window_{i}(rows):
    total = 0
    distinct = set()
    for row in rows:
        total += int(row.get("count", 0))
        distinct.add(row.get("sku", ""))
    return {{"total": total, "distinct": len(distinct)}}

# style note: helpers return plain dict/list data, avoid custom classes.
# style note: callers sort with key=(-count, name) when deterministic order matters.
''')
        sections.append(f'''### file: tests/test_{name}_{i:03d}.py
from analytics.{name}_{i:03d} import stable_key_{i}, choose_rank_{i}

def test_stable_key_{i}():
    row = {{"group": "ops", "region": "west", "sku": "NW-{100+i}"}}
    assert stable_key_{i}(row) == "ops::west::nw-{100+i}"

def test_choose_rank_{i}():
    assert choose_rank_{i}(4, 3, 2) == 35
''')
        if i % 3 == 0:
            sections.append(f'''### file: docs/notes_{i:03d}.md
- reporting batch {i} normalized duplicate SKU rows before ranking
- small helper functions favored over inheritance-heavy abstractions
- if frequencies tie, deterministic order uses ascending integer or lexical order
- repository code generally avoids external dependencies and keeps demos tiny
''')
    footer = '''### file: analytics/top_k_frequent.py
"""Implement the missing public helper for ranked frequency extraction."""
from __future__ import annotations

def top_k_frequent(nums, k):
    # TODO: implement using the same repository style.
    raise NotImplementedError

Task:
Complete `top_k_frequent(nums, k)` as a clean standalone module that matches the repository style above.
Remember:
- output code only
- include docstring, 3 compact tests, and a tiny demo
- no external dependencies
'''
    return header, sections, footer


def build_synthesis_parts() -> tuple[str, list[str], str]:
    header = '''You are given a long operations archive compiled from shipping notes, service incidents,
facility updates, and SKU-level exception records. Use only the archive below.

Task:
Return a single minified JSON object only.

Required schema:
{
  "overview": string,
  "facility_status": [{"site": string, "status": string, "dominant_issue": string, "next_step": string}],
  "sku_impacts": [{"sku": string, "severity": string, "issue": string, "action": string}],
  "timeline": [{"stamp": string, "event": string}],
  "recommended_actions": [string],
  "unresolved_questions": [string]
}

Output constraints:
- valid minified JSON only
- no markdown
- no explanatory text outside JSON
- `overview` should be 90-140 words
- include 6 facility_status objects
- include 8 sku_impacts objects
- include 8 timeline objects
- include 6 recommended_actions strings
- include 4 unresolved_questions strings
- keep values concise but specific
- total output should land around 450-650 tokens

Operations archive:
'''
    sites = ["North Basin", "Juniper Yard", "Atlas Dock", "Cedar Spur", "Lumen Hub", "Morrow Annex"]
    issues = ["power fluctuation", "seal failure", "label mismatch", "carrier delay", "sensor drift", "packing backlog"]
    sku_prefix = ["NW", "JN", "AT", "CD", "LM", "MR"]
    sections: list[str] = []
    for i in range(1, 1200):
        site = sites[i % len(sites)]
        issue = issues[i % len(issues)]
        sku = f"{sku_prefix[i % len(sku_prefix)]}-{100 + i:03d}"
        status = "open" if i % 4 else "contained"
        route = "west" if i % 2 else "coastal"
        immediate = "manual review" if i % 3 else "reroute and relabel"
        delay = "late truck" if i % 5 else "scanner reset"
        confidence = "high" if i % 6 else "moderate"
        route_code = "W2" if i % 2 else "C4"
        sections.append(f'''Record {i:03d}
facility={site}
stamp=2026-04-{(i % 28) + 1:02d}T{(i * 3) % 24:02d}:{(i * 7) % 60:02d}Z
sku={sku}
status={status}
issue={issue}
units_held={12 + (i % 19)}
route={route}
note=inspection batch {i} found {issue} affecting pallet lane {(i % 11) + 1}
immediate_action={immediate}
''')
        if i % 2 == 0:
            sections.append(f'''Service memo {i:03d}
site={site}
sku={sku}
queue_cause={delay}
recovery_confidence={confidence}
action=request another count before release
''')
        if i % 5 == 0:
            sections.append(f'''Supervisor note {i:03d}
sku={sku}
repeat_issue_watch={issue}
route_code={route_code}
instruction=separate flagged cartons before release
''')
    footer = '''Final instruction:
Synthesize the archive into the required JSON object. Use only information supported by the archive
and keep the output concise, machine-readable, and complete.
'''
    return header, sections, footer


def build_grounded_parts() -> tuple[str, list[str], str]:
    header = '''Source dossier for a grounded long-form continuation follows.
Use the material below as the only authority for setting, objects, weather, and recent events.

Task:
Write the next section of the expedition log.

Constraints:
- 6 paragraphs
- third-person past tense
- no dialogue in the first paragraph
- stay consistent with the dossier details
- do not summarize the dossier
- target about 450-650 tokens
- preserve the observatory / mountain weather setting
- end with a small concrete sign of forward motion

Source dossier:
'''
    places = ["meridian stair", "east dome", "lens room", "wind gallery", "copper catwalk", "snow gate"]
    objects = ["brass eyepiece", "canvas tool roll", "frosted logbook", "signal lamp", "coil of wire", "oak transit case"]
    weather = ["granular sleet", "dry snow", "salt-cold fog", "needle rain", "glassy rime", "hard wind"]
    sections: list[str] = []
    for i in range(1, 1000):
        place = places[i % len(places)]
        obj = objects[i % len(objects)]
        wx = weather[i % len(weather)]
        sections.append(f'''Entry {i:03d}
On the climb back from the {place}, Mara noted that the {wx} had crusted the railings in a pale seam.
She set the {obj} beside the maintenance ledger, checked the dome gearing by hand,
and marked a faint vibration in the housing above bracket {(i % 9) + 1}.
The mountain lights below vanished and returned in brief intervals,
while the shutters answered the gusts with a dull iron shudder.
''')
        if i % 3 == 0:
            sections.append(f'''Supplement {i:03d}
The older charts still named the ridge stations in ink faded to tea-brown,
and the observatory smelled of oil, cold stone, and wool that never quite dried.
She kept finding traces of graphite on the latch plates,
as if someone had checked them in a hurry and meant to come back before first light.
''')
        if i % 5 == 0:
            sections.append(f'''Weather log {i:03d}
gusts crossed from the north-northeast,
the dome skin rang softly near the seam,
and drifted ice gathered at the threshold below the {places[(i + 2) % len(places)]}.
''')
    footer = '''Final instruction:
Write the next expedition-log section now. Keep it grounded in the dossier,
descriptive, and forward-moving.
'''
    return header, sections, footer


BUILDERS = {
    "code": build_code_parts,
    "synthesis": build_synthesis_parts,
    "grounded": build_grounded_parts,
}

manifest: dict[str, dict[str, dict[str, int | str]]] = {}
for family, builder in BUILDERS.items():
    header, sections, footer = builder()
    manifest[family] = {}
    for length_label, target in TARGET_TOKENS.items():
        trimmed, token_count, n_sections = trim_sections_to_token_target(header, sections, footer, target)
        path = OUT / f"{family}_{length_label}.txt"
        path.write_text(trimmed)
        manifest[family][length_label] = {
            "file": str(path.relative_to(ROOT.parent.parent)),
            "target_tokens": target,
            "actual_tokens": token_count,
            "sections_kept": n_sections,
            "actual_words": len(trimmed.split()),
            "chars": len(trimmed),
        }

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(OUT / "README.md").write_text(
    "# Long-context prompt corpus\n\n"
    "Current active prompt corpus for long-input DSpark vs baseline benchmarking.\n\n"
    "Families:\n"
    "- `code_{4k,8k,16k}.txt` — repository-style code completion with ~500-token target output\n"
    "- `synthesis_{4k,8k,16k}.txt` — long operations archive summarized as minified JSON output\n"
    "- `grounded_{4k,8k,16k}.txt` — grounded long-form continuation with ~500-token target output\n\n"
    "Lengths are trimmed against actual token counts measured with the local `ds4` tokenizer.\n"
    "Files preserve natural section and line boundaries for review.\n"
    "See `manifest.json` for retained sizes.\n"
)
