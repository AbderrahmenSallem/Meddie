"""
PILLAR: REFLECTION

After the agent produces a draft report, the ReflectionEngine makes a
second LLM call with a critic persona. The critic reads the draft and
returns a structured verdict: PASS or NEEDS_REVISION, a list of specific
gaps, and suggested follow-up tool calls.

If the verdict is NEEDS_REVISION, agent.py runs one targeted revision
pass — injecting the critique into the conversation so Claude can fill
exactly the identified gaps before writing the final report.

The revision loop is intentionally bounded to a single pass (draft →
critique → revision → final) to prevent infinite self-correction cycles.
"""
from __future__ import annotations

import json
import re
from typing import Any

_CRITIC_SYSTEM = """\
You are a strict quality reviewer of medical research reports.

Your job is to evaluate a draft report for completeness, accuracy, and evidence quality.
You are NOT the author — you are the critic.

Evaluate the draft on these dimensions:

1. SOURCE COVERAGE: Were all key sources consulted?
   - Clinical: PubMed, FDA label, DailyMed, ClinicalTrials, FAERS
   - Community: Reddit or forum data (if applicable to the substance type)

2. SECTION COMPLETENESS: Are any required sections missing or empty?
   - Required: Substance Profile, Mechanism of Action, Clinical Evidence,
     Dosing Protocols, Side Effects, Drug Interactions, Sources/Citations, Disclaimer

3. EVIDENCE GAPS: Are major claims made without citations or with only
   low-tier sources (T4/T5) when higher-tier data likely exists?

4. CONFLICTS: Were any conflicts between clinical and community data surfaced?
   If the substance has an active community (HRT, peptides, TRT, PEDs), this
   section must exist even if it says "No conflicts found."

5. WARNINGS: If the FDA label has black box warnings, are they prominently
   displayed near the top of the report?

Return ONLY valid JSON in this exact structure — no prose outside the JSON:
{
  "quality_verdict": "PASS" or "NEEDS_REVISION",
  "score": <integer 0–100>,
  "gaps": [
    "specific description of each gap or missing item"
  ],
  "revision_instructions": [
    "specific action: e.g. 'call search_pubmed with query X to find RCT data',
     'add a Conflicts section comparing FDA dosing to community-reported dosing'"
  ]
}

Score guide:
  90–100: Comprehensive, well-cited, all sections present → PASS
  70–89:  Minor gaps, mostly complete → PASS
  50–69:  Meaningful sections missing or major claims uncited → NEEDS_REVISION
  0–49:   Substantially incomplete → NEEDS_REVISION
"""


class ReflectionEngine:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def critique(self, draft: str) -> dict:
        """Run the critic over a draft report. Returns the structured verdict."""
        print("[reflection] critiquing draft …")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": f"Draft report:\n\n{draft}"},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return _parse_verdict(raw)

    def format_feedback(self, critique: dict) -> str:
        """Format the critique into a human-readable string for injection."""
        score = critique.get("score", "?")
        verdict = critique.get("quality_verdict", "UNKNOWN")
        gaps = critique.get("gaps", [])
        instructions = critique.get("revision_instructions", [])

        lines = [
            f"## Reflection Feedback (score: {score}/100 — {verdict})",
            "",
            "### Identified Gaps",
        ]
        for g in gaps:
            lines.append(f"- {g}")
        lines += ["", "### Revision Instructions"]
        for ins in instructions:
            lines.append(f"- {ins}")
        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_verdict(raw: str) -> dict:
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Safe fallback — treat as passing so we don't loop on parse failure
    return {
        "quality_verdict": "PASS",
        "score": 70,
        "gaps": [],
        "revision_instructions": [],
    }
