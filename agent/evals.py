"""
PILLAR: EVALS

Deterministic quality scoring for a completed research run.
No LLM call needed — this scores observable facts about the pipeline
output and the final report text.

Four sub-scores are computed and combined into an overall score (0–100).
The result is appended to the saved report as a metadata block so every
report is self-documenting about its own data quality.

Sub-scores
──────────
1. source_coverage   — how many of the 7 key data sources returned usable data
2. completeness      — how many required report sections are present and non-empty
3. evidence_quality  — tier distribution (penalises reports backed only by T4/T5)
4. reflection_score  — the score returned by the ReflectionEngine (if available)

Weights: coverage 30 %, completeness 30 %, evidence 25 %, reflection 15 %
"""
from __future__ import annotations

import re
from typing import Any

# Sources checked against pipeline_data keys
_SOURCES = ["pubmed", "fda_label", "adverse_events", "clinical_trials", "dailymed", "reddit"]

# Sections that must appear (case-insensitive substring match)
_REQUIRED_SECTIONS = [
    "substance profile",
    "mechanism of action",
    "clinical evidence",
    "dosing",
    "side effects",
    "drug interaction",
    "sources",
    "disclaimer",
]

# Evidence tier markers in the report
_TIER_PATTERNS = {
    "T1": re.compile(r"\[T1\]", re.IGNORECASE),
    "T2": re.compile(r"\[T2\]", re.IGNORECASE),
    "T3": re.compile(r"\[T3\]", re.IGNORECASE),
    "T4": re.compile(r"\[T4\]", re.IGNORECASE),
    "T5": re.compile(r"\[T5\]", re.IGNORECASE),
}


class EvalEngine:
    def score(
        self,
        report_text: str,
        pipeline_data: dict,
        reflection_score: int | None = None,
    ) -> dict:
        coverage = _source_coverage(pipeline_data)
        completeness = _completeness(report_text)
        evidence = _evidence_quality(report_text)
        reflect = reflection_score if reflection_score is not None else 70

        overall = round(
            coverage * 0.30
            + completeness * 0.30
            + evidence * 0.25
            + reflect * 0.15
        )

        return {
            "source_coverage": coverage,
            "completeness": completeness,
            "evidence_quality": evidence,
            "reflection_score": reflect,
            "overall": overall,
            "grade": _grade(overall),
            "details": {
                "sources_hit": _sources_hit(pipeline_data),
                "sections_found": _sections_found(report_text),
                "tier_counts": _tier_counts(report_text),
            },
        }

    def format_block(self, scores: dict) -> str:
        """Return a Markdown block to append to the report."""
        d = scores["details"]
        tier = d["tier_counts"]
        sources = ", ".join(d["sources_hit"]) or "none"
        sections = ", ".join(d["sections_found"]) or "none"

        lines = [
            "",
            "---",
            "",
            "## Research Quality Scorecard",
            "",
            f"| Dimension | Score |",
            f"|---|---|",
            f"| Source Coverage | {scores['source_coverage']}/100 |",
            f"| Report Completeness | {scores['completeness']}/100 |",
            f"| Evidence Quality | {scores['evidence_quality']}/100 |",
            f"| Reflection Score | {scores['reflection_score']}/100 |",
            f"| **Overall** | **{scores['overall']}/100 ({scores['grade']})** |",
            "",
            f"**Sources with data:** {sources}",
            f"**Sections present:** {sections}",
            f"**Evidence tiers:** "
            + "  ".join(f"{k}×{v}" for k, v in tier.items() if v > 0),
        ]
        return "\n".join(lines)


# ── scoring functions ─────────────────────────────────────────────────────────

def _source_coverage(pipeline_data: dict) -> int:
    hit = len(_sources_hit(pipeline_data))
    return round((hit / len(_SOURCES)) * 100)


def _sources_hit(pipeline_data: dict) -> list[str]:
    return [
        src for src in _SOURCES
        if _is_nonempty(pipeline_data.get(src))
    ]


def _is_nonempty(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, dict):
        # Error-only dicts count as empty
        return bool(val) and set(val.keys()) != {"error", "tool"}
    if isinstance(val, (str, list)):
        return bool(val)
    return True


def _completeness(report_text: str) -> int:
    found = len(_sections_found(report_text))
    return round((found / len(_REQUIRED_SECTIONS)) * 100)


def _sections_found(report_text: str) -> list[str]:
    lower = report_text.lower()
    return [s for s in _REQUIRED_SECTIONS if s in lower]


def _evidence_quality(report_text: str) -> int:
    counts = _tier_counts(report_text)
    high = counts["T1"] + counts["T2"]
    mid = counts["T3"]
    low = counts["T4"] + counts["T5"]
    total = high + mid + low
    if total == 0:
        return 40  # no tier labels at all — penalise but don't zero
    # Weighted score: T1/T2 full, T3 half, T4/T5 quarter
    weighted = (high * 1.0 + mid * 0.5 + low * 0.25) / total
    return round(weighted * 100)


def _tier_counts(report_text: str) -> dict[str, int]:
    return {tier: len(pat.findall(report_text)) for tier, pat in _TIER_PATTERNS.items()}


def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
