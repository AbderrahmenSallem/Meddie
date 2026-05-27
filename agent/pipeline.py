"""
Research pipeline — structured first-pass data gathering before the agentic loop.

Order (mirrors the architecture doc):
  [1] resolve_drug_name   — parallel across all substances in the query
  [2] check_interactions  — sequential, needs RxCUIs from step 1
  [3] fan-out             — parallel: pubmed, fda label, adverse events,
                            clinical trials, dailymed, reddit
  [4] follow-up scrapes   — best-effort scrapes of URLs found in Reddit results
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from mcp import ClientSession


class ResearchPipeline:
    def __init__(self, session: ClientSession) -> None:
        self.session = session

    async def run(self, query: str) -> dict:
        substances = _parse_substances(query)
        print(f"[pipeline] substances detected: {substances}")

        # [1] Resolve all drug names in parallel
        resolved_raw = await asyncio.gather(
            *[self._call("resolve_drug_name", {"name": s}) for s in substances],
            return_exceptions=True,
        )

        rxcuis: list[str] = []
        canonical_names: list[str] = []
        for r in resolved_raw:
            if isinstance(r, Exception):
                continue
            if isinstance(r, dict):
                if r.get("rxcui"):
                    rxcuis.append(str(r["rxcui"]))
                if r.get("rxcuis"):
                    rxcuis.extend(str(x) for x in r["rxcuis"])
                if r.get("canonical_name"):
                    canonical_names.append(r["canonical_name"])

        search_term = canonical_names[0] if canonical_names else substances[0]
        print(f"[pipeline] canonical search term: {search_term}")

        # [2] Interactions — only useful for stacks (2+ substances)
        interactions: Any = None
        if len(rxcuis) >= 2:
            interactions = await self._call("check_interactions", {"rxcuis": rxcuis})

        # [3] Fan out — all parallel
        (
            pubmed_results,
            fda_label,
            adverse_events,
            trials,
            dailymed_label,
            reddit_results,
        ) = await asyncio.gather(
            self._call("search_pubmed", {"query": search_term, "max_results": 20}),
            self._call("get_fda_label", {"drug_name": search_term}),
            self._call("get_adverse_events", {"drug_name": search_term}),
            self._call("search_clinical_trials", {"drug_name": search_term}),
            self._call("get_dailymed_label", {"drug_name": search_term}),
            self._call(
                "search_reddit",
                {
                    "query": search_term,
                    "category": _detect_category(query),
                    "post_limit": 20,
                    "comment_limit": 15,
                },
            ),
            return_exceptions=True,
        )

        # [4] Follow-up scrapes from URLs found in Reddit results (non-blocking)
        follow_ups = await self._scrape_reddit_links(reddit_results)

        return {
            "query": query,
            "substances": substances,
            "canonical_name": search_term,
            "resolved_names": [r for r in resolved_raw if not isinstance(r, Exception)],
            "interactions": interactions,
            "pubmed": _safe(pubmed_results),
            "fda_label": _safe(fda_label),
            "adverse_events": _safe(adverse_events),
            "clinical_trials": _safe(trials),
            "dailymed": _safe(dailymed_label),
            "reddit": _safe(reddit_results),
            "follow_up_scrapes": follow_ups,
        }

    async def _scrape_reddit_links(self, reddit_data: Any) -> list[Any]:
        if not reddit_data or isinstance(reddit_data, Exception):
            return []

        text = reddit_data if isinstance(reddit_data, str) else json.dumps(reddit_data, default=str)
        # Only scrape known high-value domains
        urls = re.findall(
            r"https?://(?:www\.)?(?:examine\.com|longecity\.org|exrx\.net)[^\s\"'<>]+",
            text,
        )
        if not urls:
            return []

        results = await asyncio.gather(
            *[self._call("scrape_webpage", {"url": u}) for u in urls[:3]],
            return_exceptions=True,
        )
        return [_safe(r) for r in results if not isinstance(r, Exception)]

    async def _call(self, tool: str, args: dict) -> Any:
        try:
            result = await self.session.call_tool(tool, args)
            if result.content:
                texts = [b.text for b in result.content if hasattr(b, "text")]
                raw = "\n".join(texts)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            return {}
        except Exception as exc:
            return {"error": str(exc), "tool": tool}


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe(value: Any) -> Any:
    return {} if isinstance(value, Exception) else value


def _parse_substances(query: str) -> list[str]:
    """Split 'testosterone + anastrozole' → ['testosterone', 'anastrozole']."""
    parts = re.split(r"\s*[\+&,]\s*|\s+and\s+", query, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _detect_category(query: str) -> str:
    q = query.lower()
    if any(w in q for w in [
        "hrt", "estradiol", "estrogen", "spironolactone", "progesterone",
        "bicalutamide", "feminiz", "masculin", "gnrh", "lupron",
    ]):
        return "hrt"
    if any(w in q for w in ["testosterone", "trt", "androgen", "enanthate", "cypionate"]):
        return "trt"
    if any(w in q for w in [
        "bpc", "tb-500", "tb500", "igf", "cjc", "peptide",
        "ghrp", "ipamorelin", "sermorelin", "hexarelin",
    ]):
        return "peptides"
    if any(w in q for w in [
        "sarm", "lgd", "rad", "ostarine", "mk-", "yk11",
        "aas", "steroid", "ped", "winstrol", "anavar",
    ]):
        return "ped"
    return "general"
