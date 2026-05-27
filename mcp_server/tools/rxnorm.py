"""RxNorm REST client — drug name normalization + interaction lookup.

Free, keyless. https://rxnav.nlm.nih.gov/RxNormAPIs.html

NOTE: The NLM Drug Interaction API endpoint was retired in 2024. We still
expose `check_interactions` so the agent can call it; if the endpoint
is offline, it returns an empty list with a `note` explaining the status.
"""
from __future__ import annotations

import httpx

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


async def resolve_drug_name(name: str) -> dict:
    """Resolve a user-typed drug name to its canonical RxCUI.

    Tries the approximate matcher if exact match fails, so "T-cyp" still
    resolves to "testosterone cypionate".
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        # Exact match
        r = await client.get(f"{BASE_URL}/rxcui.json", params={"name": name, "search": 2})
        r.raise_for_status()
        ids = r.json().get("idGroup", {}).get("rxnormId", []) or []

        approximate_used = False
        if not ids:
            # Approximate match (handles typos / abbreviations)
            r = await client.get(
                f"{BASE_URL}/approximateTerm.json",
                params={"term": name, "maxEntries": 5},
            )
            r.raise_for_status()
            candidates = r.json().get("approximateGroup", {}).get("candidate", []) or []
            ids = [c["rxcui"] for c in candidates if c.get("rxcui")]
            approximate_used = True

        if not ids:
            return {"input": name, "resolved": False, "rxcui": None, "canonical_name": None}

        rxcui = ids[0]

        # Get canonical name + concept properties
        r = await client.get(f"{BASE_URL}/rxcui/{rxcui}/properties.json")
        r.raise_for_status()
        props = r.json().get("properties", {})

        return {
            "input": name,
            "resolved": True,
            "rxcui": rxcui,
            "canonical_name": props.get("name", ""),
            "tty": props.get("tty", ""),
            "synonym": props.get("synonym", ""),
            "approximate_match": approximate_used,
        }


async def get_related_names(rxcui: str) -> dict:
    """Get all related brand/generic names for a drug.

    Useful for OpenFDA/DailyMed which sometimes index by brand, sometimes generic.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{BASE_URL}/rxcui/{rxcui}/allrelated.json")
        r.raise_for_status()
        groups = r.json().get("allRelatedGroup", {}).get("conceptGroup", [])
        out: dict = {"rxcui": rxcui, "by_type": {}}
        for g in groups:
            tty = g.get("tty", "")
            concepts = g.get("conceptProperties", []) or []
            out["by_type"][tty] = [
                {"rxcui": c.get("rxcui"), "name": c.get("name")} for c in concepts
            ]
        return out


async def check_interactions(rxcuis: list[str]) -> dict:
    """Check pairwise drug interactions for a list of RxCUIs.

    The NLM-hosted Drug Interaction API was retired in January 2024.
    We attempt the call for completeness; if it fails we surface a note
    so the agent knows the data is unavailable and can flag the gap.
    """
    if not rxcuis or len(rxcuis) < 1:
        return {"rxcuis": rxcuis, "interactions": [], "note": "no rxcuis provided"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            if len(rxcuis) == 1:
                r = await client.get(f"{BASE_URL}/interaction/interaction.json",
                                     params={"rxcui": rxcuis[0]})
            else:
                r = await client.get(
                    f"{BASE_URL}/interaction/list.json",
                    params={"rxcuis": "+".join(rxcuis)},
                )
            r.raise_for_status()
            return _parse_interactions(r.json(), rxcuis)
        except (httpx.HTTPError, ValueError) as e:
            return {
                "rxcuis": rxcuis,
                "interactions": [],
                "note": f"interaction API unavailable (retired by NLM Jan 2024): {e}",
            }


# ---------- helpers ----------

def _parse_interactions(data: dict, rxcuis: list[str]) -> dict:
    """Flatten RxNorm interaction response."""
    pairs: list[dict] = []

    full = data.get("fullInteractionTypeGroup", []) or []
    for group in full:
        for fit in group.get("fullInteractionType", []) or []:
            for pair in fit.get("interactionPair", []) or []:
                concepts = pair.get("interactionConcept", []) or []
                names = [
                    c.get("minConceptItem", {}).get("name", "") for c in concepts
                ]
                pairs.append({
                    "drugs": names,
                    "severity": pair.get("severity", ""),
                    "description": pair.get("description", ""),
                    "source": group.get("sourceName", ""),
                })

    single = data.get("interactionTypeGroup", []) or []
    for group in single:
        for it in group.get("interactionType", []) or []:
            for pair in it.get("interactionPair", []) or []:
                concepts = pair.get("interactionConcept", []) or []
                names = [c.get("minConceptItem", {}).get("name", "") for c in concepts]
                pairs.append({
                    "drugs": names,
                    "severity": pair.get("severity", ""),
                    "description": pair.get("description", ""),
                    "source": group.get("sourceName", ""),
                })

    return {"rxcuis": rxcuis, "interactions": pairs}
