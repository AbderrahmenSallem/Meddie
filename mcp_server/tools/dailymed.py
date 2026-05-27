"""DailyMed v2 client — full FDA structured product labels (SPL).

Free, keyless. https://dailymed.nlm.nih.gov/dailymed/app-support-mapi.cfm
"""
from __future__ import annotations

import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


async def search_dailymed(drug_name: str, page_size: int = 10) -> dict:
    """Find SPL set-IDs matching a drug name."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            f"{BASE_URL}/spls.json",
            params={"drug_name": drug_name, "pagesize": page_size},
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        return {
            "drug": drug_name,
            "count": len(data),
            "results": [
                {
                    "setid": s.get("setid"),
                    "title": s.get("title"),
                    "published_date": s.get("published_date"),
                }
                for s in data
            ],
        }


async def get_dailymed_label(
    drug_name: Optional[str] = None,
    setid: Optional[str] = None,
) -> dict:
    """Get a parsed DailyMed label.

    Pass either a `drug_name` (we pick the first match) or a specific `setid`.
    Returns structured sections: indications, dosing, contraindications,
    warnings, interactions, adverse reactions, special populations.
    """
    if not setid and not drug_name:
        raise ValueError("Must provide drug_name or setid")

    if not setid:
        search = await search_dailymed(drug_name or "", page_size=1)
        results = search.get("results", [])
        if not results:
            return {"drug": drug_name, "found": False, "label": None}
        setid = results[0]["setid"]

    # The JSON SPL endpoint returns metadata; the XML SPL contains the
    # actual prescribing text. We fetch XML and parse the structured sections.
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{BASE_URL}/spls/{setid}.xml")
        if r.status_code == 404:
            return {"drug": drug_name, "setid": setid, "found": False, "label": None}
        r.raise_for_status()
        sections = _parse_spl_xml(r.text)
        return {
            "drug": drug_name,
            "setid": setid,
            "found": True,
            "label": sections,
            "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
        }


async def verify_drug_name(name: str) -> dict:
    """Check whether a drug name exists in DailyMed's drugnames index."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            f"{BASE_URL}/drugnames.json",
            params={"drug_name": name},
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        return {"name": name, "exists": bool(data), "matches": data[:10]}


# ---------- helpers ----------

# Common LOINC codes for the major SPL sections (used inside <section><code/>).
_SECTION_CODES = {
    "34066-1": "boxed_warning",
    "34067-9": "indications_and_usage",
    "34068-7": "dosage_and_administration",
    "34070-3": "contraindications",
    "34071-1": "warnings",
    "43685-7": "warnings_and_precautions",
    "34073-7": "drug_interactions",
    "34084-4": "adverse_reactions",
    "42228-7": "pregnancy",
    "34081-0": "pediatric_use",
    "34082-8": "geriatric_use",
    "43678-2": "use_in_specific_populations",
    "43680-8": "clinical_pharmacology",
}


def _parse_spl_xml(xml_text: str) -> dict:
    """Extract the main prescribing-information sections from SPL XML.

    SPL is an HL7 v3 doc; sections are tagged with a LOINC code and contain
    HTML-ish prose. We strip tags and collapse whitespace per section.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")
    out: dict = {key: "" for key in _SECTION_CODES.values()}

    for section in soup.find_all("section"):
        code_el = section.find("code")
        if not code_el:
            continue
        code = code_el.get("code", "")
        key = _SECTION_CODES.get(code)
        if not key:
            continue
        text = section.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        out[key] = text

    return out
