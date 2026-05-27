"""OpenFDA client — drug labels, FAERS adverse events, recalls.

Free, keyless. https://open.fda.gov/
"""
from __future__ import annotations

import httpx

BASE_URL = "https://api.fda.gov"


async def get_fda_label(drug_name: str) -> dict:
    """Fetch the FDA drug label for a substance.

    Returns black box warnings, contraindications, dosing,
    interactions, adverse reactions, and openfda metadata.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {
            "search": (
                f'openfda.generic_name:"{drug_name}" '
                f'OR openfda.brand_name:"{drug_name}" '
                f'OR openfda.substance_name:"{drug_name}"'
            ),
            "limit": 1,
        }
        r = await client.get(f"{BASE_URL}/drug/label.json", params=params)
        if r.status_code == 404:
            return {"drug": drug_name, "found": False, "label": None}
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return {"drug": drug_name, "found": False, "label": None}

        label = results[0]
        openfda = label.get("openfda", {})
        return {
            "drug": drug_name,
            "found": True,
            "label": {
                "boxed_warning": _join(label.get("boxed_warning")),
                "warnings": _join(label.get("warnings")),
                "contraindications": _join(label.get("contraindications")),
                "indications_and_usage": _join(label.get("indications_and_usage")),
                "dosage_and_administration": _join(label.get("dosage_and_administration")),
                "drug_interactions": _join(label.get("drug_interactions")),
                "adverse_reactions": _join(label.get("adverse_reactions")),
                "pregnancy": _join(label.get("pregnancy")),
                "pediatric_use": _join(label.get("pediatric_use")),
                "geriatric_use": _join(label.get("geriatric_use")),
                "active_ingredients": openfda.get("substance_name", []),
                "brand_names": openfda.get("brand_name", []),
                "generic_names": openfda.get("generic_name", []),
                "route": openfda.get("route", []),
                "manufacturer": openfda.get("manufacturer_name", []),
            },
        }


async def get_adverse_events(drug_name: str, limit: int = 100) -> dict:
    """Get aggregated FAERS adverse event counts for a drug."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit,
        }
        r = await client.get(f"{BASE_URL}/drug/event.json", params=params)
        if r.status_code == 404:
            return {"drug": drug_name, "found": False, "events": []}
        r.raise_for_status()
        events = [
            {"reaction": e["term"], "count": e["count"]}
            for e in r.json().get("results", [])
        ]
        return {"drug": drug_name, "found": True, "events": events}


async def get_recalls(drug_name: str, limit: int = 10) -> dict:
    """Get FDA enforcement / recall actions for a drug."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {
            "search": f'product_description:"{drug_name}"',
            "limit": limit,
        }
        r = await client.get(f"{BASE_URL}/drug/enforcement.json", params=params)
        if r.status_code == 404:
            return {"drug": drug_name, "recalls": []}
        r.raise_for_status()
        recalls = [
            {
                "reason": rec.get("reason_for_recall"),
                "classification": rec.get("classification"),
                "status": rec.get("status"),
                "recall_date": rec.get("recall_initiation_date"),
                "product": rec.get("product_description"),
            }
            for rec in r.json().get("results", [])
        ]
        return {"drug": drug_name, "recalls": recalls}


# ---------- helpers ----------

def _join(field) -> str:
    """OpenFDA returns most text fields as a list of strings; collapse to one."""
    if field is None:
        return ""
    if isinstance(field, list):
        return " ".join(str(x) for x in field)
    return str(field)
