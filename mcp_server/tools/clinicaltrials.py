"""ClinicalTrials.gov v2 REST API client.

Free, keyless. https://clinicaltrials.gov/data-api/api
"""
from __future__ import annotations

from typing import Optional

import httpx

BASE_URL = "https://clinicaltrials.gov/api/v2"

_DEFAULT_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "Phase",
    "Condition",
    "InterventionName",
    "PrimaryOutcomeMeasure",
    "SecondaryOutcomeMeasure",
    "EnrollmentCount",
    "StartDate",
    "CompletionDate",
    "StudyType",
    "LeadSponsorName",
]


async def search_clinical_trials(
    drug_name: str,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    page_size: int = 20,
) -> dict:
    """Search ClinicalTrials.gov for studies involving a substance.

    Args:
        drug_name: Intervention name to search for.
        status: Optional status filter — "COMPLETED", "RECRUITING",
                "ACTIVE_NOT_RECRUITING", "TERMINATED", etc.
        phase: Optional phase filter — "PHASE1", "PHASE2", "PHASE3", "PHASE4".
        page_size: Max number of studies (default 20, max 1000).
    """
    params: dict = {
        "query.intr": drug_name,
        "pageSize": page_size,
        "format": "json",
        "fields": ",".join(_DEFAULT_FIELDS),
    }
    if status:
        params["filter.overallStatus"] = status.upper()
    if phase:
        params["filter.phase"] = phase.upper()

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{BASE_URL}/studies", params=params)
        r.raise_for_status()
        data = r.json()
        trials = [_flatten_study(s) for s in data.get("studies", [])]
        return {
            "drug": drug_name,
            "count": len(trials),
            "total_available": data.get("totalCount", len(trials)),
            "trials": trials,
        }


async def get_trial_details(nct_id: str) -> dict:
    """Get the full record for a single trial by NCT ID."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{BASE_URL}/studies/{nct_id}", params={"format": "json"})
        if r.status_code == 404:
            return {"nct_id": nct_id, "found": False}
        r.raise_for_status()
        return {"nct_id": nct_id, "found": True, "study": r.json()}


# ---------- helpers ----------

def _flatten_study(study: dict) -> dict:
    """Pull the most useful fields out of the deeply-nested CTGov response."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    arms = proto.get("armsInterventionsModule", {})
    outcomes = proto.get("outcomesModule", {})
    cond = proto.get("conditionsModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})

    nct_id = ident.get("nctId", "")
    interventions = [
        i.get("name", "")
        for i in arms.get("interventions", [])
        if i.get("name")
    ]

    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "status": status.get("overallStatus", ""),
        "phase": design.get("phases", []),
        "study_type": design.get("studyType", ""),
        "conditions": cond.get("conditions", []),
        "interventions": interventions,
        "primary_outcomes": [
            o.get("measure", "") for o in outcomes.get("primaryOutcomes", [])
        ],
        "secondary_outcomes": [
            o.get("measure", "") for o in outcomes.get("secondaryOutcomes", [])
        ],
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "start_date": status.get("startDateStruct", {}).get("date", ""),
        "completion_date": status.get("completionDateStruct", {}).get("date", ""),
        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
    }
