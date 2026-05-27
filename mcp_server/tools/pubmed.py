"""PubMed E-utilities client — clinical literature search.

Free, keyless. Optionally pass NCBI_EMAIL via env to raise the rate limit
from 3 req/s to 10 req/s.
"""
from __future__ import annotations

import os
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = os.getenv("NCBI_EMAIL", "")


async def search_pubmed(
    query: str,
    max_results: int = 20,
    filters: Optional[list[str]] = None,
) -> dict:
    """Search PubMed for papers mentioning a substance.

    Args:
        query: The substance name or topic.
        max_results: Max number of papers (default 20).
        filters: Optional list of publication-type filters,
                 e.g. ["clinical trial", "review", "meta-analysis"].

    Returns:
        {"query": str, "count": int, "papers": [...]}
        Each paper has: pmid, title, abstract, authors, journal, year,
        study_type, doi, url.
    """
    search_term = f'"{query}"[Title/Abstract]'
    if filters:
        clauses: list[str] = []
        for f in filters:
            fl = f.lower().strip()
            if fl in ("clinical trial", "rct", "randomized controlled trial"):
                clauses.append('"randomized controlled trial"[Publication Type]')
            elif fl == "review":
                clauses.append('"review"[Publication Type]')
            elif fl in ("meta-analysis", "meta analysis"):
                clauses.append('"meta-analysis"[Publication Type]')
            elif fl == "systematic review":
                clauses.append('"systematic review"[Publication Type]')
        if clauses:
            search_term += " AND (" + " OR ".join(clauses) + ")"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: esearch — get PMIDs
        search_params: dict = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if EMAIL:
            search_params["email"] = EMAIL

        r = await client.get(f"{BASE_URL}/esearch.fcgi", params=search_params)
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return {"query": query, "count": 0, "papers": []}

        # Step 2: efetch — pull full records as XML
        fetch_params: dict = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if EMAIL:
            fetch_params["email"] = EMAIL

        r = await client.get(f"{BASE_URL}/efetch.fcgi", params=fetch_params)
        r.raise_for_status()

        papers = _parse_pubmed_xml(r.text)
        return {"query": query, "count": len(papers), "papers": papers}


async def get_related_articles(pmid: str, max_results: int = 10) -> dict:
    """Find related papers via PubMed's elink service."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        params: dict = {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
            "cmd": "neighbor",
        }
        if EMAIL:
            params["email"] = EMAIL
        r = await client.get(f"{BASE_URL}/elink.fcgi", params=params)
        r.raise_for_status()
        data = r.json()
        related: list[str] = []
        for linkset in data.get("linksets", []):
            for link in linkset.get("linksetdbs", []):
                if link.get("linkname") == "pubmed_pubmed":
                    related.extend(link.get("links", [])[:max_results])
                    break
        return {"pmid": pmid, "related_pmids": related}


# ---------- helpers ----------

def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML response into structured paper records."""
    root = ET.fromstring(xml_text)
    papers: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="")
        abstract_parts = [el.text or "" for el in article.findall(".//AbstractText")]
        abstract = " ".join(abstract_parts).strip()
        journal = article.findtext(".//Journal/Title", default="")
        year = article.findtext(".//PubDate/Year", default="") or \
               article.findtext(".//PubDate/MedlineDate", default="")

        authors: list[str] = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", default="")
            initials = author.findtext("Initials", default="")
            if last:
                authors.append(f"{last} {initials}".strip())

        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
        study_type = _classify_study_type(pub_types)

        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text or ""
                break

        papers.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "study_type": study_type,
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return papers


def _classify_study_type(pub_types: list[str]) -> str:
    """Map publication types to a credibility-relevant study type label."""
    t = [pt.lower() for pt in pub_types]
    if any("meta-analysis" in x for x in t):
        return "meta-analysis"
    if any("systematic review" in x for x in t):
        return "systematic review"
    if any("randomized controlled trial" in x for x in t):
        return "RCT"
    if any("clinical trial" in x for x in t):
        return "clinical trial"
    if any("review" in x for x in t):
        return "review"
    if any("case report" in x for x in t):
        return "case report"
    return "other"
