"""MCP server — exposes every research tool to Claude.

Run with:
    python -m mcp_server.server          # stdio transport (default)
    python -m mcp_server.server --http   # HTTP transport on 127.0.0.1:8765

Each registered tool keeps a clean, documented signature so Claude
can call them with the right arguments without guessing.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    clinicaltrials,
    dailymed,
    openfda,
    pdf_capture,
    pubmed,
    reddit as reddit_tool,
    rxnorm,
    web_scraper,
)

load_dotenv()

mcp = FastMCP("med-research-agent")


# ---------- PubMed ----------

@mcp.tool()
async def search_pubmed(
    query: str,
    max_results: int = 20,
    filters: Optional[list[str]] = None,
) -> dict:
    """Search PubMed (35M+ biomedical papers) for a substance.

    Returns title, abstract, authors, journal, year, study_type, PMID, DOI
    for each paper. Filters can include "clinical trial", "review",
    "meta-analysis", "systematic review".
    """
    return await pubmed.search_pubmed(query, max_results=max_results, filters=filters)


@mcp.tool()
async def get_related_papers(pmid: str, max_results: int = 10) -> dict:
    """Find papers related to a given PMID via PubMed's elink service."""
    return await pubmed.get_related_articles(pmid, max_results=max_results)


# ---------- OpenFDA ----------

@mcp.tool()
async def get_fda_label(drug_name: str) -> dict:
    """Fetch the FDA drug label.

    Includes black box warnings, contraindications, dosing, interactions,
    adverse reactions, pregnancy and special-population notes.
    """
    return await openfda.get_fda_label(drug_name)


@mcp.tool()
async def get_adverse_events(drug_name: str, limit: int = 100) -> dict:
    """Get FAERS adverse-event counts for a drug (real patient reports)."""
    return await openfda.get_adverse_events(drug_name, limit=limit)


@mcp.tool()
async def get_recalls(drug_name: str, limit: int = 10) -> dict:
    """Get FDA recall / enforcement actions for a drug."""
    return await openfda.get_recalls(drug_name, limit=limit)


# ---------- ClinicalTrials.gov ----------

@mcp.tool()
async def search_clinical_trials(
    drug_name: str,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    page_size: int = 20,
) -> dict:
    """Search ClinicalTrials.gov for studies of a substance.

    `status` can be COMPLETED / RECRUITING / ACTIVE_NOT_RECRUITING / TERMINATED.
    `phase` can be PHASE1 / PHASE2 / PHASE3 / PHASE4.
    """
    return await clinicaltrials.search_clinical_trials(
        drug_name, status=status, phase=phase, page_size=page_size
    )


@mcp.tool()
async def get_trial_details(nct_id: str) -> dict:
    """Get the full record for a single trial by NCT ID."""
    return await clinicaltrials.get_trial_details(nct_id)


# ---------- RxNorm ----------

@mcp.tool()
async def resolve_drug_name(name: str) -> dict:
    """Normalize a user-typed drug name to its canonical RxCUI.

    Always call this first in the pipeline — other tools work better with
    the resolved canonical name.
    """
    return await rxnorm.resolve_drug_name(name)


@mcp.tool()
async def get_related_names(rxcui: str) -> dict:
    """Get every related brand/generic name for an RxCUI."""
    return await rxnorm.get_related_names(rxcui)


@mcp.tool()
async def check_interactions(rxcuis: list[str]) -> dict:
    """Check pairwise drug interactions for a list of RxCUIs.

    Note: NLM retired the public interaction API in Jan 2024. If unavailable,
    the response includes a `note` field so the agent can flag the gap.
    """
    return await rxnorm.check_interactions(rxcuis)


# ---------- DailyMed ----------

@mcp.tool()
async def get_dailymed_label(
    drug_name: Optional[str] = None,
    setid: Optional[str] = None,
) -> dict:
    """Fetch the structured DailyMed SPL label for a drug.

    Provide either a drug name (we pick the top match) or a specific setid.
    More detail than OpenFDA — full prescribing information.
    """
    return await dailymed.get_dailymed_label(drug_name=drug_name, setid=setid)


@mcp.tool()
async def search_dailymed(drug_name: str, page_size: int = 10) -> dict:
    """List DailyMed SPLs matching a drug name."""
    return await dailymed.search_dailymed(drug_name, page_size=page_size)


# ---------- Reddit ----------

@mcp.tool()
async def search_reddit(
    query: str,
    category: str = "general",
    post_limit: int = 20,
    comment_limit: int = 15,
    sort: str = "relevance",
    time_filter: str = "year",
) -> dict:
    """Search targeted subreddits for real-world community experience.

    `category` selects which subreddits to search:
        hrt      - r/TransDIY, r/asktransgender, r/MtF, etc.
        trt      - r/Testosterone, r/trt, r/maleHRT
        peptides - r/Peptides, r/PeptideSciences
        ped      - r/steroids, r/PEDs, r/sarmssourcetalk
        general  - r/Nootropics, r/AskDocs, r/DrugInformation
    """
    return await reddit_tool.search_reddit(
        query, category=category, post_limit=post_limit,
        comment_limit=comment_limit, sort=sort, time_filter=time_filter,
    )


@mcp.tool()
async def get_reddit_thread(submission_id: str, comment_limit: int = 30) -> dict:
    """Fetch a Reddit submission with its top comments."""
    return await reddit_tool.get_thread(submission_id, comment_limit=comment_limit)


# ---------- Web scraping & PDF capture ----------

@mcp.tool()
async def scrape_webpage(url: str, use_cache: bool = True) -> dict:
    """Fetch any URL and return cleaned visible text.

    Respects robots.txt. Pages are cached on disk so re-fetches are free.
    Use this for forum threads, blog posts, harm-reduction guides, etc.
    """
    return await web_scraper.scrape_webpage(url, use_cache=use_cache)


@mcp.tool()
async def capture_page_as_pdf(url: str) -> dict:
    """Download (or render) a URL to a PDF saved in cache/pages/.

    Auto-detects direct PDFs vs HTML pages; uses headless Chromium for HTML.
    """
    return await pdf_capture.capture_page_as_pdf(url)


@mcp.tool()
async def capture_page_as_text(url: str) -> dict:
    """Render a JS-heavy page with headless Chromium and return innerText.

    Use this when scrape_webpage returns suspiciously thin content.
    """
    return await pdf_capture.capture_page_as_text(url)


@mcp.tool()
def read_cached_pdf(path: str) -> dict:
    """Extract text from a previously-saved PDF in cache/pages/."""
    return pdf_capture.read_cached_pdf(path)


# ---------- entry point ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Med Research Agent MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run over HTTP (default: stdio).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        # FastMCP exposes an SSE/HTTP runner.
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
