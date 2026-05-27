"""PDF capture + read tool.

Three scenarios from the architecture doc:
  A) URL points directly to a PDF        → download bytes, save, return path
  B) URL is HTML, want a PDF snapshot    → playwright renders → page.pdf()
  C) URL is HTML, just want text         → playwright renders → innerText

Plus:
  read_cached_pdf(path)                   → pdfplumber text extraction
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "pages"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


async def capture_page_as_pdf(url: str, timeout: float = 60.0) -> dict:
    """Render a URL to PDF and save to cache.

    - If the URL serves a PDF directly (Scenario A), download bytes.
    - Otherwise (Scenario B), use playwright to print the rendered page.
    """
    head_info = await _head_or_get(url)
    content_type = head_info["content_type"].lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        # Scenario A: direct PDF download
        pdf_bytes = head_info.get("body") or await _download_bytes(url)
        path = _cache_path_for(url, ".pdf")
        path.write_bytes(pdf_bytes)
        return {
            "url": url,
            "mode": "direct_pdf",
            "path": str(path),
            "size_bytes": len(pdf_bytes),
        }

    # Scenario B: HTML page → render to PDF via playwright
    path = _cache_path_for(url, ".pdf")
    await _playwright_render_to_pdf(url, path, timeout=timeout)
    return {
        "url": url,
        "mode": "rendered_pdf",
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


async def capture_page_as_text(url: str, timeout: float = 60.0) -> dict:
    """Scenario C: render a JS-heavy page and return innerText.

    Use this when scrape_webpage() returns thin content because the page is
    client-rendered.
    """
    text = await _playwright_render_to_text(url, timeout=timeout)
    return {"url": url, "mode": "rendered_text", "text": text}


def read_cached_pdf(path: str | Path) -> dict:
    """Extract text from a PDF file using pdfplumber.

    Returned text preserves page boundaries with a "[Page N]" marker.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber") from e

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "found": False, "text": ""}

    pages: list[str] = []
    with pdfplumber.open(p) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append(f"[Page {i}]\n{txt}")
    return {
        "path": str(p),
        "found": True,
        "page_count": len(pages),
        "text": "\n\n".join(pages),
    }


# ---------- internal: HTTP / playwright ----------

async def _head_or_get(url: str) -> dict:
    """Try HEAD first; some servers don't allow it, so fall back to a
    streamed GET that only reads what we need to detect the content type.
    """
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        try:
            r = await client.head(url)
            if r.status_code < 400 and "content-type" in r.headers:
                return {"content_type": r.headers["content-type"], "body": None}
        except httpx.HTTPError:
            pass
        # Fallback: GET; if it's a PDF we keep the bytes to avoid a second fetch.
        r = await client.get(url)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        body = r.content if "pdf" in ct.lower() else None
        return {"content_type": ct, "body": body}


async def _download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=60.0,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


async def _playwright_render_to_pdf(url: str, path: Path, timeout: float) -> None:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from e

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))
            await page.pdf(
                path=str(path),
                format="A4",
                print_background=True,
                margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"},
            )
        finally:
            await browser.close()


async def _playwright_render_to_text(url: str, timeout: float) -> str:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from e

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(user_agent=USER_AGENT)
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))
            text = await page.evaluate("() => document.body.innerText")
            return text or ""
        finally:
            await browser.close()


# ---------- naming ----------

def _cache_path_for(url: str, suffix: str) -> Path:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "").split(".")[0] or "page"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.path)[:50].strip("_") or "root"
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    today = date.today().isoformat()
    return CACHE_DIR / f"{host}_{slug}_{today}_{digest}{suffix}"
