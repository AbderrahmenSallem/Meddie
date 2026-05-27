"""Generic web scraper — httpx + BeautifulSoup4.

- Polite headers (real-browser UA)
- robots.txt respect
- Caches every fetched page to ``cache/pages/``
- Strips boilerplate (nav, scripts, ads) before returning text
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "pages"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "MedResearchAgent/1.0"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags to strip before extracting visible text.
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside",
               "form", "noscript", "iframe", "svg")


async def scrape_webpage(
    url: str,
    use_cache: bool = True,
    timeout: float = 30.0,
) -> dict:
    """Fetch a URL and return cleaned text.

    Respects robots.txt. Caches the raw HTML to ``cache/pages/`` so
    re-scrapes are free.
    """
    cache_path = _cache_path_for(url, ".html")

    if use_cache and cache_path.exists():
        html = cache_path.read_text(encoding="utf-8", errors="replace")
        return {
            "url": url,
            "cached": True,
            "cache_path": str(cache_path),
            "text": _extract_text(html),
            "title": _extract_title(html),
        }

    # robots.txt check
    allowed = await _robots_allows(url)
    if not allowed:
        return {
            "url": url,
            "cached": False,
            "blocked_by_robots": True,
            "text": "",
            "title": "",
        }

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text

    cache_path.write_text(html, encoding="utf-8")

    return {
        "url": url,
        "cached": False,
        "cache_path": str(cache_path),
        "text": _extract_text(html),
        "title": _extract_title(html),
        "status_code": r.status_code,
    }


async def fetch_raw(url: str, timeout: float = 30.0) -> dict:
    """Fetch a URL and return raw bytes (no parsing, no cache).

    For PDFs / non-HTML content. Returns {url, content_type, bytes}.
    """
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        return {
            "url": url,
            "content_type": r.headers.get("content-type", ""),
            "bytes": r.content,
            "status_code": r.status_code,
        }


# ---------- helpers ----------

def _cache_path_for(url: str, suffix: str) -> Path:
    """Generate a stable, descriptive cache path for a URL."""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "").split(".")[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.path)[:50].strip("_") or "root"
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    today = date.today().isoformat()
    return CACHE_DIR / f"{host}_{slug}_{today}_{digest}{suffix}"


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def _extract_text(html: str) -> str:
    """Strip boilerplate and return clean visible text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    # Prefer <main> or <article> if present — they're usually the real content.
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


_robots_cache: dict[str, RobotFileParser] = {}
_robots_lock = asyncio.Lock()


async def _robots_allows(url: str) -> bool:
    """Check robots.txt for the given URL. Caches per-host parser."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    async with _robots_lock:
        rp = _robots_cache.get(base)
        if rp is None:
            rp = RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                # robotparser uses urllib (sync); run in thread.
                await asyncio.to_thread(rp.read)
            except Exception:
                # Treat fetch failures as "no rules" rather than blocking.
                rp.disallow_all = False
            _robots_cache[base] = rp
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True
