"""Reddit client via PRAW — community experience signal.

Credentials read from .env:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT

PRAW is synchronous; we run calls in a thread so they don't block
the rest of the async pipeline.
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Optional

try:
    import praw  # type: ignore
except ImportError:  # pragma: no cover
    praw = None  # surfaced at runtime in _get_client


# Subreddits to search by substance category. The agent picks the
# category before calling; "general" is the safe fallback.
SUBREDDIT_MAP: dict[str, list[str]] = {
    "hrt": ["TransDIY", "asktransgender", "TransMasc", "feminineboys", "MtF"],
    "trt": ["Testosterone", "trt", "maleHRT"],
    "peptides": ["Peptides", "PeptideSciences"],
    "ped": ["steroids", "PEDs", "sarmssourcetalk"],
    "general": ["Nootropics", "semaglutide", "DrugInformation", "AskDocs"],
}


@lru_cache(maxsize=1)
def _get_client():
    """Lazy PRAW client. Cached so we don't reauth on every call."""
    if praw is None:
        raise RuntimeError(
            "praw not installed. Run: pip install praw"
        )
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "MedResearchAgent/1.0")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in env."
        )
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


async def search_reddit(
    query: str,
    category: str = "general",
    post_limit: int = 20,
    comment_limit: int = 15,
    sort: str = "relevance",
    time_filter: str = "year",
) -> dict:
    """Search targeted subreddits for a substance and pull top posts + comments.

    Args:
        query: Search term (substance name).
        category: One of SUBREDDIT_MAP keys: hrt, trt, peptides, ped, general.
        post_limit: Max posts per subreddit.
        comment_limit: Max top comments per post.
        sort: 'relevance' | 'top' | 'new' | 'hot'.
        time_filter: 'all' | 'year' | 'month' | 'week' | 'day' | 'hour'.
    """
    subs = SUBREDDIT_MAP.get(category.lower(), SUBREDDIT_MAP["general"])
    return await asyncio.to_thread(
        _sync_search,
        query=query,
        subreddits=subs,
        post_limit=post_limit,
        comment_limit=comment_limit,
        sort=sort,
        time_filter=time_filter,
    )


async def get_thread(submission_id: str, comment_limit: int = 30) -> dict:
    """Fetch a specific Reddit thread with its top comments."""
    return await asyncio.to_thread(_sync_get_thread, submission_id, comment_limit)


# ---------- sync helpers (run inside asyncio.to_thread) ----------

def _sync_search(
    query: str,
    subreddits: list[str],
    post_limit: int,
    comment_limit: int,
    sort: str,
    time_filter: str,
) -> dict:
    reddit = _get_client()
    multi = "+".join(subreddits)
    sub = reddit.subreddit(multi)

    posts: list[dict] = []
    for submission in sub.search(
        query,
        sort=sort,
        time_filter=time_filter,
        limit=post_limit,
    ):
        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            comments = [
                {
                    "author": str(c.author) if c.author else "[deleted]",
                    "score": c.score,
                    "body": c.body,
                    "created_utc": c.created_utc,
                }
                for c in submission.comments[:comment_limit]
            ]
        except Exception:
            comments = []

        posts.append({
            "id": submission.id,
            "subreddit": str(submission.subreddit),
            "title": submission.title,
            "selftext": submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "url": f"https://reddit.com{submission.permalink}",
            "created_utc": submission.created_utc,
            "comments": comments,
        })

    return {
        "query": query,
        "subreddits": subreddits,
        "count": len(posts),
        "posts": posts,
    }


def _sync_get_thread(submission_id: str, comment_limit: int) -> dict:
    reddit = _get_client()
    submission = reddit.submission(id=submission_id)
    submission.comment_sort = "top"
    submission.comments.replace_more(limit=0)
    return {
        "id": submission.id,
        "subreddit": str(submission.subreddit),
        "title": submission.title,
        "selftext": submission.selftext,
        "score": submission.score,
        "url": f"https://reddit.com{submission.permalink}",
        "comments": [
            {
                "author": str(c.author) if c.author else "[deleted]",
                "score": c.score,
                "body": c.body,
            }
            for c in submission.comments[:comment_limit]
        ],
    }
