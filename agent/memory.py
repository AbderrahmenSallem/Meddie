"""
PILLAR: MEMORY

Long-term substance research memory.

Stores canonical name mappings and key findings per substance across runs.
When the same substance is queried again within the TTL window, the agent
loads prior findings as context so it can skip re-fetching known data and
focus its tool calls on anything that's changed or was missed.

Storage: cache/research_memory.json (simple JSON, no DB dependency)
TTL:     7 days (clinical data changes slowly; 7 days is a safe window)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_MEMORY_FILE = Path(__file__).parent.parent / "cache" / "research_memory.json"
_TTL = 7 * 24 * 3600  # 7 days in seconds


class ResearchMemory:
    def __init__(self) -> None:
        self._store: dict = _load(_MEMORY_FILE)

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, substance: str) -> dict | None:
        """Return cached findings for a substance if the entry is still fresh."""
        entry = self._store.get(_key(substance))
        if entry and (time.time() - entry.get("ts", 0)) < _TTL:
            return entry["data"]
        return None

    def save(self, substance: str, data: dict) -> None:
        """Persist findings for a substance."""
        self._store[_key(substance)] = {"ts": time.time(), "data": data}
        _flush(self._store, _MEMORY_FILE)

    def list_known(self) -> list[str]:
        """Return all substance keys that have unexpired entries."""
        now = time.time()
        return [k for k, v in self._store.items() if (now - v.get("ts", 0)) < _TTL]

    def invalidate(self, substance: str) -> None:
        """Force-expire a substance's cache entry."""
        self._store.pop(_key(substance), None)
        _flush(self._store, _MEMORY_FILE)


# ── helpers ───────────────────────────────────────────────────────────────────

def _key(substance: str) -> str:
    return substance.lower().strip()


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _flush(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")
