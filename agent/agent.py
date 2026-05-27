"""
Med Research Agent — autonomous substance research via AISuite + MCP tools.

The four pillars of this agent:

  PILLAR 1 — PLANNING   (agent/pipeline.py)
    Structured, parallel initial data gathering across all 7 sources.
    Resolves drug names → fans out tools → returns pre-fetched context.

  PILLAR 2 — TOOL USE   (mcp_server/server.py + MCP client)
    Claude decides which tools to call, in what order, with what parameters.
    All tools are served by the MCP server over stdio.

  PILLAR 3 — REFLECTION (agent/reflection.py)
    After the draft is written, a critic LLM call evaluates completeness
    and evidence quality. If gaps are found (score < 70), one revision
    pass fills them before the report is finalised.

  PILLAR 4 — EVALS      (agent/evals.py)
    Deterministic scoring of source coverage, section completeness, and
    evidence tier distribution. Appended to every saved report.

  BONUS — MEMORY        (agent/memory.py)
    7-day persistent cache of prior research per substance. Loaded as
    context at the start of each run; updated after the report is saved.

Usage:
    python -m agent.agent --query "BPC-157"
    python agent/agent.py --query "spironolactone HRT feminizing"
    python agent/agent.py --query "testosterone cypionate + anastrozole"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import aisuite as ai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.evals import EvalEngine
from agent.memory import ResearchMemory
from agent.pipeline import ResearchPipeline
from agent.reflection import ReflectionEngine

load_dotenv()

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports" / "output"
PROMPTS_DIR = Path(__file__).parent / "prompts"

_REVISION_SCORE_THRESHOLD = 70  # reflection scores below this trigger a revision pass

_DEFAULT_SYSTEM = """\
You are a rigorous medical research synthesis agent.

Your job is to gather information from multiple clinical and community sources
about a given substance and produce a comprehensive, structured research report.

Rules:
- Always start by calling resolve_drug_name to normalize the input.
- Identify the substance category (HRT, peptide, prescription drug, supplement,
  performance compound) before deciding which sources to prioritize.
- Pull clinical sources (PubMed, OpenFDA, DailyMed, ClinicalTrials) first,
  then community sources (Reddit, forums).
- When clinical evidence and community experience conflict, surface BOTH —
  never silently drop one side.
- Flag black box warnings and serious adverse-event patterns prominently.
- Note when evidence quality is weak: small studies, animal-only data, no RCTs.
- Label every finding by its source and credibility tier [T1]–[T5].
- Never give personal medical advice — present findings as research synthesis only.
"""


def _load_text(path: Path) -> str:
    if path.exists() and path.stat().st_size > 4:
        return path.read_text(encoding="utf-8").strip()
    return ""


def _to_tool_schema(tool: Any) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class MedResearchAgent:
    MODEL = "anthropic:claude-sonnet-4-6"

    def __init__(self) -> None:
        self.client = ai.Client()
        self.system_prompt = _load_text(PROMPTS_DIR / "system_prompt.txt") or _DEFAULT_SYSTEM
        self.report_template = _load_text(PROMPTS_DIR / "report_template.txt")
        # Pillar engines (stateless — instantiated once per agent)
        self._reflection = ReflectionEngine(self.client, self.MODEL)
        self._evals = EvalEngine()
        self._memory = ResearchMemory()

    # ── public entry point ────────────────────────────────────────────────────

    async def research(self, query: str) -> str:
        """Run the full four-pillar research pipeline for a substance query."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(ROOT),
            env=dict(os.environ),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_list = await session.list_tools()
                tools = [_to_tool_schema(t) for t in tools_list.tools]

                # ── PILLAR 1: PLANNING ────────────────────────────────────────
                # Check long-term memory first; merge prior findings as context
                prior = self._memory.get(query)
                if prior:
                    print(f"[memory] loaded prior research for: {query}")

                print(f"[pipeline] running initial research for: {query}")
                pipeline = ResearchPipeline(session)
                initial_data = await pipeline.run(query)
                if prior:
                    initial_data["prior_research"] = prior

                # ── PILLAR 2: TOOL USE ────────────────────────────────────────
                print("[agent] entering agentic loop …")
                draft = await self._agentic_loop(query, initial_data, session, tools)

                # ── PILLAR 3: REFLECTION ──────────────────────────────────────
                critique = self._reflection.critique(draft)
                print(
                    f"[reflection] verdict={critique['quality_verdict']}  "
                    f"score={critique['score']}/100"
                )

                if (
                    critique["quality_verdict"] == "NEEDS_REVISION"
                    and critique["score"] < _REVISION_SCORE_THRESHOLD
                ):
                    print("[reflection] running revision pass …")
                    feedback = self._reflection.format_feedback(critique)
                    draft = await self._revision_pass(
                        draft, feedback, initial_data, session, tools
                    )

                # ── PILLAR 4: EVALS ───────────────────────────────────────────
                eval_scores = self._evals.score(
                    draft, initial_data, reflection_score=critique["score"]
                )
                print(
                    f"[evals] overall={eval_scores['overall']}/100  "
                    f"grade={eval_scores['grade']}"
                )
                eval_block = self._evals.format_block(eval_scores)
                final_report = draft + eval_block

                # ── MEMORY UPDATE ─────────────────────────────────────────────
                self._memory.save(
                    query,
                    {
                        "canonical_name": initial_data.get("canonical_name"),
                        "eval_scores": eval_scores,
                        "reflection_score": critique["score"],
                    },
                )

                return self._save_report(query, final_report)

    # ── agentic loop (Pillar 2) ───────────────────────────────────────────────

    async def _agentic_loop(
        self,
        query: str,
        initial_data: dict,
        session: ClientSession,
        tools: list[dict],
        extra_context: str = "",
    ) -> str:
        system = self.system_prompt
        if self.report_template:
            system += f"\n\n## Report Format\n\n{self.report_template}"

        context_json = json.dumps(initial_data, indent=2, default=str)
        user_content = (
            f"Research this substance: **{query}**\n\n"
            f"Initial data gathered by the pipeline:\n"
            f"```json\n{context_json}\n```\n"
        )
        if extra_context:
            user_content += f"\n\n{extra_context}"
        user_content += "\n\nUse the tools to fill any gaps, then write the complete report."

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        iteration = 0
        while True:
            iteration += 1
            print(f"[agent] iteration {iteration} …")

            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                print(f"[tool] {tc.function.name}")
                result = await self._call_tool(session, tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

    # ── revision pass (Pillar 3 — bounded to one pass) ───────────────────────

    async def _revision_pass(
        self,
        draft: str,
        feedback: str,
        initial_data: dict,
        session: ClientSession,
        tools: list[dict],
    ) -> str:
        extra = (
            f"## Draft Report (to be revised)\n\n{draft}\n\n"
            f"{feedback}\n\n"
            "Address every gap listed above, then output the revised complete report."
        )
        return await self._agentic_loop(
            initial_data.get("query", ""),
            initial_data,
            session,
            tools,
            extra_context=extra,
        )

    # ── MCP tool call helper ──────────────────────────────────────────────────

    async def _call_tool(self, session: ClientSession, tc: Any) -> Any:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = await session.call_tool(tc.function.name, args)
            if result.content:
                texts = [b.text for b in result.content if hasattr(b, "text")]
                raw = "\n".join(texts)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            return {}
        except Exception as exc:
            return {"error": str(exc), "tool": tc.function.name}

    # ── file output ───────────────────────────────────────────────────────────

    def _save_report(self, query: str, content: str) -> str:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() or c in " _-" else "_" for c in query.lower())
        slug = slug.replace(" ", "_")[:60].strip("_")
        filename = f"{slug}_{date.today()}.md"
        path = REPORTS_DIR / filename
        path.write_text(content, encoding="utf-8")
        print(f"\nReport saved → {path.relative_to(ROOT)}")
        return content


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Med Research Agent")
    parser.add_argument("--query", "-q", required=True, help="Substance to research")
    args = parser.parse_args()

    agent = MedResearchAgent()
    report = asyncio.run(agent.research(args.query))
    print("\n" + "=" * 72)
    print(report)


if __name__ == "__main__":
    main()
