"""searchAgent — focused research specialist (M6)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from mokioclaw.graph.state import MokioGraphState
from mokioclaw.prompts.stage3 import SEARCH_AGENT_PROMPT
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.web_search_tool import build_web_search_tool


def run_search_agent(
    state: MokioGraphState,
    instruction: str,
    *,
    max_loops: int = 4,
) -> dict[str, Any]:
    model = create_model()
    search_agent = model.bind_tools([build_web_search_tool()])
    messages = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(
            content=(
                f"Task: {state['task']}\n\n"
                f"Planner instruction:\n{instruction}\n\n"
                f"Existing research notes:\n{state.get('research_notes', '')}\n\n"
                "Search as needed and finish with a concise research summary plus source URLs."
            )
        ),
    ]

    produced: list[Any] = []
    queries: list[str] = []
    sources: list[dict[str, Any]] = []
    answers: list[str] = []

    for _ in range(max_loops):
        response = search_agent.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            args = call.get("args") or {}
            query = str(args.get("query", ""))
            if query:
                queries.append(query)
            try:
                result = build_web_search_tool().invoke(args)
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            tm = ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=call.get("name"),
                tool_call_id=call.get("id") or f"{call.get('name')}-call",
            )
            produced.append(tm)
            messages.append(tm)
            if isinstance(result, dict):
                if result.get("answer"):
                    answers.append(str(result["answer"]))
                for item in result.get("results", []) or []:
                    if isinstance(item, dict):
                        sources.append(item)

    summary = _last_ai_content(produced) or "\n".join(answers)
    return {
        "ok": True,
        "summary": summary,
        "queries": queries,
        "sources": _dedupe_sources(sources),
    }


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for source in sources:
        url = str(source.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(source)
    return deduped


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""
