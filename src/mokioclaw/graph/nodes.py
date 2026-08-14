"""Graph nodes for the planner→actor→verifier workflow (M4).

Each node follows the same pattern:
  1. Copy state into `working_state` (so tools can mutate it via closure).
  2. Build model + tools, run an internal tool loop.
  3. Return only the updated fields (LangGraph merges them back).

The verifier uses read-only tools only, so "checking" cannot secretly "fix".
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from mokioclaw.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    persist_history_summary,
)
from mokioclaw.agents.code_agent import run_code_agent
from mokioclaw.agents.search_agent import run_search_agent
from mokioclaw.graph.state import MokioGraphState
from mokioclaw.prompts.stage2 import ACTOR_PROMPT, VERIFIER_PROMPT
from mokioclaw.prompts.stage3 import PLANNER_PROMPT
from mokioclaw.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.registry import build_read_only_tools, build_tools
from mokioclaw.tools.todo_tool import (
    persist_todos,
    todo_items_from_strings,
    update_todo,
    write_todos,
)


# --------------------------------------------------------------------------
# 工具构造（闭包捕获 working_state，让工具读写图状态）
# --------------------------------------------------------------------------

def _todo_write_tool(working_state: dict) -> StructuredTool:
    def _write(todos, acceptance_criteria, verification_commands, plan_summary: str = ""):
        result = write_todos(todos, acceptance_criteria, verification_commands)
        if result["ok"]:
            working_state["plan_summary"] = plan_summary or "Task plan"
            working_state["todos"] = todo_items_from_strings(
                result["todos"], existing=working_state.get("todos", [])
            )
            working_state["acceptance_criteria"] = result["acceptance_criteria"]
            working_state["verification_commands"] = result["verification_commands"]
            persist_todos(
                working_state["runtime"],
                working_state["todos"],
                working_state["acceptance_criteria"],
                working_state["verification_commands"],
                working_state["plan_summary"],
            )
        return {
            **result,
            "plan_summary": working_state.get("plan_summary", ""),
            "todo_items": working_state.get("todos", []),
        }

    return StructuredTool.from_function(
        name="TodoWriteTool",
        func=_write,
        description=(
            "Publish or revise the plan. Args: todos (list of todo strings), "
            "acceptance_criteria (list), verification_commands (list), "
            "optional plan_summary (string)."
        ),
    )


def _todo_update_tool(working_state: dict) -> StructuredTool:
    def _update(todo_id: str, status: str, note: str = ""):
        result = update_todo(working_state.get("todos", []), todo_id, status, note)
        if result.get("ok"):
            working_state["todos"] = result["todos"]
        return result

    return StructuredTool.from_function(
        name="TodoUpdateTool",
        func=_update,
        description=(
            "Update a todo's status. Args: todo_id (e.g. todo-1), "
            "status (pending/in_progress/completed/blocked), optional note."
        ),
    )


def _build_planner_tools(working_state: dict) -> list[StructuredTool]:
    """Supervisor 的工具：TodoWrite + 委派 searchAgent/codeAgent。"""
    return [
        _todo_write_tool(working_state),
        StructuredTool.from_function(
            name="CallSearchAgentTool",
            func=lambda instruction: _call_search_agent_tool(working_state, instruction),
            description="Delegate web research to searchAgent. Args: instruction.",
        ),
        StructuredTool.from_function(
            name="CallCodeAgentTool",
            func=lambda instruction: _call_code_agent_tool(working_state, instruction),
            description="Delegate implementation work to codeAgent. Args: instruction.",
        ),
    ]


def _call_search_agent_tool(working_state: dict, instruction: str) -> dict[str, Any]:
    result = run_search_agent(working_state, instruction)
    existing_sources = list(working_state.get("sources", []))
    working_state["research_notes"] = _join_notes(
        working_state.get("research_notes", ""), result.get("summary", "")
    )
    working_state["sources"] = _dedupe_sources(
        existing_sources + list(result.get("sources", []))
    )
    handoff = {
        "from_agent": "planner",
        "to_agent": "searchAgent",
        "instruction": instruction,
        "result": result.get("summary", ""),
    }
    working_state["agent_handoffs"] = list(working_state.get("agent_handoffs", [])) + [handoff]
    return {
        "ok": True,
        "summary": result.get("summary", ""),
        "sources": working_state.get("sources", []),
        "queries": result.get("queries", []),
    }


def _call_code_agent_tool(working_state: dict, instruction: str) -> dict[str, Any]:
    result = run_code_agent(working_state, instruction)
    working_state["todos"] = result.get("todos", working_state.get("todos", []))
    working_state["code_agent_summary"] = result.get("summary", "")
    handoff = {
        "from_agent": "planner",
        "to_agent": "codeAgent",
        "instruction": instruction,
        "result": result.get("summary", ""),
    }
    working_state["agent_handoffs"] = list(working_state.get("agent_handoffs", [])) + [handoff]
    return {
        "ok": True,
        "summary": result.get("summary", ""),
        "todos": working_state.get("todos", []),
    }


def _execute_planner_tool(working_state: dict, call: dict[str, Any]) -> ToolMessage:
    name = call.get("name", "")
    args = call.get("args") or {}
    trace = working_state["runtime"].trace
    if trace:
        trace.record_tool_call("planner", name, args)
        if name == "CallSearchAgentTool":
            trace.record_handoff("planner", "searchAgent", str(args.get("instruction", "")))
        elif name == "CallCodeAgentTool":
            trace.record_handoff("planner", "codeAgent", str(args.get("instruction", "")))
    tools = {tool.name: tool for tool in _build_planner_tools(working_state)}
    tool = tools.get(name)
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if trace:
        trace.record_tool_result("planner", name, bool(result.get("ok")), str(result.get("error", "")))
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False, default=str),
        name=name,
        tool_call_id=call.get("id") or f"{name}-call",
    )


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------

def planner_node(state: MokioGraphState) -> dict[str, Any]:
    working_state: MokioGraphState = {**state}
    planner = create_model().bind_tools(_build_planner_tools(working_state))

    messages: list[Any] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working_state)),
    ]
    produced: list[Any] = []

    for _ in range(8):
        response = planner.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            tool_message = _execute_planner_tool(working_state, tc)
            produced.append(tool_message)
            messages.append(tool_message)

    return {
        "plan_summary": working_state.get("plan_summary", ""),
        "todos": working_state.get("todos", []),
        "acceptance_criteria": working_state.get("acceptance_criteria", []),
        "verification_commands": working_state.get("verification_commands", []),
        "research_notes": working_state.get("research_notes", ""),
        "sources": working_state.get("sources", []),
        "agent_handoffs": working_state.get("agent_handoffs", []),
        "code_agent_summary": working_state.get("code_agent_summary", ""),
        "messages": produced,
    }


def actor_node(state: MokioGraphState) -> dict[str, Any]:
    working_state: MokioGraphState = {**state}
    update_tool = _todo_update_tool(working_state)
    tools = build_tools(working_state["runtime"]) + [update_tool]
    tool_map = {t.name: t for t in tools}
    actor = create_model().bind_tools(tools)

    messages: list[Any] = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=_actor_input(working_state)),
    ]
    produced: list[Any] = []

    for _ in range(10):
        response = actor.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name = tc.get("name", "")
            if working_state["runtime"].trace:
                working_state["runtime"].trace.record_tool_call("actor", name, tc.get("args", {}))
            tool = tool_map.get(name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(tc.get("args", {}))
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if working_state["runtime"].trace:
                working_state["runtime"].trace.record_tool_result(
                    "actor", name, bool(result.get("ok")), str(result.get("error", ""))
                )
            produced.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )
            messages.append(produced[-1])

    return {
        "messages": produced,
        "todos": working_state.get("todos", []),
    }


def verifier_node(state: MokioGraphState) -> dict[str, Any]:
    tools = build_read_only_tools(state["runtime"])
    tool_map = {t.name: t for t in tools}
    verifier = create_model().bind_tools(tools)

    messages: list[Any] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=_verifier_input(state)),
    ]
    produced: list[Any] = []

    for _ in range(8):
        response = verifier.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name = tc.get("name", "")
            if state["runtime"].trace:
                state["runtime"].trace.record_tool_call("verifier", name, tc.get("args", {}))
            tool = tool_map.get(name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(tc.get("args", {}))
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if state["runtime"].trace:
                state["runtime"].trace.record_tool_result(
                    "verifier", name, bool(result.get("ok")), str(result.get("error", ""))
                )
            produced.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )
            messages.append(produced[-1])

    parsed = _extract_json(_last_ai_content(produced)) or {
        "passed": False,
        "reason": "verifier did not return valid JSON",
        "checks": [],
        "recommended_next_instruction": "Return valid verifier JSON after inspecting the result.",
    }
    passed = bool(parsed.get("passed"))
    attempts = state.get("attempts", 0) + 1

    return {
        "passed": passed,
        "attempts": attempts,
        "verifier_summary": str(parsed.get("reason") or ""),
        "verification_checks": _normalize_checks(parsed.get("checks")),
        "recommended_next_instruction": str(parsed.get("recommended_next_instruction") or ""),
        "messages": produced,
        "context_next_node": verifier_route({**state, "passed": passed, "attempts": attempts}),
    }


def verifier_route(state: MokioGraphState) -> str:
    if state.get("passed"):
        return "final"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "final"
    return "planner"


def final_node(state: MokioGraphState) -> dict[str, Any]:
    status = "PASSED" if state.get("passed") else "FAILED"
    todos_text = "\n".join(
        f"- [{t.get('status', '')}] {t.get('content', '')}" for t in state.get("todos", [])
    )
    checks_text = "\n".join(
        f"- {c.get('name', 'check')}: {'PASS' if c.get('passed') else 'FAIL'} - {c.get('detail', '')}"
        for c in state.get("verification_checks", [])
    )
    final_answer = (
        f"Workflow finished: {status}\n\n"
        f"Plan: {state.get('plan_summary', '')}\n\n"
        f"Todos:\n{todos_text or '(none)'}\n\n"
        f"Verifier: {state.get('verifier_summary', '')}\n\n"
        f"Checks:\n{checks_text or '(none)'}"
    )
    return {"final_answer": final_answer}


# --------------------------------------------------------------------------
# 输入构造
# --------------------------------------------------------------------------

def _planner_input(state: MokioGraphState) -> str:
    memory = build_layered_memory(state, node="planner")
    parts = [f"Task: {state['task']}", f"Attempt: {state.get('attempts', 0) + 1}"]
    if state.get("attempts", 0) > 0 and state.get("recommended_next_instruction"):
        parts.append(
            "Previous verifier failed. Revise the plan to fix only this:\n"
            + state["recommended_next_instruction"]
        )
    parts.append("Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory))
    return "\n\n".join(parts)


def _actor_input(state: MokioGraphState) -> str:
    memory = build_layered_memory(state, node="actor")
    if not state.get("todos"):
        return state["task"] + "\n\nLayered memory snapshot:\n" + format_layered_memory_for_prompt(memory)
    todo_lines = "\n".join(
        f"- {t['id']} [{t.get('status', 'pending')}] {t['content']}" for t in state["todos"]
    )
    criteria_lines = "\n".join(f"- {c}" for c in state.get("acceptance_criteria", []))
    command_lines = "\n".join(f"- {c}" for c in state.get("verification_commands", []))
    return (
        f"Task: {state['task']}\n\n"
        f"Plan summary: {state.get('plan_summary', '')}\n\n"
        f"Todos:\n{todo_lines}\n\n"
        f"Acceptance criteria:\n{criteria_lines}\n\n"
        f"Verification commands:\n{command_lines}\n\n"
        "Execute the plan. Update todo progress as you go.\n\n"
        "Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory)
    )


def _verifier_input(state: MokioGraphState) -> str:
    memory = build_layered_memory(state, node="verifier")
    parts = [f"Task: {state['task']}"]
    todo_lines = "\n".join(f"- {t['id']} {t['content']}" for t in state.get("todos", []))
    if todo_lines:
        parts.append(f"Todos:\n{todo_lines}")
    criteria_lines = "\n".join(f"- {c}" for c in state.get("acceptance_criteria", []))
    if criteria_lines:
        parts.append(f"Acceptance criteria:\n{criteria_lines}")
    command_lines = "\n".join(f"- {c}" for c in state.get("verification_commands", []))
    if command_lines:
        parts.append(f"Verification commands:\n{command_lines}")
    parts.append("Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory))
    parts.append("Inspect the workspace with read-only tools and return only verifier JSON.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_checks(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    checks: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            checks.append(
                {
                    "name": str(item.get("name") or "check"),
                    "passed": bool(item.get("passed")),
                    "detail": str(item.get("detail") or ""),
                }
            )
    return checks


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""


# --------------------------------------------------------------------------
# Context Monitor / Compressor（M5）
# --------------------------------------------------------------------------

DEFAULT_CONTEXT_TOKEN_LIMIT = 400000


def get_context_token_limit() -> int:
    load_dotenv()
    raw = os.getenv("MOKIO_CONTEXT_TOKEN_LIMIT", str(DEFAULT_CONTEXT_TOKEN_LIMIT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_TOKEN_LIMIT
    return value if value > 0 else DEFAULT_CONTEXT_TOKEN_LIMIT


def estimate_context_tokens(state: MokioGraphState) -> int:
    messages = list(state.get("messages", []))
    payload = build_layered_memory(state, node="context_monitor")
    payload_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str))
    try:
        model = create_model()
        return int(model.get_num_tokens_from_messages(messages + [payload_message]))
    except Exception:
        text = "\n".join(_message_text(message) for message in messages)
        text += "\n" + str(payload_message.content)
        return max(1, len(text) // 4)


def context_monitor_node(state: MokioGraphState) -> dict[str, Any]:
    token_limit = get_context_token_limit()
    token_count = estimate_context_tokens(state)
    should_compress = token_count >= token_limit
    next_node = state.get("context_next_node") or "verifier"
    return {
        "context_token_count": token_count,
        "context_token_limit": token_limit,
        "context_should_compress": should_compress,
        "context_next_node": next_node,
    }


def context_monitor_route(state: MokioGraphState) -> str:
    if state.get("context_should_compress"):
        return "context_compressor"
    return state.get("context_next_node") or "verifier"


def context_compressor_node(state: MokioGraphState) -> dict[str, Any]:
    before_tokens = state.get("context_token_count") or estimate_context_tokens(state)
    before_messages = list(state.get("messages", []))
    memory = build_layered_memory(state, node="context_compressor")
    compressed = _compress_context_with_model(state)
    summary = _format_compressed_context(compressed, state)
    summary_message = AIMessage(content=summary)
    persist_history_summary(state["runtime"], summary)

    post_state: MokioGraphState = {
        **state,
        "messages": [summary_message],
        "context_summary": summary,
        "history_summary": summary,
        "memory_snapshot": build_layered_memory(
            {**state, "context_summary": summary, "history_summary": summary},
            node="context_compressor",
        ),
    }
    after_tokens = estimate_context_tokens(post_state)
    compression_event = {
        "before_tokens": int(before_tokens),
        "after_tokens": int(after_tokens),
        "removed_messages": len(before_messages),
        "summary": _short_text(summary, 1200),
        "next_node": state.get("context_next_node", "verifier"),
    }
    events = list(state.get("compression_events", [])) + [compression_event]

    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary_message],
        "context_summary": summary,
        "context_token_count": after_tokens,
        "context_should_compress": False,
        "history_summary": summary,
        "memory_snapshot": post_state.get("memory_snapshot", {}),
        "compression_events": events,
    }


def context_compressor_route(state: MokioGraphState) -> str:
    return state.get("context_next_node") or "verifier"


def _compress_context_with_model(state: MokioGraphState) -> dict[str, Any]:
    memory = build_layered_memory(state, node="context_compressor")
    payload = {
        "context_summary": state.get("context_summary", ""),
        "memory": memory,
        "messages": [_message_snapshot(message) for message in state.get("messages", [])],
    }
    messages = [
        SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    try:
        response = create_model().invoke(messages)
        parsed = _extract_json(str(response.content))
        if parsed:
            return parsed
    except Exception as exc:
        return _fallback_compression(state, error=f"{type(exc).__name__}: {exc}")
    return _fallback_compression(state, error="compressor model did not return valid JSON")


def _fallback_compression(state: MokioGraphState, *, error: str = "") -> dict[str, Any]:
    return {
        "summary": _short_text(
            "\n\n".join(
                [
                    state.get("context_summary", ""),
                    state.get("verifier_summary", ""),
                    state.get("last_error", ""),
                ]
            ),
            2400,
        ),
        "active_goal": state.get("task", ""),
        "completed_work": state.get("verifier_summary", ""),
        "open_todos": [
            todo.get("content", "")
            for todo in state.get("todos", [])
            if todo.get("status") != "completed"
        ],
        "important_files": _important_files_from_state(state),
        "tool_findings": _short_text(state.get("last_error", ""), 1200),
        "sources": [],
        "next_steps": state.get("context_next_node", ""),
        "risks": error,
    }


def _format_compressed_context(compressed: dict[str, Any], state: MokioGraphState) -> str:
    payload = {
        "type": "mokio_context_summary",
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "attempts": state.get("attempts", 0),
        "passed": state.get("passed"),
        "compression": compressed,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _message_snapshot(message: Any) -> dict[str, str]:
    return {
        "type": type(message).__name__,
        "name": str(getattr(message, "name", "") or ""),
        "content": _short_text(_message_text(message), 2000),
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _important_files_from_state(state: MokioGraphState) -> list[str]:
    files: list[str] = []
    for command in state.get("verification_commands", []):
        files.extend(re.findall(r"[\w./\\-]+\.(?:py|html|css|js|json|md|txt)", command))
    for text in [state.get("verifier_summary", ""), state.get("last_error", "")]:
        files.extend(re.findall(r"[\w./\\-]+\.(?:py|html|css|js|json|md|txt)", text))
    seen: set[str] = set()
    deduped = []
    for item in files:
        normalized = item.strip("'\"")
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _join_notes(existing: str, new: str) -> str:
    if not existing:
        return new
    if not new:
        return existing
    return existing + "\n\n" + new


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
