"""Graph nodes for the planner→actor→verifier workflow (M4).

Each node follows the same pattern:
  1. Copy state into `working_state` (so tools can mutate it via closure).
  2. Build model + tools, run an internal tool loop.
  3. Return only the updated fields (LangGraph merges them back).

The verifier uses read-only tools only, so "checking" cannot secretly "fix".
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mokioclaw.graph.state import MokioGraphState
from mokioclaw.prompts.stage2 import ACTOR_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
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


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------

def planner_node(state: MokioGraphState) -> dict[str, Any]:
    working_state: MokioGraphState = {**state}
    write_tool = _todo_write_tool(working_state)
    planner = create_model().bind_tools([write_tool])

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
            try:
                result = write_tool.invoke(tc.get("args", {}))
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            produced.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )
            messages.append(produced[-1])

    return {
        "plan_summary": working_state.get("plan_summary", ""),
        "todos": working_state.get("todos", []),
        "acceptance_criteria": working_state.get("acceptance_criteria", []),
        "verification_commands": working_state.get("verification_commands", []),
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
            tool = tool_map.get(name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(tc.get("args", {}))
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
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
            tool = tool_map.get(name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(tc.get("args", {}))
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
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
    parts = [f"Task: {state['task']}", f"Attempt: {state.get('attempts', 0) + 1}"]
    if state.get("attempts", 0) > 0 and state.get("recommended_next_instruction"):
        parts.append(
            "Previous verifier failed. Revise the plan to fix only this:\n"
            + state["recommended_next_instruction"]
        )
    return "\n\n".join(parts)


def _actor_input(state: MokioGraphState) -> str:
    if not state.get("todos"):
        return state["task"]
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
        "Execute the plan. Update todo progress as you go."
    )


def _verifier_input(state: MokioGraphState) -> str:
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
