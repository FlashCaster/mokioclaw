"""codeAgent — focused implementation specialist (M6)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mokioclaw.graph.memory import build_layered_memory, format_layered_memory_for_prompt
from mokioclaw.graph.state import MokioGraphState
from mokioclaw.prompts.stage3 import CODE_AGENT_PROMPT
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.registry import build_tools
from mokioclaw.tools.todo_tool import persist_todos, update_todo


def run_code_agent(
    state: MokioGraphState,
    instruction: str,
    *,
    max_loops: int = 10,
) -> dict[str, Any]:
    runtime = state["runtime"]
    todos = [dict(todo) for todo in state.get("todos", [])]
    memory = build_layered_memory({**state, "todos": todos}, node="codeAgent")
    model = create_model()
    code_agent = model.bind_tools(build_tools(runtime) + [_build_todo_update_tool(todos)])

    messages = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=_code_agent_input(state, instruction, memory)),
    ]
    produced: list[Any] = []

    for _ in range(max_loops):
        response = code_agent.invoke(messages)
        produced.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            tool_result, todos = _execute_code_agent_tool(runtime, todos, call)
            if call.get("name") == "TodoUpdateTool":
                persist_todos(
                    runtime,
                    todos,
                    state.get("acceptance_criteria", []),
                    state.get("verification_commands", []),
                    state.get("plan_summary", ""),
                )
            produced.append(tool_result)
            messages.append(tool_result)
    else:
        produced.append(AIMessage(content="codeAgent stopped after the maximum tool loop count."))

    summary = _last_ai_content(produced)
    return {"ok": True, "summary": summary, "todos": todos or state.get("todos", [])}


def _execute_code_agent_tool(runtime, todos: list[dict[str, Any]], call: dict[str, Any]):
    name = call.get("name", "")
    args = call.get("args") or {}
    if name == "TodoUpdateTool":
        result = update_todo(todos, args.get("todo_id", ""), args.get("status", ""), args.get("note", ""))
        if result.get("ok"):
            todos = result["todos"]
    else:
        tools = {tool.name: tool for tool in build_tools(runtime)}
        tool = tools.get(name)
        if tool is None:
            result = {"ok": False, "error": f"unknown tool: {name}"}
        else:
            try:
                result = tool.invoke(args)
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return (
        ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            name=name,
            tool_call_id=call.get("id") or f"{name}-call",
        ),
        todos,
    )


def _build_todo_update_tool(todos: list[dict[str, Any]]) -> StructuredTool:
    return StructuredTool.from_function(
        name="TodoUpdateTool",
        func=lambda todo_id, status, note="": update_todo(todos, todo_id, status, note),
        description="Update one existing todo status. Args: todo_id, status, optional note.",
    )


def _code_agent_input(state: MokioGraphState, instruction: str, memory: dict[str, Any]) -> str:
    parts = [
        f"Task: {state['task']}",
        f"Planner instruction:\n{instruction}",
    ]
    parts.append("Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory))
    return "\n\n".join(parts)


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""
