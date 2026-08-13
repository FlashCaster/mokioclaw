"""Agent loop — planner then actor (M3).

stream_agent_events() runs two phases:

1. Planner: model + TodoWriteTool → structured plan
   (plan_summary / todos / acceptance_criteria / verification_commands)
2. Actor: model + file/bash/grep/todo-update tools → execute the plan

Each phase yields structured events for the CLI layer to display.
"""

import json
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState
from mokioclaw.prompts.stage2 import ACTOR_PROMPT, PLANNER_PROMPT
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.registry import build_tools
from mokioclaw.tools.todo_tool import (
    persist_todos,
    todo_items_from_strings,
    update_todo,
    write_todos,
)


def _todo_write_tool(plan: dict, state: RuntimeState) -> StructuredTool:
    """TodoWriteTool：闭包捕获 plan 字典，planner 调用后更新计划并落盘 TODO.md。"""

    def _write(todos, acceptance_criteria, verification_commands, plan_summary: str = ""):
        result = write_todos(todos, acceptance_criteria, verification_commands)
        if result["ok"]:
            plan["plan_summary"] = plan_summary or "Task plan"
            plan["todos"] = todo_items_from_strings(result["todos"], existing=plan.get("todos", []))
            plan["acceptance_criteria"] = result["acceptance_criteria"]
            plan["verification_commands"] = result["verification_commands"]
            persist_todos(
                state,
                plan["todos"],
                plan["acceptance_criteria"],
                plan["verification_commands"],
                plan["plan_summary"],
            )
        return {
            **result,
            "plan_summary": plan.get("plan_summary", ""),
            "todo_items": plan.get("todos", []),
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


def _todo_update_tool(plan: dict) -> StructuredTool:
    """TodoUpdateTool：闭包捕获 plan，actor 调用后更新某个 todo 的状态。"""

    def _update(todo_id: str, status: str, note: str = ""):
        result = update_todo(plan.get("todos", []), todo_id, status, note)
        if result.get("ok"):
            plan["todos"] = result["todos"]
        return result

    return StructuredTool.from_function(
        name="TodoUpdateTool",
        func=_update,
        description=(
            "Update a todo's status. Args: todo_id (e.g. todo-1), "
            "status (pending/in_progress/completed/blocked), optional note."
        ),
    )


def _actor_input(task: str, plan: dict) -> str:
    """把计划注入 actor 的输入，让它知道要做什么、怎么算完成。"""
    if not plan.get("todos"):
        return task
    todo_lines = "\n".join(
        f"- {t['id']} [{t.get('status', 'pending')}] {t['content']}" for t in plan["todos"]
    )
    criteria_lines = "\n".join(f"- {c}" for c in plan.get("acceptance_criteria", []))
    command_lines = "\n".join(f"- {c}" for c in plan.get("verification_commands", []))
    return (
        f"Task: {task}\n\n"
        f"Plan summary: {plan.get('plan_summary', '')}\n\n"
        f"Todos:\n{todo_lines}\n\n"
        f"Acceptance criteria:\n{criteria_lines}\n\n"
        f"Verification commands:\n{command_lines}\n\n"
        "Execute the plan. Update todo progress as you go."
    )


def _extract_text(response) -> str:
    """从 AIMessage 提取纯文本内容（兼容 str 与 content blocks）。"""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def stream_agent_events(
    task: str,
    state: RuntimeState,
    *,
    max_loops: int = 10,
    plan_loops: int = 8,
) -> Iterator[dict]:
    """Run planner → actor and yield structured events.

    Event types yielded:
        {"type": "tool_call", "name": str, "args": dict, "node": "planner"|"actor"}
        {"type": "tool_result", "name": str, "result": dict, "node": ...}
        {"type": "plan_snapshot", "plan_summary", "todos", "acceptance_criteria", "verification_commands"}
        {"type": "ai_message", "content": str, "loop": int}
        {"type": "final_answer", "content": str}
        {"type": "error", "message": str}
    """
    model = create_model()
    plan: dict = {"plan_summary": "", "todos": [], "acceptance_criteria": [], "verification_commands": []}
    write_tool = _todo_write_tool(plan, state)
    update_tool = _todo_update_tool(plan)

    # Phase 1: Planner —— 产出结构化计划
    planner = model.bind_tools([write_tool])
    planner_messages = [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=task)]
    for _ in range(plan_loops):
        response = planner.invoke(planner_messages)
        planner_messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            yield {"type": "tool_call", "name": name, "args": args, "node": "planner"}
            try:
                result = write_tool.invoke(args)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
            yield {"type": "tool_result", "name": name, "result": result, "node": "planner"}
            planner_messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )

    yield {
        "type": "plan_snapshot",
        "plan_summary": plan["plan_summary"],
        "todos": plan["todos"],
        "acceptance_criteria": plan["acceptance_criteria"],
        "verification_commands": plan["verification_commands"],
    }

    # Phase 2: Actor —— 按计划执行
    tools = build_tools(state) + [update_tool]
    actor = model.bind_tools(tools)
    messages = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=_actor_input(task, plan)),
    ]
    tool_map = {t.name: t for t in tools}

    for loop_idx in range(max_loops):
        response = actor.invoke(messages)
        messages.append(response)

        text_content = _extract_text(response)
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            yield {"type": "final_answer", "content": text_content}
            return

        yield {"type": "ai_message", "content": text_content, "loop": loop_idx + 1}

        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            yield {"type": "tool_call", "name": name, "args": args, "node": "actor"}

            tool = tool_map.get(name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}

            yield {"type": "tool_result", "name": name, "result": result, "node": "actor"}

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"],
                )
            )

    yield {
        "type": "error",
        "message": f"Agent reached max loops ({max_loops}) without finishing.",
    }
