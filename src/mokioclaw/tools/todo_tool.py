"""TODO tool: structured plan management for planner and actor.

M3 规划器：planner 用 TodoWriteTool 产出计划（todos + 验收标准 + 验证命令），
actor 用 TodoUpdateTool 更新进度；persist_todos 把计划落到工作区 TODO.md。
"""

from __future__ import annotations

import json
from typing import Any

from mokioclaw.core.paths import resolve_workspace_path
from mokioclaw.core.state import RuntimeState

VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}
TODO_FILE = "TODO.md"


def _normalize_items(items: Any) -> list[str]:
    """把 LLM 可能返回的各种形状（str/list/dict/json）规整成字符串列表。"""
    if isinstance(items, str):
        stripped = items.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            # 不是 JSON，按 markdown 列表逐行拆
            return [line.strip("- ").strip() for line in stripped.splitlines() if line.strip()]
        return _normalize_items(decoded)
    if isinstance(items, dict):
        value = items.get("content") or items.get("description") or items.get("title") or items.get("text")
        if value:
            return [str(value).strip()]
        normalized: list[str] = []
        for key, item in items.items():
            child = _normalize_items(item)
            normalized.extend(child or [str(key).strip()])
        return [x for x in normalized if x]
    if isinstance(items, list):
        normalized: list[str] = []
        for item in items:
            normalized.extend(_normalize_items(item))
        return [x for x in normalized if x]
    if items is not None and str(items).strip():
        return [str(items).strip()]
    return []


def write_todos(
    todos: Any,
    acceptance_criteria: Any,
    verification_commands: Any,
) -> dict[str, Any]:
    """规范化计划三要素；ok 表示三要素都非空。"""
    cleaned_todos = _normalize_items(todos)
    cleaned_criteria = _normalize_items(acceptance_criteria)
    cleaned_commands = _normalize_items(verification_commands)
    return {
        "ok": bool(cleaned_todos and cleaned_criteria and cleaned_commands),
        "todos": cleaned_todos,
        "acceptance_criteria": cleaned_criteria,
        "verification_commands": cleaned_commands,
    }


def todo_items_from_strings(
    todos: list[str],
    *,
    existing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """字符串列表 → 带 id/status/note 的 todo 字典列表（修订时保留已有进度）。"""
    existing_by_content = {t.get("content", ""): t for t in existing or []}
    items: list[dict[str, Any]] = []
    for idx, todo in enumerate(todos, start=1):
        prev = existing_by_content.get(todo, {})
        items.append(
            {
                "id": str(prev.get("id") or f"todo-{idx}"),
                "content": todo,
                "status": str(prev.get("status") or "pending"),
                "note": str(prev.get("note") or ""),
            }
        )
    return items


def update_todo(
    todos: list[dict[str, Any]],
    todo_id: str,
    status: str,
    note: str = "",
) -> dict[str, Any]:
    """更新单个 todo 的状态/备注，返回更新后的完整列表。"""
    if status not in VALID_TODO_STATUSES:
        return {
            "ok": False,
            "error": f"status must be one of: {', '.join(sorted(VALID_TODO_STATUSES))}",
            "todos": todos,
        }

    updated: list[dict[str, Any]] = []
    found = False
    for todo in todos:
        item = dict(todo)
        if item.get("id") == todo_id:
            item["status"] = status
            item["note"] = note
            found = True
        updated.append(item)

    if not found:
        return {"ok": False, "error": f"unknown todo_id: {todo_id}", "todos": todos}
    return {"ok": True, "todo_id": todo_id, "status": status, "note": note, "todos": updated}


def render_todo_markdown(
    todos: list[dict[str, Any]],
    acceptance_criteria: list[str],
    verification_commands: list[str],
    plan_summary: str = "",
) -> str:
    """渲染 TODO.md 内容。"""
    lines = ["# MokioClaw Todo", ""]
    if plan_summary:
        lines.extend(["## Plan", "", plan_summary, ""])
    lines.extend(["## Todos", ""])
    if todos:
        for todo in todos:
            status = str(todo.get("status", "pending"))
            box = {"pending": " ", "in_progress": "-", "completed": "x", "blocked": "!"}.get(status, " ")
            note = str(todo.get("note", ""))
            note_text = f" — {note}" if note else ""
            lines.append(
                f"- [{box}] **{todo.get('id', '')}** `{status}` {todo.get('content', '')}{note_text}"
            )
    else:
        lines.append("- [ ] No todos yet.")
    if acceptance_criteria:
        lines.extend(["", "## Acceptance Criteria", ""])
        lines.extend(f"- {item}" for item in acceptance_criteria)
    if verification_commands:
        lines.extend(["", "## Verification Commands", ""])
        lines.extend(f"- `{command}`" for command in verification_commands)
    lines.append("")
    return "\n".join(lines)


def persist_todos(
    state: RuntimeState,
    todos: list[dict[str, Any]],
    acceptance_criteria: list[str] | None = None,
    verification_commands: list[str] | None = None,
    plan_summary: str = "",
) -> dict[str, Any]:
    """把计划持久化到工作区 TODO.md（沙箱内，路径逃逸会被拦截）。"""
    try:
        target = resolve_workspace_path(state.workspace, TODO_FILE)
        content = render_todo_markdown(
            todos,
            acceptance_criteria or [],
            verification_commands or [],
            plan_summary,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": TODO_FILE, "lines": len(content.splitlines()), "todos": todos}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"persist failed: {e}"}
