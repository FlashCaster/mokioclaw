"""Three-layer memory assembly (M5).

Layers:
  - Rules: fixed, never rewritten by the agent (assembled by the runtime).
  - Working Memory: current task state, read from the graph state.
  - History Summary Store: HISTORY_SUMMARY.md + NOTEPAD.md + context summary.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mokioclaw.core.paths import resolve_workspace_path
from mokioclaw.core.state import RuntimeState
from mokioclaw.tools.notepad_tool import NOTEPAD_FILE, read_notepad

HISTORY_SUMMARY_FILE = "HISTORY_SUMMARY.md"

RULES_LAYER = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}

MAX_TEXT_CHARS = {
    "notepad": 1800,
    "history_summary": 2200,
    "context_summary": 1600,
    "verifier_summary": 1000,
    "last_error": 1400,
}


def build_layered_memory(state: dict[str, Any], *, node: str = "graph") -> dict[str, Any]:
    runtime: RuntimeState = state["runtime"]
    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)

    working_memory = {
        "node": node,
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "verifier_summary": _short_text(state.get("verifier_summary", ""), MAX_TEXT_CHARS["verifier_summary"]),
        "verification_checks": state.get("verification_checks", []),
        "last_error": _short_text(state.get("last_error", ""), MAX_TEXT_CHARS["last_error"]),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
        "passed": state.get("passed"),
        "context_next_node": state.get("context_next_node", ""),
    }

    history_summary = state.get("history_summary") or history.get("content", "")
    history_summary_store = {
        "history_path": HISTORY_SUMMARY_FILE,
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history_summary, MAX_TEXT_CHARS["history_summary"]),
        "notepad_path": NOTEPAD_FILE,
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), MAX_TEXT_CHARS["notepad"]),
        "context_summary": _short_text(state.get("context_summary", ""), MAX_TEXT_CHARS["context_summary"]),
        "compression_events": state.get("compression_events", [])[-3:],
    }

    return {
        "rules": dict(RULES_LAYER),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def format_layered_memory_for_prompt(memory: dict[str, Any]) -> str:
    return json.dumps(memory, ensure_ascii=False, indent=2, default=str)


def memory_event(memory: dict[str, Any], *, node: str) -> dict[str, Any]:
    working = memory.get("working_memory", {})
    history = memory.get("history_summary_store", {})
    return {
        "type": "memory_snapshot",
        "node": node,
        "rules_count": len(memory.get("rules", {}).get("rules", [])),
        "todo_count": len(working.get("todos", [])),
        "notepad_exists": bool(history.get("notepad_exists")),
        "history_exists": bool(history.get("history_exists")),
        "history_path": history.get("history_path", HISTORY_SUMMARY_FILE),
        "layers": {
            "rules": _event_layer_summary(memory.get("rules", {})),
            "working_memory": _event_layer_summary(working),
            "history_summary_store": _event_layer_summary(history),
        },
    }


def read_history_summary(state: RuntimeState) -> dict[str, Any]:
    try:
        target = resolve_workspace_path(state.workspace, HISTORY_SUMMARY_FILE)
        if not target.exists():
            return {"ok": True, "path": HISTORY_SUMMARY_FILE, "content": "", "exists": False}
        content = target.read_text(encoding="utf-8")
        return {"ok": True, "path": HISTORY_SUMMARY_FILE, "content": content, "exists": True}
    except ValueError as e:
        return {"ok": False, "error": str(e), "exists": False}
    except Exception as e:
        return {"ok": False, "error": f"read history failed: {e}", "exists": False}


def persist_history_summary(state: RuntimeState, summary: str) -> dict[str, Any]:
    try:
        target = resolve_workspace_path(state.workspace, HISTORY_SUMMARY_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"# MokioClaw History Summary\n\n_Updated: {timestamp}_\n\n{summary.strip()}\n"
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": HISTORY_SUMMARY_FILE, "lines": len(content.splitlines())}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"persist history failed: {e}"}


def _event_layer_summary(layer: dict[str, Any]) -> str:
    if not layer:
        return "(empty)"
    return _short_text(json.dumps(layer, ensure_ascii=False, default=str), 420)


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
